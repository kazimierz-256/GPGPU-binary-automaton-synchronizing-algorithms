#define benchmark
using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea.CSharp;
using Alea;
using Alea.FSharp;
using System.Diagnostics;

namespace GPGPU
{
    public class SlimGPULeastSynchronizable : IComputable
    {
        private static readonly GlobalVariableSymbol<int> problemSize = Gpu.DefineConstantVariableSymbol<int>();

        public ComputationResult ComputeOne(Problem problemToSolve)
        => Compute(new[] { problemToSolve }, 1).First();

        public ComputationResult[] Compute(
            IEnumerable<Problem> problemsToSolve,
            int streamCount)
            => Compute(problemsToSolve, streamCount, null);

        public ComputationResult[] Compute(
            IEnumerable<Problem> problemsToSolve,
            int streamCount,
            Action asyncAction = null,
            int warpCount = 2)
        // cannot be more warps since more memory should be allocated
        {
#if (benchmark)
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();
#endif
            var gpu = Gpu.Default;
            var n = problemsToSolve.First().size;

            var power = 1 << n;
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            // in order to atomically add to a checking array (designed for queue consistency) one int is used by four threads, so in a reeeeaally pessimistic case 255 is the maximum number of threads (everyone go to the same vertex)
            var maximumWarps = gpu.Device.Attributes.MaxThreadsPerBlock / gpu.Device.Attributes.WarpSize;
            if (warpCount > maximumWarps)
                warpCount = maximumWarps;

            var problemsPerStream = (problemsToSolve.Count() + streamCount - 1) / streamCount;
            var problemsPartitioned = Enumerable.Range(0, streamCount)
                .Select(i => problemsToSolve.Skip(problemsPerStream * i)
                    .Take(problemsPerStream)
                    .ToArray())
                .Where(partition => partition.Length > 0)
                .ToArray();
            streamCount = problemsPartitioned.Length;
            var streams = Enumerable.Range(0, streamCount)
                .Select(_ => gpu.CreateStream()).ToArray();

            var gpuA = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var gpuB = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var shortestSynchronizingWordLength = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length)).ToArray();
            var isSynchronizable = problemsPartitioned.Select(problems => gpu.Allocate<bool>(problems.Length)).ToArray();
            gpu.Copy(n, problemSize);
            var queueUpperBound = power / 2 + 1;
            var launchParameters = new LaunchParam(
                new dim3(1, 1, 1),
                new dim3(gpu.Device.Attributes.WarpSize * warpCount, 1, 1),
                warpCount * (
                    sizeof(ushort) * queueUpperBound
                    + sizeof(ushort) * n * 2
                    + sizeof(byte) * power
                )
            );
            var gpuResultsIsSynchronizable = problemsPartitioned
                .Select(problems => new bool[problems.Length])
                .ToArray();
            var gpuResultsShortestSynchronizingWordLength = problemsPartitioned
                .Select(problems => new int[problems.Length])
                .ToArray();


            for (int stream = 0; stream < streamCount; stream++)
            {
                var problems = problemsPartitioned[stream];

                var matrixA = new int[problems.Length * n];
                var matrixB = new int[problems.Length * n];
                Parallel.For(0, problems.Length, problem =>
                {
                    Array.ConstrainedCopy(problems[problem].stateTransitioningMatrixA, 0, matrixA, problem * n, n);
                    Array.ConstrainedCopy(problems[problem].stateTransitioningMatrixB, 0, matrixB, problem * n, n);
                });

                streams[stream].Copy(matrixA, gpuA[stream]);
                streams[stream].Copy(matrixB, gpuB[stream]);

                streams[stream].Launch(
                    Kernel,
                    launchParameters,
                    gpuA[stream],
                    gpuB[stream],
                    isSynchronizable[stream],
                    shortestSynchronizingWordLength[stream]
                    );
            }

            asyncAction?.Invoke();

            for (int stream = 0; stream < streamCount; stream++)
            {
#if (benchmark)
                benchmarkTiming.Start();
#endif
                streams[stream].Synchronize();
#if (benchmark)
                benchmarkTiming.Stop();
#endif
                streams[stream].Copy(isSynchronizable[stream], gpuResultsIsSynchronizable[stream]);
                streams[stream].Copy(shortestSynchronizingWordLength[stream], gpuResultsShortestSynchronizingWordLength[stream]);
            }

            gpu.Synchronize();

#if (benchmark)
#endif
            var results = Enumerable.Range(0, streamCount).SelectMany(i => gpuResultsIsSynchronizable[i].Zip(gpuResultsShortestSynchronizingWordLength[i], (isSyncable, shortestWordLength)
                            => new ComputationResult()
                            {
                                computationType = ComputationType.GPU,
                                size = problemsToSolve.First().size,
                                isSynchronizable = isSyncable,
                                shortestSynchronizingWordLength = shortestWordLength,
                                algorithmName = GetType().Name
                            }
                ).ToArray()
            ).ToArray();

            foreach (var array in gpuA.AsEnumerable<Array>()
                .Concat(gpuB)
                .Concat(shortestSynchronizingWordLength)
                .Concat(isSynchronizable))
                Gpu.Free(array);

            foreach (var stream in streams)
                stream.Dispose();

            if (results.Any(result => result.isSynchronizable && result.shortestSynchronizingWordLength > maximumPermissibleWordLength))
                throw new Exception("Cerny conjecture is false");

#if (benchmark)
            results[0].benchmarkResult = new BenchmarkResult
            {
                benchmarkedTime = benchmarkTiming.Elapsed,
                totalTime = totalTiming.Elapsed
            };
#endif
            return results;
        }

        public static void Kernel(
            int[] precomputedStateTransitioningMatrixA,
            int[] precomputedStateTransitioningMatrixB,
            bool[] isSynchronizing,
            int[] shortestSynchronizingWordLength)
        {
            if (threadIdx.x % DeviceFunction.WarpSize != 0)
                return;

            var n = problemSize.Value;
            var arrayCount = precomputedStateTransitioningMatrixA.Length / n;
            var powerSetCount = 1 << n;
            var queueUpperBound = powerSetCount / 2 + 1;
            var initialVertex = (ushort)(powerSetCount - 1);
            var warpCount = blockDim.x / DeviceFunction.WarpSize;

            var threadOffset = threadIdx.x / DeviceFunction.WarpSize;

            #region Pointer setup
            var byteOffset = 0;

            var queueItself = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Ptr(threadOffset * queueUpperBound);
            byteOffset += sizeof(ushort) * queueUpperBound * warpCount;

            var gpuAs = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Ptr(threadOffset * n)
                .Volatile();
            byteOffset += sizeof(ushort) * n * warpCount;

            var gpuBs = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Ptr(threadOffset * n)
                .Volatile();
            byteOffset += sizeof(ushort) * n * warpCount;

            var isDiscovered = DeviceFunction.AddressOfArray(__shared__.ExternArray<byte>())
                .Ptr(byteOffset / sizeof(byte))
                .Ptr(threadOffset * powerSetCount);
            byteOffset += sizeof(byte) * powerSetCount * warpCount;
            #endregion
            int index;
            var acPart = (arrayCount + gridDim.x - 1) / gridDim.x;
            var acBegin = blockIdx.x * acPart;
            var acEnd = acBegin + acPart;
            if (arrayCount < acEnd)
                acEnd = arrayCount;
            index = acBegin * n;
            int queueReadingIndexModQueueSize = 0, queueWritingIndexModQueueSize = 0;
            int queueReadingIndexTotals = 0, queueWritingIndexTotals = 0;
            byte localCyclicProblemId = 1;
            for (int ac = acBegin; ac < acEnd; ac++, index += n, localCyclicProblemId++)
            {
                queueReadingIndexTotals = queueWritingIndexTotals = 0;
                queueReadingIndexModQueueSize = queueWritingIndexModQueueSize = 0;

                if (localCyclicProblemId == 0)
                {
                    for (int i = 0; i < powerSetCount; i++)
                        isDiscovered[i] = 0;

                    localCyclicProblemId = 1;
                }
                for (int i = 0; i < n; i++)
                {
                    gpuAs[i] = (ushort)(1 << precomputedStateTransitioningMatrixA[index + i]);
                    gpuBs[i] = (ushort)(1 << precomputedStateTransitioningMatrixB[index + i]);
                }

                // the queue is surely at least 2 vertices long...
                // no need for modulo operations
                queueItself[queueWritingIndexModQueueSize++] = initialVertex;

                var discoveredSingleton = false;
                ushort consideringVertex;
                ushort vertexAfterTransitionA;
                ushort vertexAfterTransitionB;
                ushort firstSingletonDistance = 0;

                ushort currentNextDistance = 1;
                var verticesUntilBump = int.MaxValue;
                var seekingFirstNext = true;

                // there is something to read
                while (queueWritingIndexModQueueSize > queueReadingIndexModQueueSize || queueWritingIndexTotals > queueReadingIndexTotals)
                {
                    consideringVertex = queueItself[queueReadingIndexModQueueSize++];
                    if (queueReadingIndexModQueueSize == queueUpperBound)
                    {
                        queueReadingIndexTotals++;
                        queueReadingIndexModQueueSize = 0;
                    }

                    if (--verticesUntilBump == 0)
                    {
                        ++currentNextDistance;
                        seekingFirstNext = true;
                    }

                    vertexAfterTransitionA = vertexAfterTransitionB = 0;

                    for (int i = 0, mask = 1; i < n; i++, mask <<= 1)
                    {
                        if (0 != (mask & consideringVertex))
                        {
                            vertexAfterTransitionA |= gpuAs[i];
                            vertexAfterTransitionB |= gpuBs[i];
                        }
                    }

                    if (localCyclicProblemId != isDiscovered[vertexAfterTransitionA])
                    {
                        if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                        {
                            discoveredSingleton = true;
                            firstSingletonDistance = currentNextDistance;
                            break;
                        }

                        isDiscovered[vertexAfterTransitionA] = localCyclicProblemId;
                        queueItself[queueWritingIndexModQueueSize++] = vertexAfterTransitionA;
                        if (queueWritingIndexModQueueSize == queueUpperBound)
                        {
                            queueWritingIndexTotals++;
                            queueWritingIndexModQueueSize = 0;
                        }

                        if (seekingFirstNext)
                        {
                            seekingFirstNext = false;
                            verticesUntilBump = (queueWritingIndexTotals - queueReadingIndexTotals) * queueUpperBound + (queueWritingIndexModQueueSize - queueReadingIndexModQueueSize);
                        }
                    }

                    if (localCyclicProblemId != isDiscovered[vertexAfterTransitionB])
                    {
                        if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                        {
                            discoveredSingleton = true;
                            firstSingletonDistance = currentNextDistance;
                            break;
                        }

                        isDiscovered[vertexAfterTransitionB] = localCyclicProblemId;
                        queueItself[queueWritingIndexModQueueSize++] = vertexAfterTransitionB;
                        if (queueWritingIndexModQueueSize == queueUpperBound)
                        {
                            queueWritingIndexTotals++;
                            queueWritingIndexModQueueSize = 0;
                        }

                        if (seekingFirstNext)
                        {
                            seekingFirstNext = false;
                            verticesUntilBump = (queueWritingIndexTotals - queueReadingIndexTotals) * queueUpperBound + (queueWritingIndexModQueueSize - queueReadingIndexModQueueSize);
                        }
                    }


                }
                if (discoveredSingleton)
                {
                    isSynchronizing[ac] = true;
                    shortestSynchronizingWordLength[ac] = firstSingletonDistance;
                }
            }
        }
        public int GetBestParallelism() => 16;
    }
}
