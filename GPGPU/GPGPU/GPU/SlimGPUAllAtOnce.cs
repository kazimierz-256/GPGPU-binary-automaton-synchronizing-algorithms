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
    public class SlimGPUAllAtOnce : IComputable
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
            int warpCount = 8)
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

            var launchParameters = new LaunchParam(
                new dim3(1, 1, 1),
                new dim3(gpu.Device.Attributes.WarpSize * warpCount, 1, 1),
                sizeof(ushort) * n * 2
                + sizeof(bool) * (power * 3)
                + sizeof(int) * gpu.Device.Attributes.WarpSize * warpCount * 4
                + sizeof(bool) * 3
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
                                shortestSynchronizingWordLength = shortestWordLength
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
            //Console.WriteLine(results[0].isSynchronizable);
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
            var n = problemSize.Value;
            var arrayCount = precomputedStateTransitioningMatrixA.Length / n;
            var power = 1 << n;

            #region Pointer setup
            var byteOffset = 0;

            var minEven = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(byteOffset / sizeof(int));
            byteOffset += blockDim.x * sizeof(int);

            var minOdd = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(byteOffset / sizeof(int));
            byteOffset += blockDim.x * sizeof(int);

            var maxEven = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(byteOffset / sizeof(int));
            byteOffset += blockDim.x * sizeof(int);

            var maxOdd = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(byteOffset / sizeof(int));
            byteOffset += blockDim.x * sizeof(int);

            var gpuA = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Volatile();
            byteOffset += n * sizeof(ushort);

            var gpuB = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Volatile();
            byteOffset += n * sizeof(ushort);

            var isActiveEven = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                   .Ptr(byteOffset / sizeof(bool));
            byteOffset += power * sizeof(bool);

            var isActiveOdd = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr(byteOffset / sizeof(bool));
            byteOffset += power * sizeof(bool);

            var isDiscovered = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr(byteOffset / sizeof(bool));
            byteOffset += power * sizeof(bool);

            var addedAnythingOdd = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr(byteOffset / sizeof(bool))
                .Volatile();
            byteOffset += sizeof(bool);

            var addedAnythingEven = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr(byteOffset / sizeof(bool))
                .Volatile();
            byteOffset += sizeof(bool);

            var shouldStop = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr(byteOffset / sizeof(bool))
                .Volatile();
            byteOffset += sizeof(bool);
            #endregion


            ushort nextDistance;
            int vertexAfterTransitionA,
                vertexAfterTransitionB,
                index;
            var acPart = (arrayCount + gridDim.x - 1) / gridDim.x;
            var acBegin = blockIdx.x * acPart;
            var acEnd = acBegin + acPart;
            if (arrayCount < acEnd)
                acEnd = arrayCount;
            index = acBegin * n;
            var threadWork = (power + blockDim.x - 1) / blockDim.x;
            DeviceFunction.SyncThreads();
            for (int ac = acBegin; ac < acEnd; ac++, index += n)
            {
                //Console.WriteLine("Begin {0}", threadIdx.x);
                // cleanup
                var readingActive = isActiveOdd;
                var writingActive = isActiveEven;
                var readingAnythingAdded = addedAnythingOdd;
                var writingAnythingAdded = addedAnythingEven;
                var minRead = minOdd;
                var minWrite = minEven;
                var maxRead = maxOdd;
                var maxWrite = maxEven;

                if (threadIdx.x == DeviceFunction.WarpSize || (threadIdx.x == 0 && blockDim.x <= DeviceFunction.WarpSize))
                    for (int i = 0; i < n; i++)
                    {
                        gpuA[i] = (ushort)(1 << precomputedStateTransitioningMatrixA[index + i]);
                        gpuB[i] = (ushort)(1 << precomputedStateTransitioningMatrixB[index + i]);
                    }

                minRead[threadIdx.x] = int.MaxValue;
                maxRead[threadIdx.x] = 0;
                minWrite[threadIdx.x] = int.MaxValue;
                maxWrite[threadIdx.x] = 0;
                if (threadIdx.x == 0)
                {
                    shouldStop[0] = false;
                    readingAnythingAdded[0] = true;
                }
                else if (threadIdx.x == blockDim.x - 1)
                {
                    minRead[blockDim.x - 1] = (power - 1) % threadWork;
                    maxRead[blockDim.x - 1] = power % threadWork;
                }
                nextDistance = 1;
                for (int consideringVertex = threadIdx.x, endingVertex = power, powerm1 = power - 1;
                    consideringVertex < endingVertex;
                    consideringVertex += blockDim.x)
                {
                    isActiveEven[consideringVertex] = false;
                    isActiveOdd[consideringVertex] = consideringVertex == powerm1;
                    isDiscovered[consideringVertex] = consideringVertex == powerm1;
                    //Console.WriteLine("cleaning {0}, {1} {2} {3}", consideringVertex,
                    //    readingActive[consideringVertex], writingActive[consideringVertex], isDiscovered[consideringVertex]);
                }
                DeviceFunction.SyncThreads();

                while (readingAnythingAdded[0] && !shouldStop[0])
                {
                    if (threadIdx.x == 0)
                    {
                        readingAnythingAdded[0] = false;
                        writingAnythingAdded[0] = false;
                        //Console.WriteLine("distance {0}", nextDistance);
                    }

                    int myPart = (power + blockDim.x - 1) / blockDim.x;
                    int beginningPointer = threadIdx.x * myPart;
                    int endingPointer = beginningPointer + myPart;
                    if (power < endingPointer)
                        endingPointer = power;
                    if (minRead[threadIdx.x] > beginningPointer)
                        beginningPointer = minRead[threadIdx.x];
                    //if (maxRead[threadIdx.x] < endingPointer)
                    //    endingPointer = maxRead[threadIdx.x];
                    minRead[threadIdx.x] = int.MaxValue;
                    maxRead[threadIdx.x] = 0;
                    minWrite[threadIdx.x] = int.MaxValue;
                    maxWrite[threadIdx.x] = 0;
                    DeviceFunction.SyncThreads();
                    //Console.WriteLine("ac {3}, threadix {0}, begin {1}, end {2}", threadIdx.x, beginningPointer, endingPointer, ac);
                    for (int consideringVertex = beginningPointer; consideringVertex < endingPointer; ++consideringVertex)
                    {
                        //Console.WriteLine("writeIsActive[{0}]=={1}", consideringVertex, writingActive[consideringVertex]);
                        if (!readingActive[consideringVertex])
                        {
                            //Console.WriteLine("Skipping {0}", consideringVertex);
                            continue;
                        }
                        else
                        {
                            //Console.WriteLine("Considering {0} dist {1}", consideringVertex, nextDistance);
                            readingActive[consideringVertex] = false;
                        }

                        vertexAfterTransitionA = vertexAfterTransitionB = 0;
                        for (int i = 0, mask = 1; i < n; i++, mask <<= 1)
                        {
                            if (0 != (mask & consideringVertex))
                            {
                                vertexAfterTransitionA |= gpuA[i];
                                vertexAfterTransitionB |= gpuB[i];
                            }
                        }

                        if (!isDiscovered[vertexAfterTransitionA])
                        {
                            isDiscovered[vertexAfterTransitionA] = true;
                            //Console.WriteLine("Discovered {0} by {1}", vertexAfterTransitionA, consideringVertex);
                            DeviceFunction.AtomicMin(minWrite.Ptr(vertexAfterTransitionA / threadWork), vertexAfterTransitionA % threadWork);
                            DeviceFunction.AtomicMax(maxWrite.Ptr(vertexAfterTransitionA / threadWork), 1 + (vertexAfterTransitionA % threadWork));

                            if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                            {
                                shortestSynchronizingWordLength[ac] = nextDistance;
                                isSynchronizing[ac] = true;
                                shouldStop[0] = true;
                                break;
                            }

                            writingActive[vertexAfterTransitionA] = true;
                            writingAnythingAdded[0] = true;
                        }

                        if (!isDiscovered[vertexAfterTransitionB])
                        {
                            isDiscovered[vertexAfterTransitionB] = true;
                            //Console.WriteLine("Discovered {0} by {1}", vertexAfterTransitionB, consideringVertex);
                            DeviceFunction.AtomicMin(minWrite.Ptr(vertexAfterTransitionB / threadWork), vertexAfterTransitionB % threadWork);
                            DeviceFunction.AtomicMax(maxWrite.Ptr(vertexAfterTransitionB / threadWork), 1 + (vertexAfterTransitionB % threadWork));

                            if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                            {
                                shortestSynchronizingWordLength[ac] = nextDistance;
                                isSynchronizing[ac] = true;
                                shouldStop[0] = true;
                                break;
                            }

                            writingActive[vertexAfterTransitionB] = true;
                            writingAnythingAdded[0] = true;
                        }
                    }
                    ++nextDistance;
                    readingActive = nextDistance % 2 == 0 ? isActiveEven : isActiveOdd;
                    writingActive = nextDistance % 2 != 0 ? isActiveEven : isActiveOdd;
                    readingAnythingAdded = nextDistance % 2 == 0 ? addedAnythingEven : addedAnythingOdd;
                    writingAnythingAdded = nextDistance % 2 != 0 ? addedAnythingEven : addedAnythingOdd;
                    minRead = nextDistance % 2 == 0 ? minEven : minOdd;
                    minWrite = nextDistance % 2 != 0 ? minEven : minOdd;
                    maxRead = nextDistance % 2 == 0 ? maxEven : maxOdd;
                    maxWrite = nextDistance % 2 != 0 ? maxEven : maxOdd;
                    DeviceFunction.SyncThreads();
                }
            }
        }
        public int GetBestParallelism() => 16;
    }
}
