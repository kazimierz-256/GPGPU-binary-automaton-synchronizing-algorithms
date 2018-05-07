﻿#define benchmark
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
    public class SlimGPUQueue : IComputable
    {
        private static readonly GlobalVariableSymbol<int> problemSize = Gpu.DefineConstantVariableSymbol<int>();

        public ComputationResult ComputeOne(Problem problemToSolve)
        => Compute(new[] { problemToSolve }, 1)[0];

        public ComputationResult[] Compute(
            IEnumerable<Problem> problemsToSolve,
            int streamCount)
            => Compute(problemsToSolve, streamCount, null);

        public ComputationResult[] Compute(
            IEnumerable<Problem> problemsToSolve,
            int streamCount,
            Action asyncAction = null,
            int warps = 12)
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
            if (warps > maximumWarps)
                warps = maximumWarps;

            var problemsPerStream = (problemsToSolve.Count() + streamCount - 1) / streamCount;
            var problemsPartitioned = Enumerable.Range(0, streamCount)
                .Select(i => problemsToSolve.Skip(problemsPerStream * i)
                    .Take(problemsPerStream)
                    .ToArray())
                .Where(partition => partition.Length > 0)
                .ToArray();

            var streams = Enumerable.Range(0, streamCount)
                .Select(_ => gpu.CreateStream()).ToArray();

            var gpuA = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var gpuB = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var shortestSynchronizingWordLength = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length)).ToArray();
            var isSynchronizable = problemsPartitioned.Select(problems => gpu.Allocate<bool>(problems.Length)).ToArray();
            gpu.Copy(n, problemSize);

            var launchParameters = new LaunchParam(
                new dim3(1, 1, 1),
                new dim3(gpu.Device.Attributes.WarpSize * warps, 1, 1),
                sizeof(int) * 2 + sizeof(bool) * 2 + power * sizeof(bool) + (power / 2 + 1) * sizeof(ushort) * 2 + 2 * n * sizeof(ushort)
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
                streams[stream].Copy(isSynchronizable[stream], gpuResultsIsSynchronizable[stream]);
#if (benchmark)
                benchmarkTiming.Stop();
#endif
                streams[stream].Copy(shortestSynchronizingWordLength[stream], gpuResultsShortestSynchronizingWordLength[stream]);
            }

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
            var queueEvenCount = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                   .Ptr(0);

            var queueOddCount = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(sizeof(int) / sizeof(int));

            var shouldStop = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr((2 * sizeof(int)) / sizeof(bool))
                .Volatile();

            var isDiscoveredPtr = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr((2 * sizeof(int) + sizeof(bool)) / sizeof(bool))
                .Volatile();

            var queueEven = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr((2 * sizeof(int) + sizeof(bool) * 2 + power * sizeof(bool)) / sizeof(ushort))
                .Volatile();

            var queueOdd = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr((2 * sizeof(int) + sizeof(bool) * 2 + power * sizeof(bool) + (power / 2 + 1) * sizeof(ushort)) / sizeof(ushort))
                .Volatile();

            var gpuA = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr((2 * sizeof(int) + sizeof(bool) * 2 + power * sizeof(bool) + (power / 2 + 1) * sizeof(ushort) * 2) / sizeof(ushort))
                .Volatile();

            var gpuB = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr((2 * sizeof(int) + sizeof(bool) * 2 + power * sizeof(bool) + (power / 2 + 1) * sizeof(ushort) * 2 + n * sizeof(ushort)) / sizeof(ushort))
                .Volatile();
            #endregion

            if (threadIdx.x == 0)
                isDiscoveredPtr[power - 1] = true;

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
            DeviceFunction.SyncThreads();
            for (int ac = acBegin; ac < acEnd; ac++, index += n)
            {
                // cleanup
                for (int consideringVertex = threadIdx.x, endingVertex = power - 1;
                    consideringVertex < endingVertex;
                    consideringVertex += blockDim.x)
                    isDiscoveredPtr[consideringVertex] = false;

                if (threadIdx.x == 0)
                {
                    shouldStop[0] = false;
                    queueEvenCount[0] = 0;
                    queueOddCount[0] = 1;
                    queueOdd[0] = (ushort)(power - 1);

                }
                if (threadIdx.x == DeviceFunction.WarpSize || (threadIdx.x == 0 && blockDim.x <= DeviceFunction.WarpSize))
                    for (int i = 0; i < n; i++)
                    {
                        gpuA[i] = (ushort)(1 << precomputedStateTransitioningMatrixA[index + i]);
                        gpuB[i] = (ushort)(1 << precomputedStateTransitioningMatrixB[index + i]);
                    }
                var readingQueue = queueOdd;
                var writingQueue = queueEven;
                var readingQueueCount = queueOddCount;
                var writingQueueCount = queueEvenCount;

                nextDistance = 1;
                int readingQueueCountCached = 1;
                DeviceFunction.SyncThreads();

                while (readingQueueCountCached > 0 && !shouldStop[0])
                {
                    int myPart = (readingQueueCountCached + blockDim.x - 1) / blockDim.x;
                    int beginningPointer = threadIdx.x * myPart;
                    int endingPointer = beginningPointer + myPart;
                    if (readingQueueCountCached < endingPointer)
                        endingPointer = readingQueueCountCached;
                    //Console.WriteLine("ac {3}, threadix {0}, begin {1}, end {2}", threadIdx.x, beginningPointer, endingPointer, ac);
                    for (int iter = beginningPointer; iter < endingPointer; ++iter)
                    {
                        int consideringVertex = readingQueue[iter];
                        vertexAfterTransitionA = vertexAfterTransitionB = 0;
                        for (int i = 0; i < n; i++)
                        {
                            if (0 != ((1 << i) & consideringVertex))
                            {
                                vertexAfterTransitionA |= gpuA[i];
                                vertexAfterTransitionB |= gpuB[i];
                            }
                        }
                        if (!isDiscoveredPtr[vertexAfterTransitionA])
                        {
                            if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                            {
                                shortestSynchronizingWordLength[ac] = nextDistance;
                                isSynchronizing[ac] = true;
                                shouldStop[0] = true;
                                break;
                            }

                            isDiscoveredPtr[vertexAfterTransitionA] = true;
                            var writeToPointer = DeviceFunction.AtomicAdd(writingQueueCount, 1);
                            writingQueue[writeToPointer] = (ushort)vertexAfterTransitionA;
                        }

                        if (!isDiscoveredPtr[vertexAfterTransitionB])
                        {
                            if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                            {
                                shortestSynchronizingWordLength[ac] = nextDistance;
                                isSynchronizing[ac] = true;
                                shouldStop[0] = true;
                                break;
                            }

                            isDiscoveredPtr[vertexAfterTransitionB] = true;
                            var writeToPointer = DeviceFunction.AtomicAdd(writingQueueCount, 1);
                            writingQueue[writeToPointer] = (ushort)vertexAfterTransitionB;
                        }
                    }
                    ++nextDistance;
                    readingQueue = nextDistance % 2 == 0 ? queueEven : queueOdd;
                    writingQueue = nextDistance % 2 != 0 ? queueEven : queueOdd;
                    readingQueueCount = nextDistance % 2 == 0 ? queueEvenCount : queueOddCount;
                    writingQueueCount = nextDistance % 2 != 0 ? queueEvenCount : queueOddCount;
                    DeviceFunction.SyncThreads();
                    readingQueueCountCached = nextDistance % 2 == 0 ? queueEvenCount[0] : queueOddCount[0];
                    if (threadIdx.x == 0)
                        writingQueueCount[0] = 0;
                    DeviceFunction.SyncThreads();
                }
            }
        }
        public int GetBestParallelism() => 16;
    }
}
