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
    public class SlimGPUQueue : IComputable
    {
        public ComputationResult ComputeOne(Problem problemToSolve)
        => Compute(new[] { problemToSolve }, 1)[0];

        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int streamCount)
            => Compute(problemsToSolve, streamCount, null);

        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int streamCount, Action asyncAction, int warps = 13)
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

            var takeRatio = (problemsToSolve.Count() + streamCount - 1) / streamCount;
            var problemsPartitioned = Enumerable.Range(0, streamCount)
                    .Select(i => problemsToSolve.Skip(takeRatio * i)
                        .Take(takeRatio)
                        .ToArray())
                    .Where(partition => partition.Length > 0)
                .ToArray();
            streamCount = problemsPartitioned.Length;

            var streams = Enumerable.Range(0, streamCount)
                .Select(_ => gpu.CreateStream()).ToArray();

            var precomputedStateTransitioningMatrixA = problemsPartitioned.Select(problems =>
            {
                var matrixA = new int[problems.Length * n];
                for (int problem = 0; problem < problems.Length; problem++)
                    for (int i = 0; i < n; i++)
                        matrixA[problem * n + i] = (ushort)(1 << problems[problem].stateTransitioningMatrixA[i]);
                return matrixA;
            }).ToArray();
            var precomputedStateTransitioningMatrixB = problemsPartitioned.Select(problems =>
            {
                var matrixB = new int[problems.Length * n];
                for (int problem = 0; problem < problems.Length; problem++)
                    for (int i = 0; i < n; i++)
                        matrixB[problem * n + i] = (ushort)(1 << problems[problem].stateTransitioningMatrixB[i]);
                return matrixB;
            }).ToArray();


            var gpuA = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var gpuB = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var shortestSynchronizingWordLength = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length)).ToArray();
            var isSynchronizable = problemsPartitioned.Select(problems => gpu.Allocate<bool>(problems.Length)).ToArray();
            var arrayCount = problemsPartitioned.Select(problems => gpu.Allocate(new[] { problems.Length })).ToArray();

            var launchParameters = new LaunchParam(
                new dim3(1, 1, 1),
                new dim3(DeviceArch.Default.WarpThreads * warps, 1, 1),
                sizeof(int) * 2 + 2 * sizeof(bool) + power * sizeof(bool) + power * sizeof(ushort) * 2
            );
#if (benchmark)
            benchmarkTiming.Start();
#endif
            for (var stream = 0; stream < streamCount; stream++)
            {
                streams[stream].Copy(precomputedStateTransitioningMatrixA[stream], gpuA[stream]);
                streams[stream].Copy(precomputedStateTransitioningMatrixB[stream], gpuB[stream]);
                streams[stream].Launch(
                    Kernel,
                    launchParameters,
                    arrayCount[stream],
                    gpuA[stream],
                    gpuB[stream],
                    isSynchronizable[stream],
                    shortestSynchronizingWordLength[stream]
                    );
            }

            asyncAction?.Invoke();

            gpu.Synchronize();

#if (benchmark)
            benchmarkTiming.Stop();
#endif
            var results = Enumerable.Range(0, streamCount).SelectMany(i
            => Gpu.CopyToHost(isSynchronizable[i])
                        .Zip(Gpu.CopyToHost(shortestSynchronizingWordLength[i]), (isSyncable, shortestWordLength)
                        => new ComputationResult()
                        {
                            size = problemsToSolve.First().size,
                            isSynchronizable = isSyncable,
                            shortestSynchronizingWordLength = shortestWordLength
                        }
                        ).ToArray()
            ).ToArray();

            foreach (var array in gpuA.AsEnumerable<Array>()
                .Concat(gpuB)
                .Concat(shortestSynchronizingWordLength)
                .Concat(isSynchronizable)
                .Concat(arrayCount))
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
            int[] arrayCount,
            int[] precomputedStateTransitioningMatrixA,
            int[] precomputedStateTransitioningMatrixB,
            bool[] isSynchronizing,
            int[] shortestSynchronizingWordLength)
        {
            var n = precomputedStateTransitioningMatrixA.Length / arrayCount[0];
            var power = 1 << n;

            var queueEvenCount = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(0);
            var queueOddCount = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(1);
            var shouldStop = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr(2 * 4).Volatile();
            var isDiscoveredPtr = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr(2 * 4 + 2).Volatile();
            var queueEven = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(2 * 2 + 1 + power / 2).Volatile();
            var queueOdd = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(2 * 2 + 1 + power / 2 + power).Volatile();

            if (threadIdx.x == 0)
            {
                queueOdd[0] = (ushort)(power - 1);
                isDiscoveredPtr[power - 1] = true;
                queueOddCount[0] = 1;
            }
            ushort nextDistance = 1;
            int vertexAfterTransitionA, vertexAfterTransitionB;

            var readingQueue = queueOdd;
            var writingQueue = queueEven;
            var readingQueueCount = queueOddCount;
            var writingQueueCount = queueEvenCount;
            int index, anotherIndex;
            DeviceFunction.SyncThreads();
            for (int ac = 0; ac < arrayCount[0]; ac++)
            {
                index = ac * n;
                while (readingQueueCount[0] > 0 && !shouldStop[0])
                {
                    int myPart = (readingQueueCount[0] + blockDim.x - 1) / blockDim.x;
                    int beginningPointer = threadIdx.x * myPart;
                    int endingPointer = (threadIdx.x + 1) * myPart;
                    if (readingQueueCount[0] < endingPointer)
                        endingPointer = readingQueueCount[0];

                    for (int iter = beginningPointer; iter < endingPointer; ++iter)
                    {
                        int consideringVertex = readingQueue[iter];
                        vertexAfterTransitionA = vertexAfterTransitionB = 0;
                        anotherIndex = index;
                        for (int i = 0; i < n; i++, anotherIndex++)
                        {
                            if (0 != ((1 << i) & consideringVertex))
                            {
                                vertexAfterTransitionA |=
                                    precomputedStateTransitioningMatrixA[anotherIndex];
                                vertexAfterTransitionB |=
                                   precomputedStateTransitioningMatrixB[anotherIndex];
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
                    DeviceFunction.SyncThreads();
                    if (threadIdx.x == 0)
                    {
                        readingQueueCount[0] = 0;
                    }
                    ++nextDistance;
                    readingQueue = nextDistance % 2 == 0 ? queueEven : queueOdd;
                    writingQueue = nextDistance % 2 != 0 ? queueEven : queueOdd;
                    readingQueueCount = nextDistance % 2 == 0 ? queueEvenCount : queueOddCount;
                    writingQueueCount = nextDistance % 2 != 0 ? queueEvenCount : queueOddCount;
                    DeviceFunction.SyncThreads();
                }

                if (ac < arrayCount[0] - 1)
                {
                    // cleanup
                    int myPart = (power + blockDim.x - 1) / blockDim.x;
                    int beginningPointer = threadIdx.x * myPart;
                    int endingPointer = (threadIdx.x + 1) * myPart;
                    if (power - 1 < endingPointer)
                        endingPointer = power;
                    for (int consideringVertex = beginningPointer; consideringVertex < endingPointer; consideringVertex++)
                    {
                        isDiscoveredPtr[consideringVertex] = false;
                    }
                    if (threadIdx.x == 0)
                    {
                        shouldStop[0] = false;
                        queueEvenCount[0] = 0;
                        queueOddCount[0] = 1;
                        queueOdd[0] = (ushort)(power - 1);
                    }
                    nextDistance = 1;
                    DeviceFunction.SyncThreads();
                }
            }
        }
        public int GetBestParallelism() => 4;
    }
}
