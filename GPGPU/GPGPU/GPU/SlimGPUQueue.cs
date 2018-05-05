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
        private static readonly GlobalVariableSymbol<int> problemSize = Gpu.DefineConstantVariableSymbol<int>();

        public ComputationResult ComputeOne(Problem problemToSolve)
        => Compute(new[] { problemToSolve }, 1)[0];

        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int streamCount)
            => Compute(problemsToSolve, streamCount, null);

        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int streamCount, Action asyncAction = null, int warps = 13, int problemsPerStream = 0b1_0000_0000_0000_000)
        {
#if (benchmark)
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();
#endif
            var gpu = Gpu.Default;
            var n = problemsToSolve.First().size;
            if (problemsToSolve.Any(problem => problem.size != n))
                throw new Exception("Inconsistent problem sizes");

            var power = 1 << n;
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            var partitionCount = (problemsToSolve.Count() + problemsPerStream - 1) / problemsPerStream;
            var problemsPartitioned = Enumerable.Range(0, partitionCount)
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
                new dim3(DeviceArch.Default.WarpThreads * warps, 1, 1),
                sizeof(int) * 2 + sizeof(bool) * 2 + power * sizeof(byte) + power * sizeof(ushort) * 2
            );
#if (benchmark)
            benchmarkTiming.Start();
#endif
            var gpuResultsIsSynchronizable = problemsPartitioned
                .Select(problems => new bool[problems.Length])
                .ToArray();
            var gpuResultsShortestSynchronizingWordLength = problemsPartitioned
                .Select(problems => new int[problems.Length])
                .ToArray();

            var streamToRecover = new Queue<KeyValuePair<int, int>>(streamCount);

            for (int partition = 0; partition < partitionCount; partition++)
            {
                var problems = problemsPartitioned[partition];

                var matrixA = new int[problems.Length * n];
                for (int problem = 0; problem < problems.Length; problem++)
                    for (int i = 0; i < n; i++)
                        matrixA[problem * n + i] = (ushort)(1 << problems[problem].stateTransitioningMatrixA[i]);

                var matrixB = new int[problems.Length * n];
                for (int problem = 0; problem < problems.Length; problem++)
                    for (int i = 0; i < n; i++)
                        matrixB[problem * n + i] = (ushort)(1 << problems[problem].stateTransitioningMatrixB[i]);

                var stream = partition % streamCount;
                var reusingStream = partition >= streamCount;
                if (reusingStream)
                {
                    streams[stream].Synchronize();
                    streams[stream].Copy(isSynchronizable[stream], gpuResultsIsSynchronizable[partition - streamCount]);
                    streams[stream].Copy(shortestSynchronizingWordLength[stream], gpuResultsShortestSynchronizingWordLength[partition - streamCount]);
                    Gpu.FreeAllImplicitMemory();
                    streamToRecover.Dequeue();
                }
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
                streamToRecover.Enqueue(new KeyValuePair<int, int>(stream, partition));
            }

            foreach (var streamPartitionKVP in streamToRecover)
            {
                streams[streamPartitionKVP.Key].Synchronize();
                streams[streamPartitionKVP.Key].Copy(isSynchronizable[streamPartitionKVP.Key], gpuResultsIsSynchronizable[streamPartitionKVP.Value]);
                streams[streamPartitionKVP.Key].Copy(shortestSynchronizingWordLength[streamPartitionKVP.Key], gpuResultsShortestSynchronizingWordLength[streamPartitionKVP.Value]);
            }
            Gpu.FreeAllImplicitMemory();

            //gpu.Synchronize();

#if (benchmark)
            benchmarkTiming.Stop();
#endif
            var results = Enumerable.Range(0, partitionCount).SelectMany(i => gpuResultsIsSynchronizable[i].Zip(gpuResultsShortestSynchronizingWordLength[i], (isSyncable, shortestWordLength)
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
                .Concat(isSynchronizable))
                Gpu.Free(array);

            foreach (var stream in streams)
                stream.Dispose();

            if (results.Any(result => result.isSynchronizable && result.shortestSynchronizingWordLength > maximumPermissibleWordLength))
                throw new Exception("Cerny conjecture is false");
            //Console.WriteLine(results[0].isSynchronizable);
            //Console.WriteLine(results[1].isSynchronizable);
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

            var queueEvenCount = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(0);
            var queueOddCount = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(sizeof(int) / sizeof(int));
            var shouldStop = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr((2 * sizeof(int)) / sizeof(bool)).Volatile();
            var isDiscoveredPtr = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>())
                .Ptr((2 * sizeof(int) + sizeof(bool)) / sizeof(bool)).Volatile();
            var queueEven = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr((2 * sizeof(int) + sizeof(bool) * 2 + power * sizeof(bool)) / sizeof(ushort)).Volatile();
            var queueOdd = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr((2 * sizeof(int) + sizeof(bool) * 2 + power * sizeof(bool) + power * sizeof(ushort)) / sizeof(ushort)).Volatile();

            if (threadIdx.x == 0)
                isDiscoveredPtr[power - 1] = true;

            ushort nextDistance;
            int vertexAfterTransitionA,
                vertexAfterTransitionB,
                index = 0;

            DeviceFunction.SyncThreads();
            for (int ac = 0; ac < arrayCount; ac++)
            {
                // cleanup
                if (ac > 0)
                {
                    int myPart = (power + blockDim.x - 1) / blockDim.x;
                    int beginningPointer = threadIdx.x * myPart;
                    int endingPointer = (threadIdx.x + 1) * myPart;
                    if (power - 1 < endingPointer)
                        endingPointer = power - 1;
                    for (int consideringVertex = beginningPointer; consideringVertex < endingPointer; consideringVertex++)
                    {
                        isDiscoveredPtr[consideringVertex] = false;
                    }
                }
                if (threadIdx.x == 0)
                {
                    shouldStop[0] = false;
                    queueEvenCount[0] = 0;
                    queueOddCount[0] = 1;
                    queueOdd[0] = (ushort)(power - 1);

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
                    int endingPointer = (threadIdx.x + 1) * myPart;
                    if (readingQueueCountCached < endingPointer)
                        endingPointer = readingQueueCountCached;
                    for (int iter = beginningPointer; iter < endingPointer; ++iter)
                    {
                        int consideringVertex = readingQueue[iter];
                        vertexAfterTransitionA = vertexAfterTransitionB = 0;
                        for (int i = 0; i < n; i++)
                        {
                            if (0 != ((1 << i) & consideringVertex))
                            {
                                vertexAfterTransitionA |=
                                    precomputedStateTransitioningMatrixA[index + i];
                                vertexAfterTransitionB |=
                                    precomputedStateTransitioningMatrixB[index + i];
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
                index += n;
            }
        }
        public int GetBestParallelism() => 4;
    }
}
