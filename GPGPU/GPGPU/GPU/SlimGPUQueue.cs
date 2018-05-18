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
        // the algorithm performs best when using 8 or 16 streams
        public int GetBestParallelism() => 4;

        private static readonly GlobalVariableSymbol<int> problemSize = Gpu.DefineConstantVariableSymbol<int>();

        public void Compute(
            Problem[] problemsToSolve,
            int problemsReadingIndex,
            ComputationResult[] computationResults,
            int resultsWritingIndex,
            int problemCount,
            int streamCount)
            => ComputeAction(problemsToSolve, problemsReadingIndex, computationResults, resultsWritingIndex, problemCount, streamCount);

        public void ComputeAction(
            Problem[] problemsToSolve,
            int problemsReadingIndex,
            ComputationResult[] computationResults,
            int resultsWritingIndex,
            int problemCount,
            int streamCount,
            Action asyncAction = null,
            int warpCount = 7)
        // cannot be more warps since more memory should be allocated
        {
#if (benchmark)
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();
#endif
            var gpu = Gpu.Default;
            var n = problemsToSolve[problemsReadingIndex].size;

            var power = 1 << n;
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            // in order to atomically add to a checking array (designed for queue consistency) one int is used by four threads, so in a reeeeaally pessimistic case 255 is the maximum number of threads (everyone go to the same vertex)
            // -1 for the already discovered vertex
            var maximumWarps = Math.Min(
                gpu.Device.Attributes.MaxThreadsPerBlock / gpu.Device.Attributes.WarpSize,
                (255 - 1) / gpu.Device.Attributes.WarpSize
                );
            if (warpCount > maximumWarps)
                warpCount = maximumWarps;
            if (problemCount < streamCount)
                streamCount = problemCount;
            var problemsPerStream = (problemCount + streamCount - 1) / streamCount;
            var streams = Enumerable.Range(0, streamCount)
                .Select(_ => gpu.CreateStream()).ToArray();

            gpu.Copy(n, problemSize);

            var launchParameters = new LaunchParam(
                new dim3(1 << 9, 1, 1),
                new dim3(gpu.Device.Attributes.WarpSize * warpCount, 1, 1),
                sizeof(int) * 3
                + power * sizeof(int) / 4
                + (power / 2 + 1) * sizeof(ushort) * 2
                + 2 * n * sizeof(ushort)
                + sizeof(bool)
            );
            var gpuResultsIsSynchronizable = new bool[streamCount][];
            var gpuResultsShortestSynchronizingWordLength = new int[streamCount][];
            var gpuAs = new int[streamCount][];
            var gpuBs = new int[streamCount][];
            var shortestSynchronizingWordLength = new int[streamCount][];
            var isSynchronizable = new bool[streamCount][];

            for (int stream = 0; stream < streamCount; stream++)
            {
                var offset = stream * problemsPerStream;
                var localProblemsCount = Math.Min(problemsPerStream, problemCount - offset);

                gpuAs[stream] = gpu.Allocate<int>(localProblemsCount * n);
                gpuBs[stream] = gpu.Allocate<int>(localProblemsCount * n);
                shortestSynchronizingWordLength[stream] = gpu.Allocate<int>(localProblemsCount);
                isSynchronizable[stream] = gpu.Allocate<bool>(localProblemsCount);

                var matrixA = new int[localProblemsCount * n];
                var matrixB = new int[localProblemsCount * n];
                Parallel.For(0, localProblemsCount, problem =>
                {
                    Array.ConstrainedCopy(problemsToSolve[problemsReadingIndex + offset + problem].stateTransitioningMatrixA, 0, matrixA, problem * n, n);
                    Array.ConstrainedCopy(problemsToSolve[problemsReadingIndex + offset + problem].stateTransitioningMatrixB, 0, matrixB, problem * n, n);
                });

                streams[stream].Copy(matrixA, gpuAs[stream]);
                streams[stream].Copy(matrixB, gpuBs[stream]);

                gpuResultsIsSynchronizable[stream] = new bool[localProblemsCount];
                gpuResultsShortestSynchronizingWordLength[stream] = new int[localProblemsCount];

                streams[stream].Launch(
                    Kernel,
                    launchParameters,
                    gpuAs[stream],
                    gpuBs[stream],
                    isSynchronizable[stream],
                    shortestSynchronizingWordLength[stream]
                    );
            }

            asyncAction?.Invoke();

            var streamId = 0;
            foreach (var stream in streams)
            {
#if (benchmark)
                benchmarkTiming.Start();
#endif
                stream.Synchronize();
#if (benchmark)
                benchmarkTiming.Stop();
#endif
                stream.Copy(isSynchronizable[streamId], gpuResultsIsSynchronizable[streamId]);
                stream.Copy(shortestSynchronizingWordLength[streamId], gpuResultsShortestSynchronizingWordLength[streamId]);

                streamId++;
            }

            gpu.Synchronize();

#if (benchmark)
#endif

            Parallel.For(0, streamCount, stream =>
            {
                var offset = stream * problemsPerStream;
                var localProblemsCount = Math.Min(problemsPerStream, problemCount - offset);

                for (int i = 0; i < localProblemsCount; i++)
                {
                    computationResults[resultsWritingIndex + offset + i] = new ComputationResult()
                    {
                        computationType = ComputationType.GPU,
                        size = problemsToSolve[problemsReadingIndex].size,
                        isSynchronizable = gpuResultsIsSynchronizable[stream][i],
                        shortestSynchronizingWordLength = gpuResultsShortestSynchronizingWordLength[stream][i],
                        algorithmName = GetType().Name
                    };
                    if (gpuResultsIsSynchronizable[stream][i] && gpuResultsShortestSynchronizingWordLength[stream][i] > maximumPermissibleWordLength)
                        throw new Exception("Cerny conjecture is false");
                }
            });

            foreach (var arrays in new IEnumerable<Array>[] { gpuAs, gpuBs, isSynchronizable, shortestSynchronizingWordLength })
                foreach (var array in arrays)
                    Gpu.Free(array);

            foreach (var stream in streams)
                stream.Dispose();

#if (benchmark)
            computationResults[resultsWritingIndex].benchmarkResult = new BenchmarkResult
            {
                benchmarkedTime = benchmarkTiming.Elapsed,
                totalTime = totalTiming.Elapsed
            };
#endif
        }

        public static void Kernel(
            int[] precomputedStateTransitioningMatrixA,
            int[] precomputedStateTransitioningMatrixB,
            bool[] isSynchronizing,
            int[] shortestSynchronizingWordLength)
        {
            const int skipEvery = 1;
            if (threadIdx.x % skipEvery != 0)
                return;
            var n = problemSize.Value;
            var arrayCount = precomputedStateTransitioningMatrixA.Length / n;
            var power = 1 << n;

            #region Pointer setup
            var byteOffset = 0;

            var queueEvenCount = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                   .Ptr(byteOffset / sizeof(int));
            byteOffset += sizeof(int);

            var readingQueueIndex = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                   .Ptr(byteOffset / sizeof(int));
            byteOffset += sizeof(int);

            var queueOddCount = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(byteOffset / sizeof(int));
            byteOffset += sizeof(int);

            var isDiscoveredPtr = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(byteOffset / sizeof(int));
            byteOffset += power * sizeof(int) / 4;

            var queueEven = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Volatile();
            byteOffset += (power / 2 + 1) * sizeof(ushort);

            var queueOdd = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Volatile();
            byteOffset += (power / 2 + 1) * sizeof(ushort);


            var gpuA = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Volatile();
            byteOffset += n * sizeof(ushort);

            var gpuB = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Volatile();
            byteOffset += n * sizeof(ushort);

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
            DeviceFunction.SyncThreads();
            for (int ac = acBegin; ac < acEnd; ac++, index += n)
            {
                // cleanup
                for (int consideringVertex = threadIdx.x / skipEvery, endingVertex = (power - 1) >> 2;
                    consideringVertex < endingVertex;
                    consideringVertex += (blockDim.x / skipEvery))
                    isDiscoveredPtr[consideringVertex] = 0;

                if (threadIdx.x == 0)
                {
                    shouldStop[0] = false;
                    queueEvenCount[0] = 0;
                    queueOddCount[0] = 1;
                    queueOdd[0] = (ushort)(power - 1);
                    // assuming n >= 2
                    isDiscoveredPtr[(power - 1) >> 2] = 1 << 24;
                    for (int i = 0; i < n; i++)
                    {
                        gpuA[i] = (ushort)(1 << precomputedStateTransitioningMatrixA[index + i]);
                        gpuB[i] = (ushort)(1 << precomputedStateTransitioningMatrixB[index + i]);
                    }
                }
                var readingQueue = queueOdd;
                var writingQueue = queueEven;
                var readingQueueCount = queueOddCount;
                var writingQueueCount = queueEvenCount;

                nextDistance = 1;
                int readingQueueCountCached = 1;
                readingQueueIndex[0] = 0;
                DeviceFunction.SyncThreads();

                while (readingQueueCountCached > 0 && !shouldStop[0])
                {
                    //Console.WriteLine("ac {3}, threadix {0}, begin {1}, end {2}", threadIdx.x, beginningPointer, endingPointer, ac);
                    while (readingQueueIndex[0] < readingQueueCountCached)
                    {
                        var iter = DeviceFunction.AtomicAdd(readingQueueIndex, 1);
                        if (iter >= readingQueueCountCached)
                            break;

                        int consideringVertex = readingQueue[iter];

                        vertexAfterTransitionA = vertexAfterTransitionB = 0;
                        for (int i = 0, mask = 1; i < n; i++, mask <<= 1)
                        {
                            if (0 != (mask & consideringVertex))
                            {
                                vertexAfterTransitionA |= gpuA[i];
                                vertexAfterTransitionB |= gpuB[i];
                            }
                        }

                        var eightTimesRemainder = (vertexAfterTransitionA % 4) << 3;

                        if (0 == (isDiscoveredPtr[vertexAfterTransitionA >> 2] & (255 << eightTimesRemainder)))
                        {
                            var beforeAdded = DeviceFunction.AtomicAdd(isDiscoveredPtr.Ptr(vertexAfterTransitionA >> 2), 1 << eightTimesRemainder) & (255 << eightTimesRemainder);
                            if (0 == beforeAdded)
                            {
                                if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                                {
                                    shortestSynchronizingWordLength[ac] = nextDistance;
                                    isSynchronizing[ac] = true;
                                    shouldStop[0] = true;
                                    break;
                                }

                                var writeToPointer = DeviceFunction.AtomicAdd(writingQueueCount, 1);
                                writingQueue[writeToPointer] = (ushort)vertexAfterTransitionA;
                            }
                            else
                            {
                                DeviceFunction.AtomicSub(isDiscoveredPtr.Ptr(vertexAfterTransitionA >> 2), 1 << eightTimesRemainder);
                            }
                        }

                        eightTimesRemainder = (vertexAfterTransitionB % 4) << 3;
                        if (0 == (isDiscoveredPtr[vertexAfterTransitionB >> 2] & (255 << eightTimesRemainder)))
                        {
                            var beforeAdded = DeviceFunction.AtomicAdd(isDiscoveredPtr.Ptr(vertexAfterTransitionB >> 2), 1 << eightTimesRemainder) & (255 << eightTimesRemainder);
                            if (0 == beforeAdded)
                            {
                                if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                                {
                                    shortestSynchronizingWordLength[ac] = nextDistance;
                                    isSynchronizing[ac] = true;
                                    shouldStop[0] = true;
                                    break;
                                }

                                var writeToPointer = DeviceFunction.AtomicAdd(writingQueueCount, 1);
                                writingQueue[writeToPointer] = (ushort)vertexAfterTransitionB;
                            }
                            else
                            {
                                DeviceFunction.AtomicSub(isDiscoveredPtr.Ptr(vertexAfterTransitionB >> 2), 1 << eightTimesRemainder);
                            }
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
                    {
                        writingQueueCount[0] = 0;
                        readingQueueIndex[0] = 0;
                    }
                    DeviceFunction.SyncThreads();
                }
            }
        }
    }
}
