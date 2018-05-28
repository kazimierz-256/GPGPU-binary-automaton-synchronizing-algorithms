//#define benchmark
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
        public int GetBestParallelism() => 1;

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
            Action asyncAction = null)
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
            const int bitSize = 6;
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            // in order to atomically add to a checking array (designed for queue consistency) one int is used by four threads, so in a reeeeaally pessimistic case 255 is the maximum number of threads (everyone go to the same vertex)
            // -1 for the already discovered vertex, so its -2 in total for both reasons
            // at least 2*n+1 threads i.e. 27
            var threads = gpu.Device.Attributes.MaxThreadsPerBlock;
            var maximumThreads = Math.Min(
                gpu.Device.Attributes.MaxThreadsPerBlock,
                ((1 << bitSize) - 1) - 1
                );
            var minimumThreads = 2 * n + 1;
            if (threads > maximumThreads)
                threads = maximumThreads;
            if (threads < minimumThreads)
                threads = minimumThreads;
            if (threads > maximumThreads)
                throw new Exception("Impossible to satisfy");

            if (problemCount < streamCount)
                streamCount = problemCount;

            var problemsPerStream = (problemCount + streamCount - 1) / streamCount;
            var streams = Enumerable.Range(0, streamCount)
                .Select(_ => gpu.CreateStream()).ToArray();

            gpu.Copy(n, problemSize);

            var isDiscoveredComplexOffset = (power * sizeof(int) + bitSize) / (8 * sizeof(int) / bitSize);

            var launchParameters = new LaunchParam(
                new dim3(1 << 9, 1, 1),
                new dim3(threads, 1, 1),
                sizeof(int) * 3
                + isDiscoveredComplexOffset + (((isDiscoveredComplexOffset % sizeof(int)) & 1) == 1 ? 1 : 0)
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
                    Array.ConstrainedCopy(
                        problemsToSolve[problemsReadingIndex + offset + problem].stateTransitioningMatrixA,
                        0,
                        matrixA,
                        problem * n,
                        n
                        );
                    Array.ConstrainedCopy(
                        problemsToSolve[problemsReadingIndex + offset + problem].stateTransitioningMatrixB,
                        0,
                        matrixB,
                        problem * n,
                        n
                        );
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

#if (benchmark)
            gpu.Synchronize();
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
            Gpu.FreeAllImplicitMemory();

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
            var n = problemSize.Value;
            var arrayCount = precomputedStateTransitioningMatrixA.Length / n;
            var power = 1 << n;
            const int bitSize = 6;
            int twoToBitsize = (1 << bitSize) - 1;
            var wordCount = (8 * sizeof(int) / bitSize);

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

            // must be the last among ints
            var isDiscoveredPtr = DeviceFunction.AddressOfArray(__shared__.ExternArray<int>())
                .Ptr(byteOffset / sizeof(int));
            var complexOffset = (power * sizeof(int) + bitSize) / wordCount;
            byteOffset += complexOffset + (((complexOffset % sizeof(int)) & 1) == 1 ? 1 : 0);

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

            for (int ac = acBegin; ac < acEnd; ac++, index += n)
            {
                // cleanup
                for (int consideringVertex = threadIdx.x, endingVertex = (power - 1) / wordCount;
                    consideringVertex < endingVertex;
                    consideringVertex += blockDim.x)
                    isDiscoveredPtr[consideringVertex] = 0;

                if (threadIdx.x < n)
                {
                    gpuA[threadIdx.x] = (ushort)(1 << precomputedStateTransitioningMatrixA[index + threadIdx.x]);
                }
                else if (threadIdx.x < (n << 1))
                {
                    gpuB[threadIdx.x - n] = (ushort)(1 << precomputedStateTransitioningMatrixB[index + threadIdx.x - n]);
                }
                else if (threadIdx.x == (n << 1) + 1)
                {
                    readingQueueIndex[0] = 0;
                    shouldStop[0] = false;
                    queueEvenCount[0] = 0;
                    queueOddCount[0] = 1;
                    queueOdd[0] = (ushort)(power - 1);
                    // assuming n >= 2
                    isDiscoveredPtr[(power - 1) / wordCount] = 1 << (((power - 1) % wordCount) * bitSize);
                }

                var readingQueue = queueOdd;
                var writingQueue = queueEven;
                var readingQueueCount = queueOddCount;
                var writingQueueIndex = queueEvenCount;

                nextDistance = 1;
                int readingQueueCountCached = 1;
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

                        var isDiscoveredOffset = (vertexAfterTransitionA % wordCount) * bitSize;

                        if (0 == (isDiscoveredPtr[vertexAfterTransitionA / wordCount] & (twoToBitsize << isDiscoveredOffset)))
                        {
                            var beforeAdded = DeviceFunction.AtomicAdd(
                                isDiscoveredPtr.Ptr(vertexAfterTransitionA / wordCount),
                                1 << isDiscoveredOffset) & (twoToBitsize << isDiscoveredOffset);

                            if (0 == beforeAdded)
                            {
                                if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                                {
                                    shortestSynchronizingWordLength[ac] = nextDistance;
                                    isSynchronizing[ac] = true;
                                    shouldStop[0] = true;
                                    break;
                                }

                                writingQueue[DeviceFunction.AtomicAdd(writingQueueIndex, 1)]
                                    = (ushort)vertexAfterTransitionA;
                            }
                            else
                            {
                                DeviceFunction.AtomicSub(
                                    isDiscoveredPtr.Ptr(vertexAfterTransitionA / wordCount),
                                    1 << isDiscoveredOffset);
                            }
                        }

                        isDiscoveredOffset = (vertexAfterTransitionB % wordCount) * bitSize;
                        if (0 == (isDiscoveredPtr[vertexAfterTransitionB / wordCount] & (twoToBitsize << isDiscoveredOffset)))
                        {
                            var beforeAdded = DeviceFunction.AtomicAdd(
                                isDiscoveredPtr.Ptr(vertexAfterTransitionB / wordCount),
                                1 << isDiscoveredOffset) & (twoToBitsize << isDiscoveredOffset);

                            if (0 == beforeAdded)
                            {
                                if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                                {
                                    shortestSynchronizingWordLength[ac] = nextDistance;
                                    isSynchronizing[ac] = true;
                                    shouldStop[0] = true;
                                    break;
                                }

                                writingQueue[DeviceFunction.AtomicAdd(writingQueueIndex, 1)]
                                    = (ushort)vertexAfterTransitionB;
                            }
                            else
                            {
                                DeviceFunction.AtomicSub(
                                    isDiscoveredPtr.Ptr(vertexAfterTransitionB / wordCount),
                                    1 << isDiscoveredOffset);
                            }
                        }
                    }
                    DeviceFunction.SyncThreads();
                    ++nextDistance;
                    readingQueue = nextDistance % 2 == 0 ? queueEven : queueOdd;
                    writingQueue = nextDistance % 2 != 0 ? queueEven : queueOdd;
                    readingQueueCount = nextDistance % 2 == 0 ? queueEvenCount : queueOddCount;
                    writingQueueIndex = nextDistance % 2 != 0 ? queueEvenCount : queueOddCount;
                    readingQueueCountCached = nextDistance % 2 == 0 ? queueEvenCount[0] : queueOddCount[0];
                    if (threadIdx.x == 0)
                    {
                        writingQueueIndex[0] = 0;
                        readingQueueIndex[0] = 0;
                    }
                    DeviceFunction.SyncThreads();
                }
            }
        }
    }
}
