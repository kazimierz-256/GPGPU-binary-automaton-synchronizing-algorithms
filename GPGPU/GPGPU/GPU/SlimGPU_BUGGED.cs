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

namespace GPGPU
{
    public class SlimGPUBuggy : IComputable
    {
        public ComputationResult ComputeOne(Problem problemToSolve)
        => Compute(new[] { problemToSolve }, 1)[0];

        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int streamCount)
            => Compute(problemsToSolve, streamCount, null);

        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int streamCount, Action asyncAction, int warps = 32)
        {
            var gpu = Gpu.Default;
            var n = problemsToSolve.First().size;
            var power = 1 << n;
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            var takeRatio = (problemsToSolve.Count() + streamCount - 1) / streamCount;
            var problemsPartitioned = Enumerable.Range(0, streamCount)
                    .Select(i => problemsToSolve.Skip(takeRatio * i)
                    .Take(takeRatio)
                    .ToArray()).Where(partition => partition.Length > 0)
                .ToArray();
            streamCount = problemsPartitioned.Length;

            var streams = Enumerable.Range(0, streamCount).Select(_ => gpu.CreateStream()).ToArray();

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
                power * 3 + 1 + 1
            );

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

            var results = Enumerable.Range(0, streamCount).SelectMany(i =>
            {
                return Gpu.CopyToHost(isSynchronizable[i])
                        .Zip(Gpu.CopyToHost(shortestSynchronizingWordLength[i]), (isSyncable, shortestWordLength)
                    => new ComputationResult()
                    {
                        size = problemsToSolve.First().size,
                        computationType = ComputationType.GPU,
                        isSynchronizable = isSyncable,
                        shortestSynchronizingWordLength = shortestWordLength
                    }
                    ).ToArray();
            }).ToArray();

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

            var ptr = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>());
            var isDiscoveredPtr = ptr.Volatile();
            var isToBeProcessedDuringNextIteration = ptr.Ptr(power).Volatile();
            var isToBeProcessedDuringNextIterationOdd = ptr.Ptr(power * 2).Volatile();
            var shouldStop = ptr.Ptr(power * 3).Volatile();
            var addedSomethingThisRound = ptr.Ptr(power * 3 + 1).Volatile();
            if (threadIdx.x == 0)
                isToBeProcessedDuringNextIterationOdd[power - 1] = true;
            ushort nextDistance = 1;
            int vertexAfterTransitionA, vertexAfterTransitionB;
            int correctlyProcessed = 0;
            int myPart = (power + blockDim.x - 1) / blockDim.x;
            int beginningPointer = threadIdx.x * myPart;
            int endingPointer = (threadIdx.x + 1) * myPart;
            if (power < endingPointer)
                endingPointer = power;

            DeviceFunction.SyncThreads();
            for (int ac = 0; ac < arrayCount[0]; ac++)
            {
                {
                    // cleanup
                    for (int consideringVertex = beginningPointer; consideringVertex < endingPointer; consideringVertex++)
                    {
                        isDiscoveredPtr[consideringVertex] = false;
                        isToBeProcessedDuringNextIteration[consideringVertex] = false;
                        isToBeProcessedDuringNextIterationOdd[consideringVertex] = consideringVertex == power - 1;
                        shouldStop[0] = false;
                        correctlyProcessed = 0;
                        nextDistance = 1;
                        addedSomethingThisRound[0] = false;
                    }
                    DeviceFunction.SyncThreads();
                }
                while (correctlyProcessed < endingPointer - beginningPointer && !shouldStop[0])
                {
                    var nextIterationRead = nextDistance % 2 == 0 ? isToBeProcessedDuringNextIteration : isToBeProcessedDuringNextIterationOdd;
                    var nextIterationWrite = nextDistance % 2 != 0 ? isToBeProcessedDuringNextIteration : isToBeProcessedDuringNextIterationOdd;

                    for (int consideringVertex = beginningPointer; consideringVertex < endingPointer; consideringVertex++)
                    {
                        if (!nextIterationRead[consideringVertex])
                            continue;
                        else
                        {
                            nextIterationRead[consideringVertex] = false;
                            ++correctlyProcessed;
                        }

                        vertexAfterTransitionA = vertexAfterTransitionB = 0;

                        for (int i = 0; i < n; i++)
                        {
                            if (0 != ((1 << i) & consideringVertex))
                            {
                                vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[ac * n + i];
                                vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[ac * n + i];
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
                            nextIterationWrite[vertexAfterTransitionA] = true;
                            addedSomethingThisRound[0] = true;

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
                            nextIterationWrite[vertexAfterTransitionB] = true;
                            addedSomethingThisRound[0] = true;
                        }
                    }
                    ++nextDistance;
                    //TODO: efficiency: should count up idle threads, if it is maximum, it means the graph is not synchronizable
                    DeviceFunction.SyncThreads();
                    if (!addedSomethingThisRound[0])
                        break;
                    DeviceFunction.ThreadFenchBlock();
                    addedSomethingThisRound[0] = false;
                    DeviceFunction.SyncThreads();
                }
                
            }
        }
        public int GetBestParallelism() => 4;
    }
}
