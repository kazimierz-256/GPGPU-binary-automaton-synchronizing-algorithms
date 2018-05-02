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
    public class SlimGPU : IComputation
    {
        public ComputationResult ComputeOne(Problem problemToSolve)
        => Compute(new[] { problemToSolve }, 1)[0];

        public ComputationResult[] Compute(Problem[] problemsToSolve, int streamCount)
        {
            var warps = 32; // dunno what to do with this guy
            var gpu = Gpu.Default;
            var n = problemsToSolve[0].size;
            var power = 1 << n;

            var takeRatio = (problemsToSolve.Length + streamCount - 1) / streamCount;
            var problemsPartitioned = Enumerable.Range(0, streamCount)
                    .Select(i => problemsToSolve.Skip(streamCount * i)
                    .Take(takeRatio)
                    .ToArray())
                .ToArray();

            var streams = Enumerable.Range(0, streamCount).Select(_ => gpu.CreateStream()).ToArray();

            var precomputedStateTransitioningMatrixA = problemsPartitioned.Select(problems =>
                {
                    var matrixA = new int[problems.Length * n];
                    for (int problem = 0; problem < problems.Length; problem++)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            matrixA[problem * n + i] = (ushort)(1 << problems[problem].stateTransitioningMatrixA[i]);
                        }
                    }
                    return matrixA;
                }).ToArray();
            var precomputedStateTransitioningMatrixB = problemsPartitioned.Select(problems =>
            {
                var matrixB = new int[problems.Length * n];
                for (int problem = 0; problem < problems.Length; problem++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        matrixB[problem * n + i] = (ushort)(1 << problems[problem].stateTransitioningMatrixB[i]);
                    }
                }
                return matrixB;
            }).ToArray();


            var gpuA = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var gpuB = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var shortestSynchronizingWordLength = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length)).ToArray();
            var isSynchronizable = problemsPartitioned.Select(problems => gpu.Allocate<bool>(problems.Length)).ToArray();
            var arrayCount = problemsPartitioned.Select(problems => gpu.Allocate(new[] { problems.Length })).ToArray();

            var launchParameters = new LaunchParam(
                1,
                32 * warps,
                power * 2 + 1 + 1
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

            foreach (var stream in streams)
                stream.Synchronize();

            var results = Enumerable.Range(0, streamCount).SelectMany(i =>
            {
                return Gpu.CopyToHost(isSynchronizable[i])
                        .Zip(Gpu.CopyToHost(shortestSynchronizingWordLength[i]), (isSyncable, shortestWordLength)
                    => new ComputationResult()
                    {
                        size = problemsToSolve[0].size,
                        isSynchronizable = isSyncable,
                        shortestSynchronizingWordLength = shortestWordLength
                    }
                    ).ToArray();
            }).ToArray();

            foreach (var item in gpuA)
                Gpu.Free(item);
            foreach (var item in gpuB)
                Gpu.Free(item);
            foreach (var item in shortestSynchronizingWordLength)
                Gpu.Free(item);
            foreach (var item in isSynchronizable)
                Gpu.Free(item);
            foreach (var item in arrayCount)
                Gpu.Free(item);

            foreach (var stream in streams) stream.Dispose();

            return results;
        }

        public static void Kernel(
            int[] arrayCount,
            int[] precomputedStateTransitioningMatrixA,
            int[] precomputedStateTransitioningMatrixB,
            bool[] isSynchronizing,
            int[] shortestSynchronizingWordLength)
        {
            //blockDim.x, blockDim.y
            var n = precomputedStateTransitioningMatrixA.Length / arrayCount[0];
            var power = 1 << n;

            var ptr = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>());
            var isDiscoveredPtr = ptr.Volatile();
            var isToBeProcessedDuringNextIteration = ptr.Ptr(power).Volatile();
            var shouldStop = ptr.Ptr(power * 2).Volatile();
            var addedSomethingThisRound = ptr.Ptr(power * 2 + 1).Volatile();
            if (threadIdx.x == 0)
                isToBeProcessedDuringNextIteration[power - 1] = true;
            ushort nextDistance = 1;
            int vertexAfterTransitionA, vertexAfterTransitionB;
            int myPart = (power + blockDim.x - 1) / blockDim.x;
            int correctlyProcessed = 0;
            int beginningPointer = threadIdx.x * myPart;
            int endingPointer = (threadIdx.x + 1) * myPart;
            if (power < endingPointer)
                endingPointer = power;

            for (int ac = 0; ac < arrayCount[0]; ac++)
            {
                // what if it is not synchronizable
                while (correctlyProcessed < endingPointer - beginningPointer && !shouldStop[0])
                {
                    for (int consideringVertex = beginningPointer; consideringVertex < endingPointer; consideringVertex++)
                    {
                        if (!isToBeProcessedDuringNextIteration[consideringVertex])
                            continue;
                        else
                        {
                            isToBeProcessedDuringNextIteration[consideringVertex] = false;
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
                            isDiscoveredPtr[vertexAfterTransitionA] = true;
                            isToBeProcessedDuringNextIteration[vertexAfterTransitionA] = true;
                            addedSomethingThisRound[0] = true;

                            if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                            {
                                shortestSynchronizingWordLength[ac] = nextDistance;
                                isSynchronizing[ac] = true;
                                shouldStop[0] = true;
                                break;
                            }

                        }

                        if (!isDiscoveredPtr[vertexAfterTransitionB])
                        {
                            isDiscoveredPtr[vertexAfterTransitionB] = true;
                            isToBeProcessedDuringNextIteration[vertexAfterTransitionB] = true;
                            addedSomethingThisRound[0] = true;

                            if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                            {
                                shortestSynchronizingWordLength[ac] = nextDistance;
                                isSynchronizing[ac] = true;
                                shouldStop[0] = true;
                                break;
                            }

                        }
                    }
                    ++nextDistance;
                    DeviceFunction.SyncThreads();
                    if (!addedSomethingThisRound[0])
                        break;
                    addedSomethingThisRound[0] = false;
                    DeviceFunction.SyncThreads();
                }

                // cleanup

                for (int consideringVertex = beginningPointer; consideringVertex < endingPointer; consideringVertex++)
                {
                    isDiscoveredPtr[consideringVertex] = false;
                    isToBeProcessedDuringNextIteration[consideringVertex] = false;
                    shouldStop[0] = false;
                    correctlyProcessed = 0;
                    nextDistance = 1;
                    addedSomethingThisRound[0] = false;
                }
                if (threadIdx.x == 0)
                    isToBeProcessedDuringNextIteration[power - 1] = true;
                DeviceFunction.SyncThreads();
            }
        }
        public int GetBestParallelism() => 4;
    }
}
