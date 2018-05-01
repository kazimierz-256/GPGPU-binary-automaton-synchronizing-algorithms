using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea.CSharp;
using Alea;

namespace GPGPU
{
    public class SlimGPU : IComputation
    {
        public ComputationResult ComputeOne(Problem problemToSolve)
        => Compute(new[] { problemToSolve }, 1)[0];

        public ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism)
        {
            //Enumerable.Range(0, problemsToSolve.Length)
            //.Select(i => ComputeOne(problemsToSolve[i]))
            //.ToArray();


            var warps = 32; // dunno what to do with this guy
            var gpu = Gpu.Default;
            var n = problemsToSolve[0].size;
            var power = 1 << n;

            // transform this into streams

            var launchParameters = new LaunchParam(
                1,
                32 * warps,
                power * 2 + 1 + 1
            );


            var precomputedStateTransitioningMatrixA = new int[problemsToSolve.Length * n];
            var precomputedStateTransitioningMatrixB = new int[problemsToSolve.Length * n];

            for (int problem = 0; problem < problemsToSolve.Length; problem++)
            {
                for (int i = 0; i < n; i++)
                {
                    precomputedStateTransitioningMatrixA[problem * n + i] = (ushort)(1 << problemsToSolve[problem].stateTransitioningMatrixA[i]);
                    precomputedStateTransitioningMatrixB[problem * n + i] = (ushort)(1 << problemsToSolve[problem].stateTransitioningMatrixB[i]);
                }
            }

            var gpuA = gpu.Allocate(precomputedStateTransitioningMatrixA);
            var gpuB = gpu.Allocate(precomputedStateTransitioningMatrixB);
            var shortestSynchronizingWordLength = gpu.Allocate<int>(problemsToSolve.Length);
            var isSynchronizable = gpu.Allocate<bool>(problemsToSolve.Length);
            var arrayCount = gpu.Allocate(new[] { problemsToSolve.Length });

            gpu.Launch(
                Kernel,
                launchParameters,
                arrayCount,
                gpuA,
                gpuB,
                isSynchronizable,
                shortestSynchronizingWordLength
                );

            var results = Gpu.CopyToHost(isSynchronizable).Zip(Gpu.CopyToHost(shortestSynchronizingWordLength), (isSyncable, shortestWordLength)
                => new ComputationResult()
                {
                    size = problemsToSolve[0].size,
                    isSynchronizable = isSyncable,
                    shortestSynchronizingWordLength = shortestWordLength
                }
                ).ToArray();

            Gpu.Free(gpuA);
            Gpu.Free(gpuB);
            Gpu.Free(shortestSynchronizingWordLength);
            Gpu.Free(isSynchronizable);
            Gpu.Free(arrayCount);

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
    }
}
