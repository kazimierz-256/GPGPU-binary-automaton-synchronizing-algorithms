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
        public ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism) =>
            Enumerable.Range(0, problemsToSolve.Length)
            .Select(i => ComputeOne(problemsToSolve[i]))
            .ToArray();

        public ComputationResult ComputeOne(Problem problemToSolve)
        {
            var gpu = Gpu.Default;
            var n = problemToSolve.size;
            var power = 1 << n;

            // transform this into streams

            // allocate isDiscovered
            var warps = 4;
            // each isDiscovered has 2 bytes of room
            var launchParameters = new LaunchParam(
                1,
                32 * warps,
                power * 2 + 1
            );


            var precomputedStateTransitioningMatrixA = new int[n];
            var precomputedStateTransitioningMatrixB = new int[n];

            for (int i = 0; i < n; i++)
            {
                precomputedStateTransitioningMatrixA[i] = (ushort)(1 << problemToSolve.stateTransitioningMatrixA[i]);
                precomputedStateTransitioningMatrixB[i] = (ushort)(1 << problemToSolve.stateTransitioningMatrixB[i]);
            }

            var gpuA = gpu.Allocate(precomputedStateTransitioningMatrixA);
            var gpuB = gpu.Allocate(precomputedStateTransitioningMatrixB);
            var shortestSynchronizingWordLength = gpu.Allocate<int>(1);
            var isSynchronizable = gpu.Allocate<bool>(1);

            gpu.Launch(
                Kernel,
                launchParameters,
                gpuA,
                gpuB,
                isSynchronizable,
                shortestSynchronizingWordLength
                );

            var result = new ComputationResult()
            {
                size = problemToSolve.size,
                isSynchronizable = Gpu.CopyToHost(isSynchronizable)[0],
                shortestSynchronizingWordLength = Gpu.CopyToHost(shortestSynchronizingWordLength)[0]
            };

            Gpu.Free(gpuA);
            Gpu.Free(gpuB);
            Gpu.Free(shortestSynchronizingWordLength);
            Gpu.Free(isSynchronizable);

            return result;
        }

        public static void Kernel(
            int[] precomputedStateTransitioningMatrixA,
            int[] precomputedStateTransitioningMatrixB,
            bool[] isSynchronizing,
            int[] shortestSynchronizingWordLength)
        {
            //blockDim.x, blockDim.y
            var n = precomputedStateTransitioningMatrixA.Length;
            var power = 1 << n;

            var ptr = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>());
            var isDiscoveredPtr = ptr.Volatile();
            var isToBeProcessedDuringNextIteration = ptr.Ptr(power).Volatile();
            var shouldStop = ptr.Ptr(power * 2).Reinterpret<bool>();
            if (threadIdx.x == 0)
                isToBeProcessedDuringNextIteration[power - 1] = true;
            ushort nextDistance = 1;
            int vertexAfterTransitionA, vertexAfterTransitionB;
            int myPart = (power + blockDim.x - 1) / blockDim.x;
            int discoveredVertices = 0;
            int beginningPointer = threadIdx.x * myPart;
            int endingPointer = (threadIdx.x + 1) * myPart;
            if (power < endingPointer)
                endingPointer = power;

            while (discoveredVertices < endingPointer - beginningPointer && !shouldStop[0])
            {
                for (int consideringVertex = beginningPointer; consideringVertex < endingPointer; consideringVertex++)
                {
                    if (!isToBeProcessedDuringNextIteration[consideringVertex])
                        continue;
                    else
                        isToBeProcessedDuringNextIteration[consideringVertex] = false;

                    vertexAfterTransitionA = vertexAfterTransitionB = 0;

                    for (int i = 0; i < n; i++)
                    {
                        if (0 != ((1 << i) & consideringVertex))
                        {
                            vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[i];
                            vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[i];
                        }
                    }

                    if (!isDiscoveredPtr[vertexAfterTransitionA])
                    {
                        isDiscoveredPtr[vertexAfterTransitionA] = true;
                        isToBeProcessedDuringNextIteration[vertexAfterTransitionA] = true;

                        if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                        {
                            shortestSynchronizingWordLength[0] = nextDistance;
                            isSynchronizing[0] = true;
                            shouldStop[0] = true;
                            break;
                        }

                    }

                    if (!isDiscoveredPtr[vertexAfterTransitionB])
                    {
                        isDiscoveredPtr[vertexAfterTransitionB] = true;
                        isToBeProcessedDuringNextIteration[vertexAfterTransitionB] = true;

                        if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                        {
                            shortestSynchronizingWordLength[0] = nextDistance;
                            isSynchronizing[0] = true;
                            shouldStop[0] = true;
                            break;
                        }

                    }
                }
                ++nextDistance;
                DeviceFunction.SyncThreads();
            }
        }
    }
}
