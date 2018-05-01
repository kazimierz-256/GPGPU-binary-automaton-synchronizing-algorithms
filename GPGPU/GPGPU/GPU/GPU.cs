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

            // transform this into streams

            // allocate isDiscovered
            var warps = 4;
            // each isDiscovered has 8 bits of room :/
            var launchParameters = new LaunchParam(
                1,
                32 * warps,
                (1 << problemToSolve.size) * 4 + 1
            );// divide by 8?


            var precomputedStateTransitioningMatrixA = new int[n + 1];
            var precomputedStateTransitioningMatrixB = new int[n + 1];

            for (int i = 0; i < n; i++)
            {
                precomputedStateTransitioningMatrixA[i + 1] = (ushort)(1 << problemToSolve.stateTransitioningMatrixA[i]);
                precomputedStateTransitioningMatrixB[i + 1] = (ushort)(1 << problemToSolve.stateTransitioningMatrixB[i]);
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
            var n = precomputedStateTransitioningMatrixA.Length - 1;
            var power = 1 << n;

            var ptr = DeviceFunction.AddressOfArray(__shared__.ExternArray<bool>());
            var isDiscoveredPtr = ptr;
            var isToBeProcessedDuringNextIteration = ptr.Ptr(power).Volatile();
            //var shouldStop = ptr.Ptr(power).Reinterpret<int>();
            isToBeProcessedDuringNextIteration[power - 1] = true;
            int nextDistance = 1;
            int vertexAfterTransitionA, vertexAfterTransitionB;
            int myPart = (power + blockDim.x - 1) / blockDim.x;
            // very poor condition
            while (!isSynchronizing[0])
            {
                for (int pointer = threadIdx.x * myPart;
                    pointer < (threadIdx.x + 1) * myPart && pointer < power;
                    pointer++)
                {
                    if (!isToBeProcessedDuringNextIteration[pointer])
                        continue;
                    else
                        isToBeProcessedDuringNextIteration[pointer] = false;

                    var consideringVertexThenExtractingBits = pointer;
                    int targetIndexPlusOne;

                    vertexAfterTransitionA = vertexAfterTransitionB = 0;

                    for (int i = 1; i <= n; i++)
                    {
                        targetIndexPlusOne = (consideringVertexThenExtractingBits & 1) * i;

                        vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[targetIndexPlusOne];
                        vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[targetIndexPlusOne];
                        consideringVertexThenExtractingBits >>= 1;
                    }

                    if (!isDiscoveredPtr[vertexAfterTransitionA])
                    {
                        isDiscoveredPtr[vertexAfterTransitionA] = true;
                        isToBeProcessedDuringNextIteration[vertexAfterTransitionA] = true;

                        if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                        {
                            //DeviceFunction.AtomicExchange(shouldStop, nextDistance);
                            isSynchronizing[0] = true;
                            shortestSynchronizingWordLength[0] = nextDistance;
                            break;
                        }

                    }

                    if (!isDiscoveredPtr[vertexAfterTransitionB])
                    {
                        isDiscoveredPtr[vertexAfterTransitionB] = true;
                        isToBeProcessedDuringNextIteration[vertexAfterTransitionB] = true;

                        if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                        {
                            //DeviceFunction.AtomicExchange(shouldStop, nextDistance);
                            isSynchronizing[0] = true;
                            shortestSynchronizingWordLength[0] = nextDistance;
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
