using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Version_1._0
{
    class CPU : IComputation
    {
        public ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism)
        {
            if (degreeOfParallelism == 1)
            {
                var results = new ComputationResult[problemsToSolve.Length];

                for (int i = 0; i < problemsToSolve.Length; i++)
                    results[i] = ComputeSerially(problemsToSolve[i]);

                return results;
            }
            else
            {
                return problemsToSolve
                    .AsParallel()
                    .WithDegreeOfParallelism(degreeOfParallelism)
                    .Select(problem => ComputeSerially(problem))
                    .ToArray();
            }
        }
        private ComputationResult ComputeSerially(Problem problemToSolve)
        {
            int powerSetCount = 1 << problemToSolve.size;
            int maximumPermissibleWordLength = (problemToSolve.size - 1) * (problemToSolve.size - 1);
            var isDiscovered = new bool[problemToSolve.size];
            var distanceToVertex = new int[powerSetCount];
            var previousVertex = new int[powerSetCount];

            isDiscovered[powerSetCount - 1] = true;

            var queue = new Queue<int>(powerSetCount);
            queue.Enqueue(powerSetCount - 1);

            var discoveredSingleton = false;
            int consideringVertex;
            int vertexAfterTransitionA;
            int vertexAfterTransitionB;
            int targetIndexPlusOne;
            int stateCount;
            int extractingBits;
            int firstSingleton;
            int distanceToConsideredVertex;

            var precomputedStateTransitioningMatrixA = new int[problemToSolve + 1];
            var precomputedStateTransitioningMatrixB = new int[problemToSolve + 1];

            precomputedStateTransitioningMatrixA[0] = 0;
            precomputedStateTransitioningMatrixB[0] = 0;
            for (int i = 0; i < problemToSolve.size; i++)
            {
                precomputedStateTransitioningMatrixA[i + 1] = problemToSolve.stateTransitioningMatrixA[i];
                precomputedStateTransitioningMatrixB[i + 1] = problemToSolve.stateTransitioningMatrixB[i];
            }

            while (!discoveredSingleton || queue.Count > 0)
            {
                extractingBits = consideringVertex = queue.Dequeue();
                distanceToConsideredVertex = distanceToVertex[consideringVertex];
                stateCount = 0;
                vertexAfterTransitionA = vertexAfterTransitionB = 0;

                // watch out for the index!
                for (int i = 1; i <= problemToSolve.size; i++)
                {
                    stateCount += extractingBits & 1;
                    targetIndexPlusOne = (extractingBits & 1) * i;

                    vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[targetIndexPlusOne];
                    vertexAfterTransitionB |= precomputedStateTransitioningMatrixA[targetIndexPlusOne];
                    extractingBits >>= 1;
                }

                if (stateCount == 1)
                {
                    // we've just discovered a singleton
                    discoveredSingleton = true;
                    firstSingleton = consideringVertex;
                    break;
                }

                if (distanceToVertex > maximumPermissibleWordLength)
                {
                    // now if this automata happens to be synchronizing then Cerny Conjecture is false!
                    // better be that this automata is not synchronizable!
                }

                if (!isDiscovered[vertexAfterTransitionA])
                {
                    distanceToVertex[vertexAfterTransitionA] = distanceToVertex[consideringVertex] + 1;
                    previousVertex[vertexAfterTransitionA] = consideringVertex;
                    isDiscovered[vertexAfterTransitionA] = true;
                    queue.Enqueue(vertexAfterTransitionA);
                }
                if (!isDiscovered[vertexAfterTransitionB])
                {
                    distanceToVertex[vertexAfterTransitionB] = distanceToVertex[consideringVertex] + 1;
                    previousVertex[vertexAfterTransitionA] = consideringVertex;
                    isDiscovered[vertexAfterTransitionB] = true;
                    queue.Enqueue(vertexAfterTransitionB);
                }
            }

            if (discoveredSingleton)
            {
                // watch out for off by one error!
                if (distanceToVertex[firstSingleton] > maximumPermissibleWordLength)
                {
                    // Cerny Conjecture is false!
                }
                else
                {
                    // everything is ok
                }
            }
            else
            {
                // not a synchronizing automata
            }
        }

    }
}
