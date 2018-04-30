using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Version_1._0
{
    public class CPU : IComputation
    {
        public ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism)
        {
            if (degreeOfParallelism == 1)
            {
                return Enumerable.Range(0, problemsToSolve.Length)
                    .Select(i => ComputeOne(problemsToSolve[i]))
                    .ToArray();
            }
            else
            {
                //var results = new ComputationResult[problemsToSolve.Length];

                //Parallel.For(0, problemsToSolve.Length, new ParallelOptions() { MaxDegreeOfParallelism = degreeOfParallelism }, i =>
                //    results[i] = ComputeOne(problemsToSolve[i])
                //);
                //return results;
                return problemsToSolve
                    .AsParallel()
                    .WithDegreeOfParallelism(degreeOfParallelism)
                    .Select(problem => ComputeOne(problem))
                    .ToArray();
            }
        }
        private static int latestPowerSetCount = 13;

        private static int[] distanceToVertexStatic = new int[latestPowerSetCount];
        private static int[] previousVertexStatic = new int[latestPowerSetCount];
        private static bool[] previousLetterUsedEqualsBStatic = new bool[latestPowerSetCount];

        public ComputationResult ComputeOne(Problem problemToSolve)
        {
            var benchmarkTiming = new Stopwatch();
            var result = new ComputationResult
            {
                benchmarkResult = new BenchmarkResult()
            };
            int powerSetCount = 1 << problemToSolve.size;
            int maximumPermissibleWordLength = (problemToSolve.size - 1) * (problemToSolve.size - 1);
            int initialVertex = powerSetCount - 1;
            benchmarkTiming.Start();
            var distanceToVertex = new int[powerSetCount];
            var previousVertex = new int[powerSetCount];
            var previousLetterUsedEqualsB = new bool[powerSetCount];


            var queue = new Queue<int>(powerSetCount);
            queue.Enqueue(initialVertex);

            var discoveredSingleton = false;
            int consideringVertex;
            int vertexAfterTransitionA;
            int vertexAfterTransitionB;
            int targetIndexPlusOne;
            int stateCount;
            int extractingBits;
            int firstSingleton = 0;
            int distanceToConsideredVertex;

            var precomputedStateTransitioningMatrixA = new int[problemToSolve.size + 1];
            var precomputedStateTransitioningMatrixB = new int[problemToSolve.size + 1];
            benchmarkTiming.Stop();

            precomputedStateTransitioningMatrixA[0] = 0;
            precomputedStateTransitioningMatrixB[0] = 0;
            for (int i = 0; i < problemToSolve.size; i++)
            {
                precomputedStateTransitioningMatrixA[i + 1] = 1 << problemToSolve.stateTransitioningMatrixA[i];
                precomputedStateTransitioningMatrixB[i + 1] = 1 << problemToSolve.stateTransitioningMatrixB[i];
            }
            while (!discoveredSingleton && queue.Count > 0)
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
                    vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[targetIndexPlusOne];
                    extractingBits >>= 1;
                }

                if (stateCount == 1)
                {
                    // we've just discovered a singleton
                    discoveredSingleton = true;
                    firstSingleton = consideringVertex;
                    break;
                }

                //if (distanceToVertex[consideringVertex] > maximumPermissibleWordLength)
                //{
                //    // now if this automata happens to be synchronizing then Cerny Conjecture is false!
                //    // better be that this automata is not synchronizable!
                //}

                if (0 == distanceToVertex[vertexAfterTransitionA])
                {
                    distanceToVertex[vertexAfterTransitionA] = distanceToVertex[consideringVertex] + 1;
                    previousVertex[vertexAfterTransitionA] = consideringVertex;
                    previousLetterUsedEqualsB[vertexAfterTransitionA] = false;
                    queue.Enqueue(vertexAfterTransitionA);
                }
                if (0 == distanceToVertex[vertexAfterTransitionB])
                {
                    distanceToVertex[vertexAfterTransitionB] = distanceToVertex[consideringVertex] + 1;
                    previousVertex[vertexAfterTransitionB] = consideringVertex;
                    previousLetterUsedEqualsB[vertexAfterTransitionB] = true;
                    queue.Enqueue(vertexAfterTransitionB);
                }
            }

            // finished main loop

            if (discoveredSingleton)
            {
                // watch out for off by one error!
                if (distanceToVertex[firstSingleton] > maximumPermissibleWordLength)
                {
                    // Cerny Conjecture is false!
                    throw new Exception("Cerny conjecture is false");
                }
                else
                {
                    // everything is ok
                    result.isSynchronizable = true;

                    int wordLength = distanceToVertex[firstSingleton];
                    int currentVertex = firstSingleton;
                    result.shortestSynchronizingWord = new bool[wordLength];
                    for (int i = wordLength - 1; i >= 0; i--)
                    {
                        result.shortestSynchronizingWord[i] = previousLetterUsedEqualsB[currentVertex];
                        currentVertex = previousVertex[currentVertex];
                    }
                }
            }
            else
            {
                // not a synchronizing automata
                result.isSynchronizable = false;
            }

            result.benchmarkResult.benchmarkedTime = benchmarkTiming.Elapsed;
            return result;
        }

    }
}
