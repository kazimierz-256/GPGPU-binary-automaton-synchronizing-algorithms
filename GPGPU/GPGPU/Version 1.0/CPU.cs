using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Version_1._0
{
    public class CPU : IComputation
    {
        private static readonly int mostProbablePowerSetCount = 1 << 13;
        private short[] previousVertexStatic = new short[mostProbablePowerSetCount];
        private bool[] previousLetterUsedEqualsBStatic = new bool[mostProbablePowerSetCount];

        public ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism)
        {
            if (degreeOfParallelism == 1)
            {
                return Enumerable.Range(0, problemsToSolve.Length)
                    .Select(i => ComputeOne(
                        problemsToSolve[i],
                        true,
                        previousVertexStatic,
                        previousLetterUsedEqualsBStatic))
                    .ToArray();
            }
            else
            {
                var results = new ComputationResult[problemsToSolve.Length];
                var reusableLengths = degreeOfParallelism;
                var previousVertexReusables = Enumerable.Range(0, reusableLengths).Select(_ => new short[mostProbablePowerSetCount]).ToArray();
                var previousLetterUsedEqualsBReusables = Enumerable.Range(0, reusableLengths).Select(_ => new bool[mostProbablePowerSetCount]).ToArray();

                var reusablesQueue = new ConcurrentQueue<int>();
                for (int i = 0; i < reusableLengths; i++)
                    reusablesQueue.Enqueue(i);

                //Parallel.For(0, problemsToSolve.Length, new ParallelOptions() { MaxDegreeOfParallelism = degreeOfParallelism }, i =>
                //{
                //    //if (Task.CurrentId.HasValue)
                //    //{
                //    reusablesQueue.TryDequeue(out int id);
                //    results[i] = ComputeOne(
                //        problemsToSolve[i],
                //        true,
                //        previousVertexReusables[id],
                //        previousLetterUsedEqualsBReusables[id]
                //        );
                //    reusablesQueue.Enqueue(id);
                //    //}
                //    //else
                //    //{
                //    //    results[i] = ComputeOne(problemsToSolve[i], false);
                //    //}
                //});
                //return results;
                return problemsToSolve
                    .AsParallel()
                    .WithDegreeOfParallelism(degreeOfParallelism)
                    .Select(problem =>
                    {
                        var success = reusablesQueue.TryDequeue(out int id);
                        if (success)
                        {
                            var result = ComputeOne(
                                problem,
                                true,
                                previousVertexReusables[id],
                                previousLetterUsedEqualsBReusables[id]
                                );
                            reusablesQueue.Enqueue(id);
                            return result;
                        }
                        else
                            return ComputeOne(problem, false);
                    })
                    .ToArray();
            }
        }

        public ComputationResult ComputeOne(Problem problemToSolve) => ComputeOne(problemToSolve, false);

        public ComputationResult ComputeOne(
            Problem problemToSolve,
            bool reuseResources = false,
            short[] previousVertexReusable = null,
            bool[] previousLetterUsedEqualsBReusable = null
            )
        {
            var benchmarkTiming = new Stopwatch();
            var totalTiming = new Stopwatch();
            totalTiming.Start();

            var result = new ComputationResult
            {
                benchmarkResult = new BenchmarkResult()
            };
            var n = problemToSolve.size;
            int powerSetCount = 1 << n;
            int maximumPermissibleWordLength = (n - 1) * (n - 1);
            short initialVertex = (short)(powerSetCount - 1);

            short[] distanceToVertex;
            short[] previousVertex;
            bool[] previousLetterUsedEqualsB;

            benchmarkTiming.Start();
            if (reuseResources)
            {
                // use whatever size is needed, they all should be consistent (of same size)
                if (previousVertexReusable.Length != powerSetCount)
                {
                    previousVertexReusable = new short[powerSetCount];
                    previousLetterUsedEqualsBReusable = new bool[powerSetCount];
                }
                else
                {
                    // about 15% of overall computation!
                }
                distanceToVertex = new short[powerSetCount];
                previousVertex = previousVertexReusable;
                previousLetterUsedEqualsB = previousLetterUsedEqualsBReusable;
            }
            else
            {
                distanceToVertex = new short[powerSetCount];
                previousVertex = new short[powerSetCount];
                previousLetterUsedEqualsB = new bool[powerSetCount];
            }
            benchmarkTiming.Stop();


            var queue = new Queue<short>(powerSetCount);
            queue.Enqueue(initialVertex);

            var discoveredSingleton = false;
            short consideringVertex;
            short vertexAfterTransitionA;
            short vertexAfterTransitionB;
            int targetIndexPlusOne;
            int stateCount;
            int extractingBits;
            int firstSingleton = 0;
            int distanceToConsideredVertex;

            var precomputedStateTransitioningMatrixA = new short[n + 1];
            var precomputedStateTransitioningMatrixB = new short[n + 1];

            precomputedStateTransitioningMatrixA[0] = 0;
            precomputedStateTransitioningMatrixB[0] = 0;
            for (int i = 0; i < n; i++)
            {
                precomputedStateTransitioningMatrixA[i + 1] = (short)(1 << problemToSolve.stateTransitioningMatrixA[i]);
                precomputedStateTransitioningMatrixB[i + 1] = (short)(1 << problemToSolve.stateTransitioningMatrixB[i]);
            }
            while (!discoveredSingleton && queue.Count > 0)
            {
                extractingBits = consideringVertex = queue.Dequeue();
                distanceToConsideredVertex = distanceToVertex[consideringVertex];
                stateCount = 0;
                vertexAfterTransitionA = vertexAfterTransitionB = 0;

                // watch out for the index!
                for (int i = 1; i <= n; i++)
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
                    distanceToVertex[vertexAfterTransitionA] = (short)(distanceToVertex[consideringVertex] + 1);
                    previousVertex[vertexAfterTransitionA] = consideringVertex;
                    previousLetterUsedEqualsB[vertexAfterTransitionA] = false;
                    queue.Enqueue(vertexAfterTransitionA);
                }
                if (0 == distanceToVertex[vertexAfterTransitionB])
                {
                    distanceToVertex[vertexAfterTransitionB] = (short)(distanceToVertex[consideringVertex] + 1);
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
            result.benchmarkResult.totalTime = totalTiming.Elapsed;
            return result;
        }

    }
}
