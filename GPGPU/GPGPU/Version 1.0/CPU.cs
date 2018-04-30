﻿using GPGPU.Interfaces;
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
        private ushort[] previousVertexStatic = new ushort[mostProbablePowerSetCount];
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
                var previousVertexReusables = Enumerable.Range(0, reusableLengths).Select(_ => new ushort[mostProbablePowerSetCount]).ToArray();
                var previousLetterUsedEqualsBReusables = Enumerable.Range(0, reusableLengths).Select(_ => new bool[mostProbablePowerSetCount]).ToArray();



                //var n = problemsToSolve.Count();
                //var fraction = (n + degreeOfParallelism - 1) / degreeOfParallelism;

                //Parallel.For(
                //    0,
                //    degreeOfParallelism,
                //    new ParallelOptions() { MaxDegreeOfParallelism = degreeOfParallelism },
                //    i =>
                //{
                //    for (int j = i * fraction; j < (i + 1) * fraction && j < n; j++)
                //    {
                //        results[j] = ComputeOne(
                //            problemsToSolve[j],
                //            true,
                //            previousVertexReusables[i],
                //            previousLetterUsedEqualsBReusables[i]
                //            );
                //    }
                //});
                //return results;       

                var reusablesQueue = new ConcurrentQueue<int>();
                for (int i = 0; i < reusableLengths; i++)
                    reusablesQueue.Enqueue(i);
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
            ushort[] previousVertexReusable = null,
            bool[] previousLetterUsedEqualsBReusable = null
            )
        {
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();

            var result = new ComputationResult
            {
                benchmarkResult = new BenchmarkResult()
            };
            var n = problemToSolve.size;
            var powerSetCount = 1 << n;
            var initialVertex = (ushort)(powerSetCount - 1);
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            ushort[] distanceToVertex;
            ushort[] previousVertex;
            bool[] previousLetterUsedEqualsB;
            bool[] isDiscovered = new bool[powerSetCount];
            isDiscovered[initialVertex] = true;

            // about 15% of overall computation using single thread
            // about 30% of overall computation using multiple threads + reusage
            if (reuseResources)
            {
                // use whatever size is needed, they all should be consistent (of same size)
                if (previousVertexReusable.Length != powerSetCount)
                {
                    previousVertexReusable = new ushort[powerSetCount];
                    previousLetterUsedEqualsBReusable = new bool[powerSetCount];
                }
                distanceToVertex = new ushort[powerSetCount];
                previousVertex = previousVertexReusable;
                previousLetterUsedEqualsB = previousLetterUsedEqualsBReusable;
            }
            else
            {
                distanceToVertex = new ushort[powerSetCount];
                previousVertex = new ushort[powerSetCount];
                previousLetterUsedEqualsB = new bool[powerSetCount];
            }


            var queue = new Queue<ushort>(powerSetCount);
            queue.Enqueue(initialVertex);

            var discoveredSingleton = false;
            ushort consideringVertex;
            ushort vertexAfterTransitionA;
            ushort vertexAfterTransitionB;
            int targetIndexPlusOne;
            int extractingBits;
            int firstSingleton = 0;
            int distanceToConsideredVertex;

            var precomputedStateTransitioningMatrixA = new ushort[n + 1];
            var precomputedStateTransitioningMatrixB = new ushort[n + 1];

            precomputedStateTransitioningMatrixA[0] = 0;
            precomputedStateTransitioningMatrixB[0] = 0;
            for (int i = 0; i < n; i++)
            {
                precomputedStateTransitioningMatrixA[i + 1] = (ushort)(1 << problemToSolve.stateTransitioningMatrixA[i]);
                precomputedStateTransitioningMatrixB[i + 1] = (ushort)(1 << problemToSolve.stateTransitioningMatrixB[i]);
            }
            //benchmarkTiming.Start();
            while (!discoveredSingleton && queue.Count > 0)
            {
                extractingBits = consideringVertex = queue.Dequeue();
                distanceToConsideredVertex = distanceToVertex[consideringVertex];
                vertexAfterTransitionA = vertexAfterTransitionB = 0;

                // check for singleton existance
                //b && !(b & (b-1)) https://stackoverflow.com/questions/12483843/test-if-a-bitboard-have-only-one-bit-set-to-1
                if (consideringVertex != 0 && (consideringVertex & (consideringVertex - 1)) == 0)
                {
                    discoveredSingleton = true;
                    firstSingleton = consideringVertex;
                    break;
                }
                // watch out for the index!
                for (int i = 1; i <= n; i++)
                {
                    targetIndexPlusOne = (extractingBits & 1) * i;

                    vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[targetIndexPlusOne];
                    vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[targetIndexPlusOne];
                    extractingBits >>= 1;
                }

                if (!isDiscovered[vertexAfterTransitionA])
                {
                    distanceToVertex[vertexAfterTransitionA] = (ushort)(distanceToVertex[consideringVertex] + 1);
                    previousVertex[vertexAfterTransitionA] = consideringVertex;
                    isDiscovered[vertexAfterTransitionA] = true;
                    previousLetterUsedEqualsB[vertexAfterTransitionA] = false;
                    queue.Enqueue(vertexAfterTransitionA);
                }
                if (!isDiscovered[vertexAfterTransitionB])
                {
                    distanceToVertex[vertexAfterTransitionB] = (ushort)(distanceToVertex[consideringVertex] + 1);
                    previousVertex[vertexAfterTransitionB] = consideringVertex;
                    isDiscovered[vertexAfterTransitionB] = true;
                    previousLetterUsedEqualsB[vertexAfterTransitionB] = true;
                    queue.Enqueue(vertexAfterTransitionB);
                }
            }
            //benchmarkTiming.Stop();

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
