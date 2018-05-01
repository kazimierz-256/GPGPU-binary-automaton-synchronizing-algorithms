using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU
{
    public class SlimCPU : IComputation
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
                var results = new ComputationResult[problemsToSolve.Length];

                return problemsToSolve
                    .AsParallel()
                    .WithDegreeOfParallelism(degreeOfParallelism)
                    .Select(problem => ComputeOne(problem))
                    .ToArray();
            }
        }

        public ComputationResult ComputeOne(Problem problemToSolve)
        {
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();


            var n = problemToSolve.size;
            var powerSetCount = 1 << n;
            var initialVertex = (ushort)(powerSetCount - 1);
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            var isDiscovered = new bool[powerSetCount];
            isDiscovered[initialVertex] = true;


            var queue = new Queue<ushort>(n * 5);
            queue.Enqueue(initialVertex);

            var discoveredSingleton = false;
            ushort consideringVertexThenExtractingBits;
            ushort vertexAfterTransitionA;
            ushort vertexAfterTransitionB;
            int targetIndexPlusOne;
            ushort firstSingletonDistance = 0;

            var precomputedStateTransitioningMatrixA = new ushort[n + 1];
            var precomputedStateTransitioningMatrixB = new ushort[n + 1];

            for (int i = 0; i < n; i++)
            {
                precomputedStateTransitioningMatrixA[i + 1] = (ushort)(1 << problemToSolve.stateTransitioningMatrixA[i]);
                precomputedStateTransitioningMatrixB[i + 1] = (ushort)(1 << problemToSolve.stateTransitioningMatrixB[i]);
            }

            var maximumBreadth = 0;
            ushort bumpUpVertex = 0;
            ushort currentNextDistance = 1;
            bool seekingFirstNext = true;

            benchmarkTiming.Start();
            while (queue.Count > 0)
            {
                //if (queue.Count > maximumBreadth)
                //    maximumBreadth = queue.Count;

                consideringVertexThenExtractingBits = queue.Dequeue();
                if (consideringVertexThenExtractingBits == bumpUpVertex)
                {
                    ++currentNextDistance;
                    seekingFirstNext = true;
                }

                vertexAfterTransitionA = vertexAfterTransitionB = 0;

                // check for singleton existance
                // b && !(b & (b-1)) https://stackoverflow.com/questions/12483843/test-if-a-bitboard-have-only-one-bit-set-to-1
                // note: consideringVertex cannot ever be equal to 0

                // watch out for the index range in the for loop
                for (int i = 1; i <= n; i++)
                {
                    targetIndexPlusOne = (consideringVertexThenExtractingBits & 1) * i;

                    vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[targetIndexPlusOne];
                    vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[targetIndexPlusOne];
                    consideringVertexThenExtractingBits >>= 1;
                }

                if (!isDiscovered[vertexAfterTransitionA])
                {
                    if (seekingFirstNext)
                    {
                        seekingFirstNext = false;
                        bumpUpVertex = vertexAfterTransitionA;
                    }
                    isDiscovered[vertexAfterTransitionA] = true;
                    queue.Enqueue(vertexAfterTransitionA);

                    if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                    {
                        discoveredSingleton = true;
                        firstSingletonDistance = currentNextDistance;
                        break;
                    }
                }
                if (!isDiscovered[vertexAfterTransitionB])
                {
                    if (seekingFirstNext)
                    {
                        seekingFirstNext = false;
                        bumpUpVertex = vertexAfterTransitionB;
                    }
                    isDiscovered[vertexAfterTransitionB] = true;
                    queue.Enqueue(vertexAfterTransitionB);

                    if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                    {
                        discoveredSingleton = true;
                        firstSingletonDistance = currentNextDistance;
                        break;
                    }
                }
            }
            benchmarkTiming.Stop();

            // finished main loop

            var result = new ComputationResult()
            {
                benchmarkResult = new BenchmarkResult(),
                computationType = ComputationType.CPU_Serial,
                queueBreadth = maximumBreadth,
                size = n,
                //discoveredVertices = isDiscovered.Sum(vertex => vertex ? 1 : 0)
            };

            if (discoveredSingleton)
            {
                result.isSynchronizable = true;
                // watch out for off by one error!
                if (firstSingletonDistance > maximumPermissibleWordLength)
                {
                    throw new Exception("Cerny conjecture is false");
                }
            }
            else
            {
                // not a synchronizing automata
                // hmm.. is this correctly computed?
                result.isSynchronizable = false;
            }

            result.shortestSynchronizingWordLength = firstSingletonDistance;
            result.benchmarkResult.benchmarkedTime = benchmarkTiming.Elapsed;
            result.benchmarkResult.totalTime = totalTiming.Elapsed;
            return result;
        }

    }
}
