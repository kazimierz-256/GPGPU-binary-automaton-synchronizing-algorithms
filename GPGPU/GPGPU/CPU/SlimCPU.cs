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
    public class SlimCPU : IComputable
    {
        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int degreeOfParallelism)
        {
            if (degreeOfParallelism == 1)
            {
                return problemsToSolve
                    .Select(problem => ComputeOne(problem))
                    .ToArray();
            }
            else
            {
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
            ushort consideringVertex;
            ushort vertexAfterTransitionA;
            ushort vertexAfterTransitionB;
            ushort firstSingletonDistance = 0;

            var precomputedStateTransitioningMatrixA = new ushort[n];
            var precomputedStateTransitioningMatrixB = new ushort[n];

            for (int i = 0; i < n; i++)
            {
                precomputedStateTransitioningMatrixA[i] = (ushort)(1 << problemToSolve.stateTransitioningMatrixA[i]);
                precomputedStateTransitioningMatrixB[i] = (ushort)(1 << problemToSolve.stateTransitioningMatrixB[i]);
            }

            //var maximumBreadth = 0;
            ushort currentNextDistance = 1;
            var verticesUntilBump = ushort.MaxValue;
            var seekingFirstNext = true;

            benchmarkTiming.Start();
            while (queue.Count > 0)
            {
                //if (queue.Count > maximumBreadth)
                //    maximumBreadth = queue.Count;

                consideringVertex = queue.Dequeue();

                if (--verticesUntilBump == 0)
                {
                    ++currentNextDistance;
                    seekingFirstNext = true;
                }

                vertexAfterTransitionA = vertexAfterTransitionB = 0;

                // check for singleton existance
                // b && !(b & (b-1)) https://stackoverflow.com/questions/12483843/test-if-a-bitboard-have-only-one-bit-set-to-1
                // note: consideringVertex cannot ever be equal to 0

                // watch out for the index range in the for loop
                for (int i = 0; i < n; i++)
                {
                    if (0 != ((1 << i) & consideringVertex))
                    {
                        vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[i];
                        vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[i];
                    }
                }

                if (!isDiscovered[vertexAfterTransitionA])
                {
                    if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                    {
                        discoveredSingleton = true;
                        firstSingletonDistance = currentNextDistance;
                        break;
                    }

                    isDiscovered[vertexAfterTransitionA] = true;
                    queue.Enqueue(vertexAfterTransitionA);

                    if (seekingFirstNext)
                    {
                        seekingFirstNext = false;
                        verticesUntilBump = (ushort)queue.Count;
                    }
                }

                if (!isDiscovered[vertexAfterTransitionB])
                {
                    if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                    {
                        discoveredSingleton = true;
                        firstSingletonDistance = currentNextDistance;
                        break;
                    }

                    isDiscovered[vertexAfterTransitionB] = true;
                    queue.Enqueue(vertexAfterTransitionB);

                    if (seekingFirstNext)
                    {
                        seekingFirstNext = false;
                        verticesUntilBump = (ushort)queue.Count;
                    }
                }
            }
            benchmarkTiming.Stop();

            // finished main loop

            var result = new ComputationResult()
            {
                benchmarkResult = new BenchmarkResult(),
                computationType = ComputationType.CPU_Serial,
                //queueBreadth = maximumBreadth,
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

        public int GetBestParallelism() => Environment.ProcessorCount;
    }
}
