#define benchmark
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
        public int GetBestParallelism() => Environment.ProcessorCount;

        public ComputationResult[] Compute(Problem[] problemsToSolve, int beginningIndex, int problemCount, int degreeOfParallelism)
        {
            if (degreeOfParallelism > problemCount)
            {
                degreeOfParallelism = problemCount;
            }
            if (degreeOfParallelism == 1)
            {
                return ComputeMany(problemsToSolve, beginningIndex, problemCount);
            }
            else
            {
                var results = new ComputationResult[problemCount];
                var partition = (problemCount + degreeOfParallelism - 1) / degreeOfParallelism;
                Parallel.For(0, degreeOfParallelism, i =>
                {
                    var miniResults = ComputeMany(
                        problemsToSolve,
                        beginningIndex + i * partition,
                        Math.Min(partition, problemCount - i * partition)
                        );
                    Array.ConstrainedCopy(miniResults, 0, results, beginningIndex + i * partition, miniResults.Length);
                });
                return results;
            }
        }

        private ComputationResult[] ComputeMany(Problem[] problemsToSolve, int beginningIndex, int problemCount)
        {
            var results = new ComputationResult[problemCount];
#if (benchmark)
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();
#endif

            var n = problemsToSolve[beginningIndex].size;
            var powerSetCount = 1 << n;
            var initialVertex = (ushort)(powerSetCount - 1);
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            byte localProblemId = 1;
            var isDiscovered = new byte[powerSetCount];

            var queue = new Queue<ushort>(n * 2);

            var precomputedStateTransitioningMatrixA = new ushort[n];
            var precomputedStateTransitioningMatrixB = new ushort[n];
            for (int problem = beginningIndex, endingIndex = beginningIndex + problemCount; problem < endingIndex; problem++, localProblemId++)
            {
#if (benchmark)
                benchmarkTiming.Start();
#endif
                if (localProblemId == 0)
                {
                    localProblemId = 1;
                    Array.Clear(isDiscovered, 0, isDiscovered.Length);
                    //// should be faster than zeroing out an array
                    //isDiscovered = new byte[powerSetCount];
                }
                isDiscovered[initialVertex] = localProblemId;
                queue.Clear();
                queue.Enqueue(initialVertex);

                var discoveredSingleton = false;
                ushort consideringVertex;
                ushort vertexAfterTransitionA;
                ushort vertexAfterTransitionB;


                for (int i = 0; i < n; i++)
                {
                    precomputedStateTransitioningMatrixA[i] = (ushort)(1 << problemsToSolve[problem].stateTransitioningMatrixA[i]);
                    precomputedStateTransitioningMatrixB[i] = (ushort)(1 << problemsToSolve[problem].stateTransitioningMatrixB[i]);
                }

                //var maximumBreadth = 0;
                ushort currentNextDistance = 1;
                var verticesUntilBump = ushort.MaxValue;
                var seekingFirstNext = true;

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
                    for (int i = 0, mask = 1; i < n; i++, mask <<= 1)
                    {
                        if (0 != (mask & consideringVertex))
                        {
                            vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[i];
                            vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[i];
                        }
                    }

                    if (localProblemId != isDiscovered[vertexAfterTransitionA])
                    {
                        if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                        {
                            discoveredSingleton = true;
                            break;
                        }

                        isDiscovered[vertexAfterTransitionA] = localProblemId;
                        queue.Enqueue(vertexAfterTransitionA);

                        if (seekingFirstNext)
                        {
                            seekingFirstNext = false;
                            verticesUntilBump = (ushort)queue.Count;
                        }
                    }

                    if (localProblemId != isDiscovered[vertexAfterTransitionB])
                    {
                        if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                        {
                            discoveredSingleton = true;
                            break;
                        }

                        isDiscovered[vertexAfterTransitionB] = localProblemId;
                        queue.Enqueue(vertexAfterTransitionB);

                        if (seekingFirstNext)
                        {
                            seekingFirstNext = false;
                            verticesUntilBump = (ushort)queue.Count;
                        }
                    }
                }
#if (benchmark)

                benchmarkTiming.Stop();
#endif

                // finished main loop

                results[problem - beginningIndex] = new ComputationResult()
                {
                    benchmarkResult = new BenchmarkResult(),
                    computationType = ComputationType.CPU_Parallel,
                    //queueBreadth = maximumBreadth,
                    size = n,
                    algorithmName = GetType().Name,
                    isSynchronizable = discoveredSingleton
                    //discoveredVertices = isDiscovered.Sum(vertex => vertex ? 1 : 0)
                };

                if (discoveredSingleton)
                {
                    results[problem - beginningIndex].shortestSynchronizingWordLength = currentNextDistance;
                    if (currentNextDistance > maximumPermissibleWordLength)
                        throw new Exception("Cerny conjecture is false");
                }

            }
#if (benchmark)
            results[0].benchmarkResult.benchmarkedTime = benchmarkTiming.Elapsed;
            results[0].benchmarkResult.totalTime = totalTiming.Elapsed;
#endif
            return results;
        }
    }
}