//#define benchmark
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

        public void Compute(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount, int degreeOfParallelism)
        {
            if (problemCount == 0)
                return;

            if (degreeOfParallelism > problemCount)
                degreeOfParallelism = problemCount;

            if (degreeOfParallelism == 1)
            {
                ComputeMany(problemsToSolve, problemsReadingIndex, computationResults, resultsWritingIndex, problemCount);
            }
            else
            {
                var partition = (problemCount + degreeOfParallelism - 1) / degreeOfParallelism;
                Parallel.For(0, degreeOfParallelism, i =>
                {
                    ComputeMany(
                        problemsToSolve,
                        problemsReadingIndex + i * partition,
                        computationResults,
                        resultsWritingIndex + i * partition,
                        Math.Min(partition, problemCount - i * partition)
                        );
                });
            }
        }

        private void ComputeMany(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount)
        {
#if (benchmark)
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();
#endif

            var n = problemsToSolve[problemsReadingIndex].size;
            var powerSetCount = 1 << n;
            var initialVertex = (ushort)(powerSetCount - 1);
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            byte localProblemId = 1;
            var isDiscovered = new byte[powerSetCount];

            var queue = new Queue<ushort>(n * 2);

            var precomputedStateTransitioningMatrixA = new ushort[n];
            var precomputedStateTransitioningMatrixB = new ushort[n];
            for (int problemId = 0, readingId = problemsReadingIndex, writingId = resultsWritingIndex; problemId < problemCount; problemId++, localProblemId++, readingId++, writingId++)
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
                    precomputedStateTransitioningMatrixA[i] = (ushort)(1 << problemsToSolve[readingId].stateTransitioningMatrixA[i]);
                    precomputedStateTransitioningMatrixB[i] = (ushort)(1 << problemsToSolve[readingId].stateTransitioningMatrixB[i]);
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

                computationResults[writingId] = new ComputationResult()
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
                    computationResults[writingId].shortestSynchronizingWordLength = currentNextDistance;
                    if (currentNextDistance > maximumPermissibleWordLength)
                        throw new Exception("Cerny conjecture is false");
                }

            }
#if (benchmark)
            computationResults[resultsWritingIndex].benchmarkResult.benchmarkedTime = benchmarkTiming.Elapsed;
            computationResults[resultsWritingIndex].benchmarkResult.totalTime = totalTiming.Elapsed;
#endif
        }
    }
}