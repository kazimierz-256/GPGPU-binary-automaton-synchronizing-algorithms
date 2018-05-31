//#define benchmark
using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
namespace GPGPU
{
    public class SlimCPU : IComputable
    {
        public int GetBestParallelism() => Environment.ProcessorCount;

        /// <returns>True if Cerny conjecture is successfully disproved, results may be incomplete in that case</returns>
        public int Compute(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount, int degreeOfParallelism)
        {
            if (problemCount == 0)
                return -1;

            if (degreeOfParallelism > problemCount)
                degreeOfParallelism = problemCount;

            if (degreeOfParallelism == 1)
            {
                return ComputeMany(problemsToSolve, problemsReadingIndex, computationResults, resultsWritingIndex, problemCount);
            }
            else
            {
                var partition = (problemCount + degreeOfParallelism - 1) / degreeOfParallelism;
                int CernyConjectureFailingIndex = -1;
                Parallel.For(0, degreeOfParallelism, i =>
                {
                    var result = ComputeMany(
                        problemsToSolve,
                        problemsReadingIndex + i * partition,
                        computationResults,
                        resultsWritingIndex + i * partition,
                        Math.Min(partition, problemCount - i * partition)
                        );
                    if (result >= 0)
                        Interlocked.CompareExchange(ref CernyConjectureFailingIndex, result, -1);
                });
                return CernyConjectureFailingIndex;
            }
        }

        private int ComputeMany(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount)
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

            // CPU has lots of memory, so we can be generous

            uint localProblemId = 1;
            var isDiscovered = new uint[powerSetCount];

            var queue = new ushort[powerSetCount];
            // always the same, no need to change this while in a loop
            queue[0] = initialVertex;
            ushort readingIndex = 0;
            ushort writingIndex = 0;

            var precomputedStateTransitioningMatrix = new uint[n + 1];

            ushort consideringVertex;
            ushort vertexAfterTransitionA;
            ushort vertexAfterTransitionB;
            uint vertexAfterTransition;
            var maskN = (1 << n) - 1;

            ushort currentNextDistance;
            ushort verticesUntilBump;
            bool seekingFirstNext;
            bool discoveredSingleton;
            byte i;

            for (int problemId = 0, readingId = problemsReadingIndex, writingId = resultsWritingIndex; problemId < problemCount; problemId++, localProblemId++, readingId++, writingId++)
            {
#if (benchmark)
                benchmarkTiming.Start();
#endif

                // that's unprobable since 2^32-1 is a very large number of problems
                //if (localProblemId <= 0)
                //{
                //    localProblemId = 1;
                //    Array.Clear(isDiscovered, 0, isDiscovered.Length);
                //    //// should be faster than zeroing out an array
                //    //isDiscovered = new byte[powerSetCount];
                //}
                isDiscovered[initialVertex] = localProblemId;
                readingIndex = 0;
                writingIndex = 1;

                for (i = 0; i < n; i++)
                {
                    precomputedStateTransitioningMatrix[i + 1] = (uint)(
                        (1 << problemsToSolve[readingId].stateTransitioningMatrixA[i] + n)
                        | (1 << problemsToSolve[readingId].stateTransitioningMatrixB[i])
                        );
                }

                //var maximumBreadth = 0;
                currentNextDistance = 1;
                verticesUntilBump = ushort.MaxValue;
                seekingFirstNext = true;
                discoveredSingleton = false;

                while (readingIndex < writingIndex)
                {
                    //if (queue.Count > maximumBreadth)
                    //    maximumBreadth = queue.Count;

                    consideringVertex = queue[readingIndex++];

                    if (--verticesUntilBump == 0)
                    {
                        ++currentNextDistance;
                        seekingFirstNext = true;
                    }

                    vertexAfterTransition = 0;
                    // check for singleton existance
                    // b && !(b & (b-1)) https://stackoverflow.com/questions/12483843/test-if-a-bitboard-have-only-one-bit-set-to-1
                    // note: consideringVertex cannot ever be equal to 0

                    // watch out for the index range in the for loop
                    for (i = 1; i <= n; i++, consideringVertex >>= 1)
                    {
                        vertexAfterTransition |= precomputedStateTransitioningMatrix[i * (1 & consideringVertex)];
                    }

                    vertexAfterTransitionA = (ushort)(vertexAfterTransition >> n);
                    vertexAfterTransitionB = (ushort)(vertexAfterTransition & maskN);

                    if (localProblemId != isDiscovered[vertexAfterTransitionA])
                    {
                        if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                        {
                            discoveredSingleton = true;
                            break;
                        }

                        isDiscovered[vertexAfterTransitionA] = localProblemId;
                        queue[writingIndex++] = vertexAfterTransitionA;

                        if (seekingFirstNext)
                        {
                            seekingFirstNext = false;
                            verticesUntilBump = (ushort)(writingIndex - readingIndex);
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
                        queue[writingIndex++] = vertexAfterTransitionB;

                        if (seekingFirstNext)
                        {
                            seekingFirstNext = false;
                            verticesUntilBump = (ushort)(writingIndex - readingIndex);
                        }
                    }
                }
#if (benchmark)

                benchmarkTiming.Stop();
#endif
                if (computationResults == null)
                {
                    if (discoveredSingleton && currentNextDistance > maximumPermissibleWordLength)
                        return writingId;
                }
                else
                {
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
                            return writingId;
                    }
                }

            }
#if (benchmark)
            computationResults[resultsWritingIndex].benchmarkResult.benchmarkedTime = benchmarkTiming.Elapsed;
            computationResults[resultsWritingIndex].benchmarkResult.totalTime = totalTiming.Elapsed;
#endif
            return -1;
        }

        public int Verify(Problem[] problemsToSolve, int problemsReadingIndex, int problemCount, int degreeOfParallelism)
            => Compute(problemsToSolve, problemsReadingIndex, null, -1, problemCount, degreeOfParallelism);
    }
}