//#define benchmark
//#define optimizeFor16
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
    public class SlimCPUCoalesced : IComputable
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
            var n2 = (byte)(n << 1);
            var n3 = (byte)(n * 3);
            var n2m1 = (byte)(n2 - 1);
            var n4 = (byte)(n << 2);
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


            ushort consideringVertex;
            ushort vertexAfterTransitionA;
            ushort vertexAfterTransitionB;
            ulong vertexAfterTransition;
            var maskN = (ulong)((1 << n) - 1);

            ushort currentNextDistance;
            ushort verticesUntilBump;
            bool seekingFirstNext;
            bool discoveredSingleton;
            byte i, i2;
            var bits = 4;
            var twoToPowerBits = (byte)(1 << bits);
            byte iMax = (byte)(twoToPowerBits * ((n + bits - 1) / bits));
            ulong tmpTransition;

            var precomputedStateTransitioningMatrix = new ulong[2 * n];
            var transitionMatrixCombined = new ulong[iMax];
            bool isCreativeId;
            if ((problemCount & 1) != 0)
            {
                throw new Exception("For now, only even number of problems is supported");
            }
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
                isCreativeId = (localProblemId & 1) == 1;

                if (isCreativeId)
                {
                    for (i = 0, i2 = 1; i < n; i++, i2 += 2)
                    {
                        precomputedStateTransitioningMatrix[i2] = (1ul << problemsToSolve[readingId].stateTransitioningMatrixB[i])
                            | (1ul << (problemsToSolve[readingId].stateTransitioningMatrixA[i] + n))
                            | (1ul << (problemsToSolve[readingId + 1].stateTransitioningMatrixB[i] + n2))
                            | (1ul << (problemsToSolve[readingId + 1].stateTransitioningMatrixA[i] + n3));
                    }

                    for (i = 0, i2 = 0; i < iMax; i += 16, i2 += 8)
                    {
                        transitionMatrixCombined[i + 0b0001]
                          = transitionMatrixCombined[i + 0b0011]
                          = transitionMatrixCombined[i + 0b0101]
                          = transitionMatrixCombined[i + 0b0111]
                          = transitionMatrixCombined[i + 0b1001]
                          = transitionMatrixCombined[i + 0b1011]
                          = transitionMatrixCombined[i + 0b1101]
                          = transitionMatrixCombined[i + 0b1111]
                          = precomputedStateTransitioningMatrix[i2 + 0b0001];
                        if (i2 + 0b0011 >= n2)
                            break;
                        tmpTransition = precomputedStateTransitioningMatrix[i2 + 0b0011];
                        transitionMatrixCombined[i + 0b0010] = tmpTransition;
                        transitionMatrixCombined[i + 0b0011] |= tmpTransition;
                        transitionMatrixCombined[i + 0b0110] = tmpTransition;
                        transitionMatrixCombined[i + 0b0111] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1010] = tmpTransition;
                        transitionMatrixCombined[i + 0b1011] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1110] = tmpTransition;
                        transitionMatrixCombined[i + 0b1111] |= tmpTransition;
                        if (i2 + 0b0101 >= n2)
                            break;
                        tmpTransition = precomputedStateTransitioningMatrix[i2 + 0b0101];
                        transitionMatrixCombined[i + 0b0100] = tmpTransition;
                        transitionMatrixCombined[i + 0b0101] |= tmpTransition;
                        transitionMatrixCombined[i + 0b0110] |= tmpTransition;
                        transitionMatrixCombined[i + 0b0111] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1100] = tmpTransition;
                        transitionMatrixCombined[i + 0b1101] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1110] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1111] |= tmpTransition;
                        if (i2 + 0b0111 >= n2)
                            break;
                        tmpTransition = precomputedStateTransitioningMatrix[i2 + 0b0111];
                        transitionMatrixCombined[i + 0b1000] = tmpTransition;
                        transitionMatrixCombined[i + 0b1001] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1010] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1011] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1100] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1101] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1110] |= tmpTransition;
                        transitionMatrixCombined[i + 0b1111] |= tmpTransition;
                    }
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

#if (optimizeFor16)
                    vertexAfterTransition = transitionMatrixCombined[15 & consideringVertex]
                    | transitionMatrixCombined[16 + (15 & (consideringVertex >> 4))]
                    | transitionMatrixCombined[32 + (15 & (consideringVertex >> 8))]
                    | transitionMatrixCombined[48 + (15 & (consideringVertex >> 12))];
#else
                    vertexAfterTransition = transitionMatrixCombined[15 & consideringVertex];
                    if (16 < iMax)
                    {
                        consideringVertex >>= 4;
                        vertexAfterTransition |= transitionMatrixCombined[16 + (15 & consideringVertex)];
                        if (32 < iMax)
                        {
                            consideringVertex >>= 4;
                            vertexAfterTransition |= transitionMatrixCombined[32 + (15 & consideringVertex)];
                            if (48 < iMax)
                            {
                                consideringVertex >>= 4;
                                vertexAfterTransition |= transitionMatrixCombined[48 + (15 & consideringVertex)];
                            }
                        }
                    }
#endif
                    if (isCreativeId)
                    {
                        vertexAfterTransitionA = (ushort)((vertexAfterTransition >> n) & maskN);
                        vertexAfterTransitionB = (ushort)(vertexAfterTransition & maskN);
                    }
                    else
                    {
                        vertexAfterTransitionA = (ushort)((vertexAfterTransition >> n3) & maskN);
                        vertexAfterTransitionB = (ushort)((vertexAfterTransition >> n2) & maskN);
                    }
                    
                    // check for singleton existance
                    // b && !(b & (b-1)) https://stackoverflow.com/questions/12483843/test-if-a-bitboard-have-only-one-bit-set-to-1
                    // note: consideringVertex cannot ever be equal to 0

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