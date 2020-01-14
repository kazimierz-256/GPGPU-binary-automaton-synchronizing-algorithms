//#define benchmark
//#define optimizeFor16
#define ignoreOutputArray
#define makeNconstant13
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
    public class SlimCPU4bits13 : IComputable
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

#if (makeNconstant13)
            const int n = 12;
#else
            var n = problemsToSolve[problemsReadingIndex].size;
#endif
            var n2 = (byte)(n << 1);
            var n2m1 = (byte)(n2 - 1);
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
            uint vertexAfterTransition;
            var maskN = (1 << n) - 1;

            ushort currentNextDistance;
            ushort verticesUntilBump;
            bool seekingFirstNext;
            bool discoveredSingleton;
            byte i, i2;
            const byte bits = 4;
            var twoToPowerBits = (byte)(1 << bits);
            byte iMax = (byte)(twoToPowerBits * ((n + bits - 1) / bits));
            uint tmpTransition;

            var precomputedStateTransitioningMatrix = new uint[2 * n];
            var transitionMatrixCombined = new uint[iMax];

            for (int problemId = 0, readingId = problemsReadingIndex, writingId = resultsWritingIndex; problemId < problemCount; problemId += 1, localProblemId += 1, readingId += 1, writingId += 1)
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

                for (i = 0, i2 = 1; i < n; i += 1, i2 += 2)
                {
                    precomputedStateTransitioningMatrix[i2] = (uint)(
                        (powerSetCount << problemsToSolve[readingId].stateTransitioningMatrixA[i])
                        | (1 << problemsToSolve[readingId].stateTransitioningMatrixB[i])
                        );
                }

                for (i = 0, i2 = 0; i < iMax; i += 16, i2 += 8)
                {
                    // first: i == 4 then
                    // tra[1]=pre[1]
                    // tra[2]=pre[3]
                    // tra[3]=pre[1]|pre[3]
                    // tra[4]=pre[5]
                    // tra[5]=pre[5]|pre[1]
                    // tra[6]=pre[5]|pre[3]
                    // tra[7]=pre[5]|pre[3]|pre[1]

                    // tra[9]=pre[7]
                    // tra[10]=pre[9]
                    // tra[11]=pre[7]|pre[9]

                    #region fullyAutomated_yet_inefficient
                    //for (int k = 1; k < twoToPowerBits; k+=1)
                    //{
                    //    var tmp = 0u;
                    //    for (int b = 0, kreduced = k; b < bits; b+=1, kreduced >>= 1)
                    //    {
                    //        if (i2 + 1 + 2 * b < n2)
                    //        {
                    //            tmp |= precomputedStateTransitioningMatrix[i2 + (kreduced & 1) + 2 * b];
                    //        }
                    //        else
                    //        {
                    //            break;
                    //        }
                    //    }
                    //    transitionMatrixCombined[i + k] = tmp;
                    //} 
                    #endregion

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

                //var maximumBreadth = 0;
                currentNextDistance = 1;
                verticesUntilBump = ushort.MaxValue;
                seekingFirstNext = true;
                discoveredSingleton = false;

                while (readingIndex < writingIndex)
                {
                    //if (queue.Count > maximumBreadth)
                    //    maximumBreadth = queue.Count;

                    consideringVertex = queue[readingIndex];
                    readingIndex += 1;

                    if ((verticesUntilBump -= 1) == 0)
                    {
                        currentNextDistance += 1;
                        seekingFirstNext = true;
                    }

                    //vertexAfterTransition = 0;
                    // check for singleton existance
                    // b && !(b & (b-1)) https://stackoverflow.com/questions/12483843/test-if-a-bitboard-have-only-one-bit-set-to-1
                    // note: consideringVertex cannot ever be equal to 0

                    // watch out for the index range in the for loop
                    //for (i = 0; i < n2; i += 2, consideringVertex >>= 1)
                    //{
                    //    vertexAfterTransition |= precomputedStateTransitioningMatrix[i + (1 & consideringVertex)];
                    //}

                    #region fully_automated
                    //for (i = 0; i < iMax; i += twoToPowerBits, consideringVertex >>= bits)
                    //{
                    //    vertexAfterTransition |= transitionMatrixCombined[i + (mask & consideringVertex)];
                    //} 
                    #endregion

#if (optimizeFor16)
                    vertexAfterTransition = transitionMatrixCombined[15 & consideringVertex]
                    | transitionMatrixCombined[16 + (15 & (consideringVertex >> 4))]
                    | transitionMatrixCombined[32 + (15 & (consideringVertex >> 8))]
                    | transitionMatrixCombined[48 + (15 & (consideringVertex >> 12))];
#else
                    vertexAfterTransition = transitionMatrixCombined[15 & consideringVertex];
                    if (16 < iMax)
                    {
                        vertexAfterTransition |= transitionMatrixCombined[16 + (15 & (consideringVertex >> 4))];
                        if (32 < iMax)
                        {
                            vertexAfterTransition |= transitionMatrixCombined[32 + (15 & (consideringVertex >> 8))];
                            if (48 < iMax)
                            {
                                vertexAfterTransition |= transitionMatrixCombined[48 + (15 & (consideringVertex >> 12))];
                            }
                        }
                    }
#endif
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
                        queue[writingIndex] = vertexAfterTransitionA;
                        writingIndex += 1;

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
                        queue[writingIndex] = vertexAfterTransitionB;
                        writingIndex += 1;

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

#if (ignoreOutputArray)
                if (discoveredSingleton && currentNextDistance > maximumPermissibleWordLength)
                    return writingId;
#else

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

#endif

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