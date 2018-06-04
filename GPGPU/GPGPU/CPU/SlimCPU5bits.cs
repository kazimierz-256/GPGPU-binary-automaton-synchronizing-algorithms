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
    public class SlimCPU5bits : IComputable
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
            uint vertexAfterTransition;
            var maskN = (1 << n) - 1;

            ushort currentNextDistance;
            ushort verticesUntilBump;
            bool seekingFirstNext;
            bool discoveredSingleton;
            byte i, i2;
            var bits = 5;
            var twoToPowerBits = (byte)(1 << bits);
            byte iMax = (byte)(twoToPowerBits * ((n + bits - 1) / bits));
            uint tmpTransition;

            var precomputedStateTransitioningMatrix = new uint[2 * n];
            var transitionMatrixCombined = new uint[iMax];

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

                for (i = 0, i2 = 1; i < n; i++, i2 += 2)
                {
                    precomputedStateTransitioningMatrix[i2] = (uint)(
                        (powerSetCount << problemsToSolve[readingId].stateTransitioningMatrixA[i])
                        | (1 << problemsToSolve[readingId].stateTransitioningMatrixB[i])
                        );
                }

                for (i = 0, i2 = 0; i < iMax; i += 32, i2 += 10)
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
                    //for (int k = 1; k < twoToPowerBits; k++)
                    //{
                    //    var tmp = 0u;
                    //    for (int b = 0, kreduced = k; b < bits; b++, kreduced >>= 1)
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

                    transitionMatrixCombined[i + 0b00001]
                  = transitionMatrixCombined[i + 0b00011]
                  = transitionMatrixCombined[i + 0b00101]
                  = transitionMatrixCombined[i + 0b00111]
                  = transitionMatrixCombined[i + 0b01001]
                  = transitionMatrixCombined[i + 0b01011]
                  = transitionMatrixCombined[i + 0b01101]
                  = transitionMatrixCombined[i + 0b01111]
                  = transitionMatrixCombined[i + 0b10001]
                  = transitionMatrixCombined[i + 0b10011]
                  = transitionMatrixCombined[i + 0b10101]
                  = transitionMatrixCombined[i + 0b10111]
                  = transitionMatrixCombined[i + 0b11001]
                  = transitionMatrixCombined[i + 0b11011]
                  = transitionMatrixCombined[i + 0b11101]
                  = transitionMatrixCombined[i + 0b11111]
                  = precomputedStateTransitioningMatrix[i2 + 0b00001];


                    if (i2 + 0b00011 >= n2)
                        break;
                    tmpTransition = precomputedStateTransitioningMatrix[i2 + 0b00011];
                    transitionMatrixCombined[i + 0b00010] = tmpTransition;
                    transitionMatrixCombined[i + 0b00011] |= tmpTransition;
                    transitionMatrixCombined[i + 0b00110] = tmpTransition;
                    transitionMatrixCombined[i + 0b00111] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01010] = tmpTransition;
                    transitionMatrixCombined[i + 0b01011] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01110] = tmpTransition;
                    transitionMatrixCombined[i + 0b01111] |= tmpTransition;

                    transitionMatrixCombined[i + 0b10010] = tmpTransition;
                    transitionMatrixCombined[i + 0b10011] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10110] = tmpTransition;
                    transitionMatrixCombined[i + 0b10111] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11010] = tmpTransition;
                    transitionMatrixCombined[i + 0b11011] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11110] = tmpTransition;
                    transitionMatrixCombined[i + 0b11111] |= tmpTransition;


                    if (i2 + 0b00101 >= n2)
                        break;
                    tmpTransition = precomputedStateTransitioningMatrix[i2 + 0b00101];
                    transitionMatrixCombined[i + 0b00100] = tmpTransition;
                    transitionMatrixCombined[i + 0b00101] |= tmpTransition;
                    transitionMatrixCombined[i + 0b00110] |= tmpTransition;
                    transitionMatrixCombined[i + 0b00111] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01100] = tmpTransition;
                    transitionMatrixCombined[i + 0b01101] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01110] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01111] |= tmpTransition;

                    transitionMatrixCombined[i + 0b10100] = tmpTransition;
                    transitionMatrixCombined[i + 0b10101] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10110] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10111] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11100] = tmpTransition;
                    transitionMatrixCombined[i + 0b11101] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11110] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11111] |= tmpTransition;


                    if (i2 + 0b00111 >= n2)
                        break;
                    tmpTransition = precomputedStateTransitioningMatrix[i2 + 0b00111];
                    transitionMatrixCombined[i + 0b01000] = tmpTransition;
                    transitionMatrixCombined[i + 0b01001] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01010] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01011] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01100] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01101] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01110] |= tmpTransition;
                    transitionMatrixCombined[i + 0b01111] |= tmpTransition;

                    transitionMatrixCombined[i + 0b11000] = tmpTransition;
                    transitionMatrixCombined[i + 0b11001] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11010] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11011] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11100] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11101] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11110] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11111] |= tmpTransition;


                    if (i2 + 0b01001 >= n2)
                        break;
                    tmpTransition = precomputedStateTransitioningMatrix[i2 + 0b01001];
                    transitionMatrixCombined[i + 0b10000] = tmpTransition;
                    transitionMatrixCombined[i + 0b10001] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10010] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10011] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10100] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10101] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10110] |= tmpTransition;
                    transitionMatrixCombined[i + 0b10111] |= tmpTransition;

                    transitionMatrixCombined[i + 0b11000] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11001] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11010] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11011] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11100] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11101] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11110] |= tmpTransition;
                    transitionMatrixCombined[i + 0b11111] |= tmpTransition;
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
                    vertexAfterTransition = transitionMatrixCombined[31 & consideringVertex]
                    | transitionMatrixCombined[32 + (31 & (consideringVertex >> 5))]
                    | transitionMatrixCombined[64 + (31 & (consideringVertex >> 10))];
#else
                    vertexAfterTransition = transitionMatrixCombined[31 & consideringVertex];
                    if (32 < iMax)
                    {
                        consideringVertex >>= 5;
                        vertexAfterTransition |= transitionMatrixCombined[32 + (31 & consideringVertex)];
                        if (64 < iMax)
                        {
                            consideringVertex >>= 5;
                            vertexAfterTransition |= transitionMatrixCombined[64 + (31 & consideringVertex)];
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