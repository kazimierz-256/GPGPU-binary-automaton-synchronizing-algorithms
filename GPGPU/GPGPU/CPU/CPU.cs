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
    public class CPU : IComputable
    {
        private static readonly int mostProbablePowerSetCount = 1 << 13;
        private ushort[] previousVertexStatic = new ushort[mostProbablePowerSetCount];
        private bool[] previousLetterUsedEqualsBStatic = new bool[mostProbablePowerSetCount];

        public ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism)
        {
            if (degreeOfParallelism == 1)
            {
                return problemsToSolve
                    .Select(problem => ComputeOne(
                        problem,
                        true,
                        previousVertexStatic,
                        previousLetterUsedEqualsBStatic))
                    .ToArray();
            }
            else
            {
                var reusableLengths = degreeOfParallelism;
                var previousVertexReusables = Enumerable.Range(0, reusableLengths)
                    .Select(_ => new ushort[mostProbablePowerSetCount]).ToArray();
                var previousLetterUsedEqualsBReusables = Enumerable.Range(0, reusableLengths)
                    .Select(_ => new bool[mostProbablePowerSetCount]).ToArray();




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
                    .AsOrdered()
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


            var n = problemToSolve.size;
            var powerSetCount = 1 << n;
            var initialVertex = (ushort)(powerSetCount - 1);
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            ushort[] previousVertex;
            bool[] previousLetterUsedEqualsB;
            benchmarkTiming.Start();
            // about 20-30% of overall computation
            bool[] isDiscovered = new bool[powerSetCount];
            isDiscovered[initialVertex] = true;

            // HACK: forbid reusing of resources
            if (reuseResources)
            {
                // use whatever size is needed, they all should be consistent (of same size)
                if (previousVertexReusable.Length != powerSetCount)
                {
                    previousVertexReusable = new ushort[powerSetCount];
                    previousLetterUsedEqualsBReusable = new bool[powerSetCount];
                }
                previousVertex = previousVertexReusable;
                previousLetterUsedEqualsB = previousLetterUsedEqualsBReusable;
            }
            else
            {
                previousVertex = new ushort[powerSetCount];
                previousLetterUsedEqualsB = new bool[powerSetCount];
            }
            benchmarkTiming.Stop();


            var queue = new Queue<ushort>(n * 5);
            queue.Enqueue(initialVertex);

            var discoveredSingleton = false;
            ushort consideringVertex;
            ushort vertexAfterTransitionA;
            ushort vertexAfterTransitionB;
            int targetIndexPlusOne;
            int extractingBits;
            ushort firstSingleton = 0;
            ushort firstSingletonDistance = 0;

            var precomputedStateTransitioningMatrixA = new ushort[n + 1];
            var precomputedStateTransitioningMatrixB = new ushort[n + 1];

            //Trace.WriteLine($"New Alg -----------------------");
            for (int i = 0; i < n; i++)
            {
                precomputedStateTransitioningMatrixA[i + 1] = (ushort)(1 << problemToSolve.stateTransitioningMatrixA[i]);
                precomputedStateTransitioningMatrixB[i + 1] = (ushort)(1 << problemToSolve.stateTransitioningMatrixB[i]);
                //Trace.WriteLine($"precomputedA[{i+1}] = {precomputedStateTransitioningMatrixA[i + 1]}");
                //Trace.WriteLine($"precomputedB[{i+1}] = {precomputedStateTransitioningMatrixB[i + 1]}");
            }

            var maximumBreadth = 0;
            ushort bumpUpVertex = 0;
            ushort currentNextDistance = 1;
            bool seekingFirstNext = true;
            while (queue.Count > 0)
            {
                if (queue.Count > maximumBreadth)
                    maximumBreadth = queue.Count;

                extractingBits = consideringVertex = queue.Dequeue();
                //Trace.WriteLine($"current {consideringVertex}");
                if (consideringVertex == bumpUpVertex)
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
                    targetIndexPlusOne = (extractingBits & 1) * i;
                    //Trace.WriteLine($"matrixA  {precomputedStateTransitioningMatrixA[targetIndexPlusOne]} indexplusone {targetIndexPlusOne}");
                    //Trace.WriteLine($"matrixB {precomputedStateTransitioningMatrixB[targetIndexPlusOne]} indexplusone {targetIndexPlusOne}");
                    vertexAfterTransitionA |= precomputedStateTransitioningMatrixA[targetIndexPlusOne];
                    vertexAfterTransitionB |= precomputedStateTransitioningMatrixB[targetIndexPlusOne];
                    extractingBits >>= 1;
                }
                //Trace.WriteLine($"a {vertexAfterTransitionA}");
                //Trace.WriteLine($"b {vertexAfterTransitionB}");
                if (!isDiscovered[vertexAfterTransitionA])
                {
                    if (seekingFirstNext)
                    {
                        seekingFirstNext = false;
                        bumpUpVertex = vertexAfterTransitionA;
                    }
                    isDiscovered[vertexAfterTransitionA] = true;
                    previousVertex[vertexAfterTransitionA] = consideringVertex;
                    previousLetterUsedEqualsB[vertexAfterTransitionA] = false;
                    queue.Enqueue(vertexAfterTransitionA);

                    if (0 == (vertexAfterTransitionA & (vertexAfterTransitionA - 1)))
                    {
                        discoveredSingleton = true;
                        firstSingleton = vertexAfterTransitionA;
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
                    previousVertex[vertexAfterTransitionB] = consideringVertex;
                    previousLetterUsedEqualsB[vertexAfterTransitionB] = true;
                    queue.Enqueue(vertexAfterTransitionB);

                    if (0 == (vertexAfterTransitionB & (vertexAfterTransitionB - 1)))
                    {
                        discoveredSingleton = true;
                        firstSingleton = vertexAfterTransitionB;
                        firstSingletonDistance = currentNextDistance;
                        break;
                    }
                }
            }

            // finished main loop

            var result = new ComputationResult()
            {
                benchmarkResult = new BenchmarkResult(),
                computationType = ComputationType.CPU_Serial,
                queueBreadth = maximumBreadth,
                size = n,
                algorithmName = GetType().Name
                //discoveredVertices = isDiscovered.Sum(vertex => vertex ? 1 : 0)
            };

            if (discoveredSingleton)
            {
                // watch out for off by one error!
                if (firstSingletonDistance > maximumPermissibleWordLength)
                {
                    throw new Exception("Cerny conjecture is false");
                }
                else
                {
                    // everything is fine
                    result.isSynchronizable = true;

                    var wordLength = firstSingletonDistance;
                    var currentVertex = firstSingleton;
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
                // is this correctly computed?
                result.isSynchronizable = false;
            }

            result.shortestSynchronizingWordLength = firstSingletonDistance;
            result.benchmarkResult.benchmarkedTime = benchmarkTiming.Elapsed;
            result.benchmarkResult.totalTime = totalTiming.Elapsed;
            return result;
        }
        public int GetBestParallelism() => 2;

    }
}
