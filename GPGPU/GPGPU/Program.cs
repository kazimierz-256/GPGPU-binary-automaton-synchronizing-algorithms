using GPGPU.Interfaces;
using GPGPU.Problem_generator;
using GPGPU.Result_veryfier;
using GPGPU.Shared;
using LinqStatistics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU
{
    class Program
    {
        static void Main(string[] args)
        {
            #region Program definitions
            const int problemSize = 13;
            IComputation theSolver = new SlimCPU();
            const long initialProblemSamplingCount = 1 << 10;
            double sizeIncrease = 2;// Math.Pow(2, 1d / 2);
            #endregion

            const int problemSeed = 123456;
            var watch = new Stopwatch();

            var version = theSolver.GetType().Namespace;
            var csvBuilder = new StringBuilder("problemcount,cputime,gputime,cpugpucombinedtime");
            var resultsDictionary = new List<ComputationResult>();
            var degreeOfParallelism = theSolver.GetBestParallelism();

            // in a loop check the performance of the CPU
            double doublePrecisionN = initialProblemSamplingCount;
            for (int n = (int)doublePrecisionN; n < 1000000; n = (int)Math.Round(doublePrecisionN *= sizeIncrease))
            {
                computeLoopUsing(theSolver);
                void computeLoopUsing(IComputation solver)
                {
                    var problems = Problem.GetArrayOfProblems(n, problemSize, problemSeed * n);
                    //var problems = new[] { ProblemGenerator.generateWorstCase(problemSize) };

                    watch.Start();
                    var results = solver.Compute(problems, degreeOfParallelism);
                    watch.Stop();

                    var computationElapsed = watch.Elapsed;
                    //var fractionOfTime = results.Sum(result => result.benchmarkResult.benchmarkedTime.TotalMilliseconds)
                    //    / results.Sum(result => result.benchmarkResult.totalTime.TotalMilliseconds);

                    watch.Reset();
                    watch.Start();
                    //if (!(
                    //        results.Zip(problems, (result, problem) =>
                    //            !result.isSynchronizable || Verify.VerifyValidityOfSynchronizingWord(problem, result, degreeOfParallelism)
                    //        ).All(isOK => isOK)
                    //        && 
                    //        results.Zip(problems, (result, problem) =>
                    //            Verify.VerifyCernyConjecture(problem, result)
                    //        ).All(isOK => isOK)
                    //    ))
                    //{
                    //    throw new Exception("Incorrect algorithm");
                    //}
                    watch.Stop();

                    var verificationElapsed = watch.Elapsed;

                    resultsDictionary.AddRange(results);

                    Console.WriteLine($"{n} problems computed using {degreeOfParallelism} processors in {computationElapsed.TotalMilliseconds:F2}ms. " +
                        $"Problems per second: {n / computationElapsed.TotalSeconds:F2}. " +
                        $"Time per problem {computationElapsed.TotalMilliseconds / n:F5}ms");

                    //Console.WriteLine($"{n} problems verified using {degreeOfParallelism} processors in {verificationElapsed.TotalMilliseconds:F2}ms. " +
                    //    $"Verifications per second: {n / verificationElapsed.TotalSeconds:F2}. " +
                    //    $"Time per verification {verificationElapsed.TotalMilliseconds / n:F5}ms");

                    //Console.WriteLine($"Summary: {results.Average(result => result.isSynchronizable ? 1 : 0) * 100:F2}% synchronizability, " +
                    //    $"{results.Where(result => result.isSynchronizable).Average(result => result.shortestSynchronizingWordLength):F2} average length of a synchronizing word");

                    //#region Histogram
                    //var histogram = results
                    //                .Where(result => result.isSynchronizable)
                    //                .Histogram(30, result => result.shortestSynchronizingWordLength);

                    //foreach (var bin in histogram)
                    //{
                    //    Console.Write($"{Math.Round(bin.RepresentativeValue)} (count: {bin.Count}): ");
                    //    for (int i = 0; i < bin.Count * 500 / n || (i == 0 && bin.Count > 0); i++)
                    //    {
                    //        Console.Write("-");
                    //    }
                    //    Console.WriteLine();
                    //}
                    //Console.WriteLine();
                    //#endregion

                    #region Benchmark
                    //Console.WriteLine($"benchmarked time took {100 * fractionOfTime:F2}%");
                    #endregion
                    //Console.WriteLine(results.Average(result => result.queueBreadth));

                    Console.WriteLine();
                    Console.WriteLine();
                }

                // in a loop check the performance of the GPU


                // in a loop check the performance of CPU + GPU combined!


                //save appropriate statistical data to a file
            }
        }
    }
}
