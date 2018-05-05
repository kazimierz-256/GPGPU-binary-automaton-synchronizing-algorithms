using Alea;
using Alea.CSharp;
using Alea.FSharp;
using GPGPU.Interfaces;
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
            var theSolver = new IComputable[]
            {
                new CPU(),
                new SlimCPU(),
                //new SlimGPUBuggy(),
                //new SlimCPUGPU(),
                new SlimGPUQueue(),
            };
            const long initialProblemSamplingCount = 1 << 16;
            double sizeIncrease = 1;// Math.Pow(2, 1d / 2);
            #endregion

            const int problemSeed = 123456;
            var random = new Random(problemSeed);
            var watch = new Stopwatch();

            var version = theSolver.GetType().Namespace;
            var csvBuilder = new StringBuilder("problemcount,cputime,gputime,cpugpucombinedtime");
            var resultsDictionary = new List<ComputationResult>();

            // in a loop check the performance of the CPU
            double doublePrecisionN = initialProblemSamplingCount;
            Console.WriteLine("Heating up the GPU... please stand by");
            Gpu.Default.Launch(() => { }, new LaunchParam(1, 1));
            for (int n = (int)doublePrecisionN; ; n = (int)Math.Round(doublePrecisionN *= sizeIncrease))
            {
                var localSeed = random.Next();
                foreach (var solver in theSolver)
                    computeLoopUsing(solver);

                Console.WriteLine();
                Console.WriteLine();
                void computeLoopUsing(IComputable solver)
                {
                    var problems = Problem.GetArrayOfProblems(n, problemSize, problemSeed);
                    //var problems = new[] { Problem.GenerateWorstCase(problemSize) };
                    //var problems = Problem.GetArrayOfProblems(16, 3, 123456).Skip(10).Take(6);

                    watch.Start();
                    var results = solver.Compute(problems, solver.GetBestParallelism());
                    watch.Stop();

                    var computationElapsed = watch.Elapsed;
                    var benchmarkedResults = results.Where(result => result.benchmarkResult != null && result.benchmarkResult.benchmarkedTime != null && result.benchmarkResult.totalTime != null);
                    var fractionOfTime = benchmarkedResults.Sum(result => result.benchmarkResult.benchmarkedTime.TotalMilliseconds)
                        / benchmarkedResults.Sum(result => result.benchmarkResult.totalTime.TotalMilliseconds);
                    Console.WriteLine(fractionOfTime);
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

                    Console.WriteLine($"{solver.GetType().ToString()} {n} problems computed using {solver.GetBestParallelism()} processors in {computationElapsed.TotalMilliseconds:F2}ms. " +
                        $"Problems per second: {n / computationElapsed.TotalSeconds:F2}. " +
                        $"Time per problem {computationElapsed.TotalMilliseconds / n}ms");

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
