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
            //var n = 1 << 18;
            var theSolver = new IComputable[]
            {
                new SlimGPUAllAtOnce(),
                //new CPU(),
                new SlimCPU(),
                //new SlimCPUGPUInbetween(),
                new SlimGPUQueue(),
            };
            //double sizeIncrease = 1;// Math.Pow(2, 1d / 2);
            #endregion
            Gpu.Default.Device.Print();
            const int problemSeed = 1234567;
            var random = new Random(problemSeed);
            var watch = new Stopwatch();

            var version = theSolver.GetType().Namespace;
            var csvBuilder = new StringBuilder("problemsize,problemcount");
            foreach (var solver in theSolver)
            {
                csvBuilder.Append(",").Append(solver.GetType().Name);
            }
            var resultsDictionary = new List<ComputationResult>();

            var sizeIncrease = Math.Sqrt(2);
            var initialProblemSamplingCount = 1 << 14;
            var problemCount = 1 << 18;
            // in a loop check the performance of the CPU
            double doublePrecisionN = initialProblemSamplingCount;
            for (int n = (int)doublePrecisionN; n < problemCount; n = (int)Math.Round(doublePrecisionN *= sizeIncrease))
            //for (int i = 0; i < 10; i++)
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
                    watch.Restart();
                    var results = solver.Compute(problems, solver.GetBestParallelism());
                    watch.Stop();
                    var summary = new ComputationResultSummary();
                    // TODO: fill in summary
                    // TODO: pump summary into csv builder
                    // TODO: export to csv
                    // TODO: create a google docs
                    var computationElapsed = watch.Elapsed;
                    var benchmarkedResults = results.Where(result => result.benchmarkResult != null && result.benchmarkResult.benchmarkedTime != null && result.benchmarkResult.totalTime != null);
                    var computationToTotalFraction = benchmarkedResults.Sum(result => result.benchmarkResult.benchmarkedTime.TotalMilliseconds)
                        / benchmarkedResults.Sum(result => result.benchmarkResult.totalTime.TotalMilliseconds);
                    Console.WriteLine(computationToTotalFraction);
                    //watch.Reset();
                    //watch.Start();
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
                    //watch.Stop();

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

                    //save appropriate statistical data to a file from resultsDictionary
                    System.IO.File.WriteAllText($@"./gpgpu.csv", csvBuilder.ToString());
                    Console.WriteLine();
                    Console.WriteLine();
                }


            }
        }
    }
}
