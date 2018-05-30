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
using System.IO;
using System.Linq;
using System.Security;
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
                new SlimCPU(),
                new SlimGPUQueue(),
                new SlimCPUGPU(),
                new SuperSlimGPUBreakthrough()
            };
            #endregion

            #region creditentials
            var email = string.Empty;
            var password = new SecureString();
            Console.WriteLine("Send email notifications? (y/n)");
            if (Console.ReadKey().KeyChar.ToString().ToLower().Equals("y"))
            {
                
                Console.WriteLine();
                Console.WriteLine("Please enter notification email: (will send to self)");
                email = Console.ReadLine();
                Console.WriteLine($"Please enter the creditentials for the selected email");
                foreach (var ch in Console.ReadLine())
                {
                    password.AppendChar(ch);
                }
                Console.Clear();
                Console.WriteLine("Creditentials saved");
            }
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
            //var resultsDictionary = new List<ComputationResult>();

            var sizeIncrease = Math.Pow(2, 1d / 8d);
            var initialProblemSamplingCount = 1 << 17;
            var maximalProblemCount = 1 << 21;

            double doublePrecisionN = initialProblemSamplingCount;
            var problems = new Problem[0];
            for (int n = (int)doublePrecisionN; n < maximalProblemCount; n = (int)Math.Round(doublePrecisionN *= sizeIncrease))
            //for (int i = 0; i < 10; i++)
            {
                Console.WriteLine();
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"{n} problems");
                Console.ResetColor();
                Console.WriteLine();

                var latestCPUPerformance = TimeSpan.FromSeconds(1);
                var latestGPUPerformance = TimeSpan.FromSeconds(1);
                csvBuilder.AppendLine();
                csvBuilder.Append(problemSize).Append(",").Append(n);
                if (problems.Length != n)
                {
                    problems = new Problem[n];
                    Problem.FillArrayOfProblems(problems, n, problemSize, problemSeed + n);
                }
                foreach (var solver in theSolver)
                {
                    if (computeLoopUsing(solver))
                    {
                        return;
                    };
                }

                bool computeLoopUsing(IComputable solver)
                {
                    //var problems = new[] { Problem.GenerateWorstCase(problemSize) };
                    //var problems = Problem.GetArrayOfProblems(16, 3, 123456).Skip(10).Take(6);

                    bool Compute(IComputable localSolver, Problem[] localProblems, ComputationResult[] localResults)
                    {
                        var result = localSolver.Verify(problems, 0, problems.Length, localSolver.GetBestParallelism());
                        if (result >= 0)
                        {
                            File.AppendAllText($@"./cernyFailed.csv", problems[result].ToString());

                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"Cerny Conjecture is false.");
                            Console.WriteLine($"Problem description: {problems[result]}");
                            Console.ResetColor();

                            new Notifications.OpenBrowserTab().Notify(problems[result]);
                            new Notifications.SendEmail().Notify(problems[result], email, password);

                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }

                    var results = new ComputationResult[problems.Length];
                    if (solver is SlimCPUGPU)
                    {
                        ((SlimCPUGPU)solver).SetCPUPart((float)(latestCPUPerformance.TotalMilliseconds / (latestCPUPerformance.TotalMilliseconds + latestGPUPerformance.TotalMilliseconds)));
                    }
                    watch.Restart();
                    if (Compute(solver, problems, results))
                    {
                        return true;
                    }
                    watch.Stop();
                    var computationElapsed = watch.Elapsed;
                    if (solver is SlimCPU)
                    {
                        latestCPUPerformance = computationElapsed;
                    }
                    else if (solver is SlimGPUQueue)
                    {
                        latestGPUPerformance = computationElapsed;
                    }
                    //var summary = new ComputationResultSummary();
                    csvBuilder.Append(",").Append(Math.Round(n / computationElapsed.TotalSeconds));
                    //var benchmarkkedResults = results.Where(result => result.benchmarkResult != null && result.benchmarkResult.benchmarkedTime != null && result.benchmarkResult.totalTime != null);
                    //var computationToTotalFraction = benchmarkedResults.Sum(result => result.benchmarkResult.benchmarkedTime.TotalMilliseconds)
                    //    / benchmarkedResults.Sum(result => result.benchmarkResult.totalTime.TotalMilliseconds);
                    //Console.WriteLine(computationToTotalFraction);
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

                    //resultsDictionary.AddRange(results);
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write(solver.GetType());
                    Console.ResetColor();
                    Console.Write($" using {solver.GetBestParallelism()} parallelism in {computationElapsed.TotalMilliseconds:F2}ms. Problems per second: ");
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write($"{n / computationElapsed.TotalSeconds:F2}");
                    Console.ResetColor();
                    Console.WriteLine($". Time per problem {computationElapsed.TotalMilliseconds / n}ms");

                    //Console.WriteLine($"{n} problems verified using {degreeOfParallelism} processors in {verificationElapsed.TotalMilliseconds:F2}ms. " +
                    //    $"Verifications per second: {n / verificationElapsed.TotalSeconds:F2}. " +
                    //    $"Time per verification {verificationElapsed.TotalMilliseconds / n:F5}ms");

                    //Console.WriteLine($"Summary: {results.Average(result => result.isSynchronizable ? 1 : 0) * 100:F2}% synchronizability, " +
                    //    $"{results.Where(result => result.isSynchronizable).Average(result => result.shortestSynchronizingWordLength):F2} average length of a synchronizing word");

                    //var lessThanOrEqualTo = 10;
                    //Console.WriteLine($"fraction of less or equal to {lessThanOrEqualTo} is {results.Select(result => result.shortestSynchronizingWordLength <= lessThanOrEqualTo ? 1 : 0).Average()}");

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
                    return false;
                }


            }
        }
    }
}
