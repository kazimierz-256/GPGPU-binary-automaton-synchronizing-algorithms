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
            const int problemSize = 13;
            //int maximumExpectedLengthOfShortestSynchronizingWord = (problemSize - 1) * (problemSize - 1);
            const long initialProblemSamplingCount = 1 << 13;
            double sizeIncrease = 2;// Math.Pow(2, 1d / 2);
            const int problemSeed = 123456;
            var watch = new Stopwatch();

            var cpuSolver = new Version_1._0.CPU();
            var gpuSolver = new Version_1._0.GPU();
            var version = cpuSolver.GetType().Namespace;
            var csvBuilder = new StringBuilder("problemcount,cputime,gputime,cpugpucombinedtime");
            var resultsDictionary = new Dictionary<int, Dictionary<ComputationType, BenchmarkResult>>();

            // in a loop check the performance of the CPU
            double dn = initialProblemSamplingCount;
            for (
                int n = (int)dn;
                ;
                n = (int)Math.Round(dn *= sizeIncrease))
            {
                for (
                    int degreeOfParallelism = 1;
                    degreeOfParallelism <= Environment.ProcessorCount;
                    degreeOfParallelism += Environment.ProcessorCount - 1)
                {

                    var problems = Problem.GetArrayOfProblems(n, problemSize, problemSeed);

                    watch.Start();
                    var results = cpuSolver.Compute(problems, degreeOfParallelism);
                    watch.Stop();

                    var computationElapsed = watch.Elapsed;

                    watch.Reset();
                    watch.Start();
                    if (!(results.Zip(problems, (result, problem) => !result.isSynchronizable || Verify.VerifyValidityOfSynchronizingWord(problem, result, degreeOfParallelism)).All(isOK => isOK)
                        && results.Zip(problems, (result, problem) => Verify.VerifyCernyConjecture(problem, result)).All(isOK => isOK)))
                    {
                        throw new Exception("Incorrect algorithm");
                    }
                    watch.Stop();

                    var verificationElapsed = watch.Elapsed;

                    if (!resultsDictionary.ContainsKey(n))
                    {
                        resultsDictionary.Add(n, new Dictionary<ComputationType, BenchmarkResult>());
                    }
                    resultsDictionary[n][ComputationType.CPU_Parallel] = new BenchmarkResult
                    {
                        problemsPerSecond = (int)Math.Round(n / computationElapsed.TotalSeconds)
                    };

                    Console.WriteLine($"{n} problems computed using {degreeOfParallelism} processors in {computationElapsed.TotalMilliseconds}ms. Problems per second: {n / computationElapsed.TotalSeconds}. Time per problem {computationElapsed.TotalMilliseconds / n}ms");
                    Console.WriteLine($"{n} problems verified using {degreeOfParallelism} processors in {verificationElapsed.TotalMilliseconds}ms. Verifications per second: {n / verificationElapsed.TotalSeconds}. Time per verification {verificationElapsed.TotalMilliseconds / n}ms");
                    Console.WriteLine($"Summary: {results.Average(result => result.isSynchronizable ? 1 : 0) * 100}% synchronizability, {results.Where(result => result.isSynchronizable).Average(result => result.shortestSynchronizingWord.Length)} average length of a synchronizing word");

                    var histogram = results
                        .Where(result => result.isSynchronizable)
                        .Histogram(30, result => result.shortestSynchronizingWord.Length);

                    foreach (var bin in histogram)
                    {
                        Console.Write($"{Math.Round(bin.RepresentativeValue)}: ");
                        for (int i = 0; i < bin.Count * 800 / n || (i == 0 && bin.Count > 0); i++)
                        {
                            Console.Write("-");
                        }
                        Console.WriteLine();
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
                Console.WriteLine();
            }

            // in a loop check the performance of the GPU


            // in a loop check the performance of CPU + GPU combined!


            //save appropriate statistical data to a file
        }
    }
}
