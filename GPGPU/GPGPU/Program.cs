using GPGPU.Shared;
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
            const int initialProblemSamplingCount = 10;
            const int maximumProblemSamplingCount = 100;
            double sizeIncrease = Math.Pow(2, 1d / 3);
            int degreeOfParallelism = Environment.ProcessorCount;
            const int problemSeed = 123456;
            var watch = new Stopwatch();

            var cpuSolver = new Version_1._0.CPU();
            var gpuSolver = new Version_1._0.GPU();
            var version = cpuSolver.GetType().Namespace;
            var csvBuilder = new StringBuilder("problemcount,cputime,gputime,cpugpucombinedtime");
            var resultsDictionary = new Dictionary<int, Dictionary<ComputationType, BenchmarkResult>>();

            // in a loop check the performance of the CPU
            for (int n = initialProblemSamplingCount; n < maximumProblemSamplingCount; n = (int)Math.Round(n * sizeIncrease))
            {
                var problems = Problem.GetArrayOfProblems(n, problemSize, problemSeed);

                watch.Start();
                cpuSolver.Compute(problems, degreeOfParallelism);
                watch.Stop();

                resultsDictionary[initialProblemSamplingCount][ComputationType.CPU_Parallel] = new BenchmarkResult
                {
                    problemsPerSecond = (int)Math.Round(n / watch.Elapsed.TotalSeconds)
                };

                Console.WriteLine($"{n} problems computed using {degreeOfParallelism} in {watch.Elapsed.TotalMilliseconds}ms. Problems per second: {n / watch.Elapsed.TotalSeconds}");
                Console.WriteLine();
            }

            // in a loop check the performance of the GPU


            // in a loop check the performance of CPU + GPU combined!


            //save appropriate statistical data to a file
        }
    }
}
