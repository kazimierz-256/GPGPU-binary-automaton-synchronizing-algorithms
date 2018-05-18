using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GPGPU
{
    public class SlimCPUGPU : IComputable
    {

        public void Compute(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount, int degreeOfParallelism)
            => ComputePart(problemsToSolve, problemsReadingIndex, computationResults, resultsWritingIndex, problemCount, degreeOfParallelism);

        public void ComputePart(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount, int degreeOfParallelism, float cpuPart = .7f)
        {
            int cpuProblems = (int)Math.Floor(problemsToSolve.Length * cpuPart);
            var thread = new Thread(() =>
            {
                new SlimCPU().Compute(problemsToSolve, 0, computationResults, 0, cpuProblems, Environment.ProcessorCount - 1);
            })
            {
                IsBackground = false
            };
            thread.Start();

            var gpu = new SlimGPUQueue();
            gpu.Compute(problemsToSolve, cpuProblems, computationResults, cpuProblems, problemCount - cpuProblems, gpu.GetBestParallelism());

            thread.Join();
        }
        public int GetBestParallelism() => -1;
    }
}
