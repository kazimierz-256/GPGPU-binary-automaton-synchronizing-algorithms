using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GPGPU
{
    public class SlimCPUGPU : IComputable
    {
        private float cpuPart = 0.5f;
        public int Compute(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount, int degreeOfParallelism)
        {
            int cpuProblems = (int)Math.Round(problemsToSolve.Length * cpuPart);
            var CernyConjectureFailingIndex = -1;
            var thread = new Thread(() =>
            {
                var result = new SlimCPU4bits13().Compute(problemsToSolve, 0, computationResults, 0, cpuProblems, Environment.ProcessorCount);
                if (result >= 0)
                    Interlocked.CompareExchange(ref CernyConjectureFailingIndex, result, -1);
            })
            {
                IsBackground = false
            };
            thread.Start();

            IComputable gpu = new SlimGPUQueue();
            var gpuResult = gpu.Compute(problemsToSolve, cpuProblems, computationResults, cpuProblems, problemCount - cpuProblems, gpu.GetBestParallelism());

            if (gpuResult >= 0)
            {
                thread.Abort();
                return gpuResult;
            }
            else
            {
                thread.Join();
                return CernyConjectureFailingIndex;
            }
        }
        public int GetBestParallelism() => -1;
        public void SetCPUPart(float cpuPart) => this.cpuPart = cpuPart;

        public int Verify(Problem[] problemsToSolve, int problemsReadingIndex, int problemCount, int degreeOfParallelism)
            => Compute(problemsToSolve, problemsReadingIndex, null, -1, problemCount, degreeOfParallelism);
    }
}
