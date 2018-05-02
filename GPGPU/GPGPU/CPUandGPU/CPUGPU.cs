using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GPGPU.CPUandGPU
{
    class SlimCPUGPU : IComputable
    {
        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int degreeOfParallelism)
        {
            IEnumerable<ComputationResult> cpuResults = null;
            var thread = new Thread(() =>
            {
                var cpu = new SlimCPU();
                cpuResults = cpu
                   .Compute(problemsToSolve
                       .Skip(problemsToSolve.Count() / 2)
                       .Take((problemsToSolve.Count() + 1) / 2),
                       cpu.GetBestParallelism());
            })
            {
                IsBackground = false
            };
            thread.Start();

            var gpu = new SlimGPU();
            var gpuResults = gpu.Compute(problemsToSolve.Take(problemsToSolve.Count() / 2), gpu.GetBestParallelism());

            thread.Join();
            return gpuResults.Concat(cpuResults).ToArray();
        }
        public ComputationResult ComputeOne(Problem problemToSolve) => new SlimCPU().ComputeOne(problemToSolve);
        public int GetBestParallelism() => -1;
    }
}
