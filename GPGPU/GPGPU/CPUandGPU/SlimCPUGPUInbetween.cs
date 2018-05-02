using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.CPUandGPU
{
    class SlimCPUGPUInbetween : IComputable
    {
        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int degreeOfParallelism)
            => Compute(problemsToSolve, degreeOfParallelism, .5f);

        public ComputationResult[] Compute(IEnumerable<Problem> problemsToSolve, int degreeOfParallelism, float cpuPart)
        {
            IEnumerable<ComputationResult> cpuResults = null;
            int cpuProblems = (int)Math.Floor(problemsToSolve.Count() * cpuPart);

            var gpu = new SlimGPUQueue();
            var gpuResults = gpu.Compute(
                problemsToSolve.Skip(cpuProblems).Take(problemsToSolve.Count() - cpuProblems),
                gpu.GetBestParallelism(),
                () =>
                {
                    var cpu = new SlimCPU();
                    cpuResults = cpu
                       .Compute(problemsToSolve
                           .Take(cpuProblems),
                           cpu.GetBestParallelism());

                });
            return cpuResults.Concat(gpuResults).ToArray();
        }

        public ComputationResult ComputeOne(Problem problemToSolve) => new SlimCPU().ComputeOne(problemToSolve);
        public int GetBestParallelism() => -1;

    }
}
