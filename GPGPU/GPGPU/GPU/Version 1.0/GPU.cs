using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Version_1._0
{
    class GPU : IComputation
    {
        public ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism) => Enumerable.Range(0, problemsToSolve.Length)
                .Select(i => ComputeOne(problemsToSolve[i])).ToArray();

        public ComputationResult ComputeOne(Problem problemToSolve)
        {
            return new ComputationResult();
        }
    }
}
