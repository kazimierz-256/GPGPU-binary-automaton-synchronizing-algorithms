using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Interfaces
{
    interface IComputation
    {
        ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism);
    }
}
