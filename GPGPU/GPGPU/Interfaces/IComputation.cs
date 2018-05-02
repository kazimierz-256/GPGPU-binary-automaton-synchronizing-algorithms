using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Interfaces
{
    public interface IComputation
    {
        ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism);
        ComputationResult ComputeOne(Problem problemToSolve);
        int GetBestParallelism();
    }
}
