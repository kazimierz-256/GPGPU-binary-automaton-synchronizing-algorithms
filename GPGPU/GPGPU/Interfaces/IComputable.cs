using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Interfaces
{
    public interface IComputable
    {
        ComputationResult[] Compute(Problem[] problemsToSolve, int beginningIndex, int problemCount, int degreeOfParallelism);
        int GetBestParallelism();
    }
}
