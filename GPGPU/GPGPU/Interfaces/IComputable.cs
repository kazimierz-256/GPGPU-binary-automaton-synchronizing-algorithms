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
        /// <returns>Is Cerny Conjecture false? if yes then return the index of the problem that fails, else return -1</returns>
        int Compute(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount, int degreeOfParallelism);

        /// <returns>Is Cerny Conjecture false? if yes then return the index of the problem that fails, else return -1</returns>
        int Verify(Problem[] problemsToSolve, int problemsReadingIndex, int problemCount, int degreeOfParallelism);
        /// <returns>Optimal parallelism parameter</returns>
        int GetBestParallelism();
        
    }
}
