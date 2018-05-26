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
        void Compute(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int resultsWritingIndex, int problemCount, int degreeOfParallelism);
        int GetBestParallelism();

        //void Compute();
        //IComputable<T> SetProblems(Problem[] problemsToSolve, int problemsReadingIndex, ComputationResult[] computationResults, int problemCount);
        //IComputable<T> SetDegreeOfParallelism(int degreeOfParallelism);
        //IComputable<T> SetSpecialArgument(T specialArgument);
    }
}
