using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Result_veryfier
{
    public class Verify
    {
        public static bool VerifyValidityOfSynchronizingWord(
            Problem problem,
            ComputationResult result,
            int degreeOfParallelism)
        {
            var initialQuery = Enumerable.Range(0, problem.size);
            // for degree of parallelism == 1 the algorithm should be smarter (faster)
            if (degreeOfParallelism > 1)
            {
                initialQuery = Enumerable.Range(0, problem.size).AsParallel().WithDegreeOfParallelism(degreeOfParallelism);
            }
            // add check for equality, not the quality of being distinct
            return initialQuery.Select(letter =>
            {
                int resultingLetter = letter;
                foreach (var isB in result.shortestSynchronizingWord)
                {
                    resultingLetter = isB ? problem.stateTransitioningMatrixB[resultingLetter]
                    : problem.stateTransitioningMatrixA[resultingLetter];
                }
                return resultingLetter;
            }).Distinct().Count() == 1;
        }

        //public static bool VerifySynchronizability() { }
        //public static bool VerifyTotalValidity() { }
        //public static bool VerifyNonsynchronizability() { };

        public static bool VerifyCernyConjecture(Problem problem, ComputationResult result) =>
            !result.isSynchronizable
            || result.shortestSynchronizingWord.Length <= (problem.size - 1) * (problem.size - 1);
    }
}
