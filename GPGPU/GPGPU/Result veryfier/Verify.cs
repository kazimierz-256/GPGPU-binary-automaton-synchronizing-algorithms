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
            ComputationResult computationResult,
            int degreeOfParallelism)
        {
            var initialQuery = Enumerable.Range(0, problem.size);
            // for degree of parallelism == 1 the algorithm should be smarter (faster)
            if (degreeOfParallelism > 1)
            {
                initialQuery = initialQuery.AsParallel().WithDegreeOfParallelism(degreeOfParallelism);
            }

            //var partialResult = initialQuery.Select(letter =>
            //{
            //    int resultingLetter = letter;
            //    foreach (var isB in computationalResult.shortestSynchronizingWord)
            //    {
            //        resultingLetter = isB ? problem.stateTransitioningMatrixB[resultingLetter]
            //        : problem.stateTransitioningMatrixA[resultingLetter];
            //    }
            //    return resultingLetter;
            //});
            //var firstElement = partialResult.First();
            //return partialResult.All(result => result == firstElement);

            return initialQuery.Select(letter =>
            {
                int resultingLetter = letter;
                foreach (var isB in computationResult.shortestSynchronizingWord)
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
