using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Shared
{
    public struct Problem
    {
        public int[] stateTransitioningMatrixA;
        public int[] stateTransitioningMatrixB;
        public int size;

        internal static Problem[] GetArrayOfProblems(int problemCount, int problemSize, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, problemCount).Select(
                 _ => Problem.GenerateProblem(problemSize, random.Next())
                 ).ToArray();
        }

        internal static Problem GenerateProblem(int problemSize, int seed)
        {
            var random = new Random(seed);
            var problem = new Problem
            {
                stateTransitioningMatrixA = new int[problemSize],
                stateTransitioningMatrixB = new int[problemSize],
                size = problemSize
            };

            for (int i = 0; i < problemSize; i++)
            {
                problem.stateTransitioningMatrixA[i] = random.Next(0, problemSize);
                problem.stateTransitioningMatrixB[i] = random.Next(0, problemSize);
            }

            return problem;
        }
    }
}
