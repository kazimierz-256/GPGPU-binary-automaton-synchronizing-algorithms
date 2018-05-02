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

        public static Problem GenerateWorstCase(int n)
        {
            if (n < 3)
                throw new Exception("wrong number!");

            var problem = new Problem
            {
                size = n,
                stateTransitioningMatrixA = Enumerable.Range(0, n)
                .Select(i => i == 0 ? 1 : i)
                .ToArray(),
                stateTransitioningMatrixB = Enumerable.Range(0, n)
                .Select(i => (i + 1) == n ? 0 : (i + 1))
                .ToArray()
            };
            return problem;
        }

        public static Problem[] GetArrayOfProblems(int problemCount, int problemSize, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, problemCount).Select(
                 _ => Problem.GenerateProblem(problemSize, random.Next())
                 ).ToArray();
        }

        public static Problem GenerateProblem(int problemSize, int seed)
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
