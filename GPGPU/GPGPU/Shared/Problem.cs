using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Shared
{
    struct Problem
    {
        public byte[] stateTransitioningMatrixA;
        public byte[] stateTransitioningMatrixB;
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
                stateTransitioningMatrixA = new byte[problemSize],
                stateTransitioningMatrixB = new byte[problemSize],
                size = problemSize
            };

            for (int i = 0; i < problemSize; i++)
            {
                problem.stateTransitioningMatrixA[i] = (byte)random.Next(0, problemSize);
                problem.stateTransitioningMatrixB[i] = (byte)random.Next(0, problemSize);
            }

            return problem;
        }
    }
}
