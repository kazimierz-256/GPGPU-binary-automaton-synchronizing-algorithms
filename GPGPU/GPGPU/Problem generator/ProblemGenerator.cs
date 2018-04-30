using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Problem_generator
{
    public class ProblemGenerator
    {
        public static Problem generateWorstCase(int n)
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
    }
}
