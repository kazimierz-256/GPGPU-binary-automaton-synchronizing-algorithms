using GPGPU;
using GPGPU.Interfaces;
using GPGPU.Shared;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmCorrectnessTestProject
{
    [TestClass]
    public class IntegrityTest
    {
        [TestMethod]
        public void CheckIntegrity()
        {
            var n = 1 << 4;
            // there are issues with multiple sized problems!
            // individually tests pass...
            // probably this is due to hardcoding problem size during compilation...
            var sizes = Enumerable.Range(13, 1);
            var seeds = Enumerable.Range(123456, 10);

            var computables = new IComputable[] {
                new CPU(),
                new SlimCPU(),
                //new SlimGPU(),// we're having issues with memory allocation...
                new SlimGPUQueue(),
                new SlimCPUGPU(),
                new SlimCPUGPUInbetween(),
            };
            foreach (var seed in seeds)
            {
                foreach (var size in sizes)
                {
                    var problems = Problem.GetArrayOfProblems(n, size, seed);

                    var results = computables.Select(computable => computable.Compute(problems, computable.GetBestParallelism()));

                    foreach (var result in results.Skip(1))
                    {
                        Assert.AreEqual(results.First().Length, result.Length, "Incomplete results");
                    }

                    results.Skip(1).AsParallel().All(result =>
                    {
                        for (int r = 0; r < n; r++)
                        {
                            if (results.First()[r].isSynchronizable != result[r].isSynchronizable)
                                return false;
                            if (results.First()[r].isSynchronizable)
                                if (results.First()[r].shortestSynchronizingWordLength !=
                                    result[r].shortestSynchronizingWordLength)
                                    return false;
                        }
                        return true;
                    });
                }
            }
        }
    }
}
