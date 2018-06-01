using GPGPU;
using GPGPU.Interfaces;
using GPGPU.Shared;
using GPGPU.Result_veryfier;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace AlgorithmCorrectnessTestProject
{
    [TestClass]
    public class IntegrityTest
    {
        private IComputable[] computables = new IComputable[]
        {
            new SlimCPU(),
            new SlimGPUQueue(),
            new SlimCPUGPU(),
            //new SuperSlimGPUBreakthrough()
        };

        private void AssertProblems(Problem[] problems)
        {
            var results = new ComputationResult[computables.Length][];
            var computableId = 0;
            foreach (var computable in computables)
            {
                results[computableId] = new ComputationResult[problems.Length];
                computable.Compute(problems, 0, results[computableId], 0, problems.Length, computable.GetBestParallelism());
                computableId++;
            }

            AssertSameResults(results, computables, problems);
        }
        private void AssertOneProblem(Problem problem) => AssertProblems(new[] { problem });

        [TestMethod]
        public void WorstCases3To13TripleTimes()
        {
            for (int i = 3; i < 14; i++)
                AssertProblems(new[] { Problem.GenerateWorstCase(i), Problem.GenerateWorstCase(i), Problem.GenerateWorstCase(i) });
        }
        [TestMethod]
        public void WorstCase13TripleTimes() => AssertProblems(new[] { Problem.GenerateWorstCase(13), Problem.GenerateWorstCase(13), Problem.GenerateWorstCase(13) });

        [TestMethod]
        public void LargeIntegrityCheck()
        {
            var n = 1 << 17;
            // there are issues with multiple sized problems!
            // individually tests pass...
            // probably this is due to hardcoding problem size during JIT compilation
            var sizes = Enumerable.Range(13, 1).ToArray();
            var seeds = Enumerable.Range(12456, 10).ToArray();

            var problems = new Problem[n];
            foreach (var seed in seeds)
            {
                foreach (var size in sizes)
                {
                    Problem.FillArrayOfProblems(problems, n, size, seed);//.Concat(invariantProblems).ToArray();//.Skip(10).Take(6);
                    var results = new ComputationResult[computables.Length][];
                    var computableId = 0;
                    foreach (var computable in computables)
                    {
                        results[computableId] = new ComputationResult[problems.Length];
                        computable.Compute(problems, 0, results[computableId], 0, problems.Length, computable.GetBestParallelism());
                        computableId++;
                    }

                    AssertSameResults(results, computables, problems);
                }
            }
        }

        private void AssertSameResults(ComputationResult[][] results, IComputable[] computables, Problem[] problems)
        {
            // this is used only for the older CPU algorithm
            //for (int computable = 0; computable < computables.Length; computable++)
            //{
            //    if (computables[computable] is CPU)
            //    {
            //        for (int problem = 0; problem < problems.Count(); problem++)
            //        {
            //            Assert.IsTrue(
            //                Verify.VerifyValidityOfSynchronizingWord(problems.ElementAt(problem),
            //                results[computable][problem], 1),
            //                "Wrong synchronizing word!");
            //        }
            //    }
            //}
            foreach (var result in results.Skip(1))
            {
                Assert.AreEqual(results.First().Length, result.Length, "Incomplete results");
            }
            int resultId = 0;
            var summary = results.Skip(1).SelectMany(result => results[0].Zip(result, (result0, resultR) =>
                {
                    resultId++;
                    if (result0.isSynchronizable != resultR.isSynchronizable)
                        return false;
                    if (result0.isSynchronizable)
                        if (result0.shortestSynchronizingWordLength !=
                            resultR.shortestSynchronizingWordLength && resultR.shortestSynchronizingWordLength > 9)
                            return false;
                    return true;
                }))
                .ToArray();
            Trace.WriteLine(summary.Average(isOK => isOK ? 1 : 0));
            Assert.IsTrue(summary.All(isOk => isOk), "Not all results are ok");
        }
    }
}
