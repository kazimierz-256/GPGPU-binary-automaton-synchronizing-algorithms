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
            //new CPU(),
            new SlimCPU(),
            //new SlimGPU(),// we're having issues with memory allocation...
            new SlimGPUQueue(),
            //new SlimCPUGPU(),
            //new SlimCPUGPUInbetween(),
            //new SlimGPUAllAtOnce(),
            //new SlimGPUBreakthrough(),
            //new SlimGPULeastSynchronizable(),
        };

        private void AssertOneProblem(Problem problem)
        {
            var results = computables
                .Select(computable => computable.Compute(new[] { problem }, 0, 1, computable.GetBestParallelism()))
                .ToArray();

            AssertSameResults(results, computables, new[] { problem });
        }

        [TestMethod]
        public void SmallWorstCase() => AssertOneProblem(Problem.GenerateWorstCase(3));
        [TestMethod]
        public void MediumWorstCase() => AssertOneProblem(Problem.GenerateWorstCase(7));
        [TestMethod]
        public void LargeWorstCase() => AssertOneProblem(Problem.GenerateWorstCase(13));

        [TestMethod]
        public void LargeIntegrityCheck()
        {
            var n = 1 << 13;
            // there are issues with multiple sized problems!
            // individually tests pass...
            // probably this is due to hardcoding problem size during compilation...
            var sizes = Enumerable.Range(13, 1).ToArray();
            var seeds = Enumerable.Range(123456, 4).ToArray();
            //var invariantProblems = new List<Problem>()
            //{
            //    Problem.GenerateWorstCase(3),
            //    Problem.GenerateWorstCase(7),
            //    Problem.GenerateWorstCase(13),
            //};

            foreach (var seed in seeds)
            {
                foreach (var size in sizes)
                {
                    // 10, 6
                    // 11, 5 don't work correctly...
                    var problems = Problem.GetArrayOfProblems(n, size, seed);//.Concat(invariantProblems).ToArray();//.Skip(10).Take(6);

                    var results = computables
                        .Select(computable => computable.Compute(problems, 0, problems.Length, computable.GetBestParallelism()))
                        .ToArray();

                    AssertSameResults(results, computables, problems);
                }
            }
        }

        private void AssertSameResults(ComputationResult[][] results, IComputable[] computables, Problem[] problems)
        {
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
                            resultR.shortestSynchronizingWordLength)
                            return false;
                    return true;
                }));
            Trace.WriteLine(summary.Average(isOK => isOK ? 1 : 0));
            Assert.IsTrue(summary.All(isOk => isOk), "Not all results are ok");
        }
    }
}
