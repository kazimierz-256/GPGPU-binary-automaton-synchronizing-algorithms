using System;
using GPGPU;
using GPGPU.Interfaces;
using GPGPU.Result_veryfier;
using GPGPU.Shared;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AlgorithmCorrectnessTestProject
{
    [TestClass]
    public class SimpleAutomataTests
    {
        private IComputable GetLatestComputingUnit() => new SlimCPU();

        [TestMethod]
        public void SmallWorstCase()
        {
            var problem = Problem.GenerateWorstCase(3);

            var result = GetLatestComputingUnit().Compute(new[] { problem }, 0, 1, GetLatestComputingUnit().GetBestParallelism())[0];

            Assert.IsNotNull(result);
            Assert.IsTrue(result.isSynchronizable);
            //Assert.IsNotNull(result.shortestSynchronizingWord);
            //Assert.IsTrue(Verify.VerifyValidityOfSynchronizingWord(problem, result, 1));
            Assert.AreEqual(4, result.shortestSynchronizingWordLength);
        }
        [TestMethod]
        public void LargeWorstCase()
        {
            var n = 13;
            var problem = Problem.GenerateWorstCase(n);

            var result = GetLatestComputingUnit().Compute(new[] { problem }, 0, 1, GetLatestComputingUnit().GetBestParallelism())[0];

            Assert.IsNotNull(result);
            Assert.IsTrue(result.isSynchronizable);
            //Assert.IsNotNull(result.shortestSynchronizingWord);
            //Assert.IsTrue(Verify.VerifyValidityOfSynchronizingWord(problem, result, 1));
            Assert.AreEqual((n - 1) * (n - 1), result.shortestSynchronizingWordLength);
        }

    }
}
