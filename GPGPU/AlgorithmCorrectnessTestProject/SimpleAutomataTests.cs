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
        private void CheckWorstCase(int n)
        {
            var problem = Problem.GenerateWorstCase(n);

            var result = new ComputationResult[1];
            GetLatestComputingUnit().Compute(new[] { problem }, 0, result, 0, 1, GetLatestComputingUnit().GetBestParallelism());

            Assert.IsNotNull(result[0]);
            Assert.IsTrue(result[0].isSynchronizable);
            //Assert.IsNotNull(result.shortestSynchronizingWord);
            //Assert.IsTrue(Verify.VerifyValidityOfSynchronizingWord(problem, result, 1));
            Assert.AreEqual((n - 1) * (n - 1), result[0].shortestSynchronizingWordLength);
        }

        [TestMethod]
        public void WorstCasesFrom3To13()
        {
            for (int i = 3; i < 14; i++)
                CheckWorstCase(i);
        }
        [TestMethod]
        public void WorstCase13() => CheckWorstCase(13);
    }
}
