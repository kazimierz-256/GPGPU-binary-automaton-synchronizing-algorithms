using System;
using GPGPU;
using GPGPU.Interfaces;
using GPGPU.Problem_generator;
using GPGPU.Result_veryfier;
using GPGPU.Shared;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AlgorithmCorrectnessTestProject
{
    [TestClass]
    public class SimpleAutomataTests
    {
        private IComputation GetLatestCPU() => new GPGPU.Version_1._0.CPU();

        [TestMethod]
        public void YoutubeExperimentalMathematicsSize3()
        {
            var problem = ProblemGenerator.generateWorstCase(3);

            var result = GetLatestCPU().ComputeOne(problem);

            Assert.IsNotNull(result);
            Assert.IsTrue(result.isSynchronizable);
            Assert.IsNotNull(result.shortestSynchronizingWord);
            Assert.IsTrue(Verify.VerifyValidityOfSynchronizingWord(problem, result, 1));
            Assert.AreEqual(result.shortestSynchronizingWord.Length, 4);
        }
        [TestMethod]
        public void YoutubeExperimentalMathematicsSizeLarge()
        {
            var problem = ProblemGenerator.generateWorstCase(13);

            var result = GetLatestCPU().ComputeOne(problem);

            Assert.IsNotNull(result);
            Assert.IsTrue(result.isSynchronizable);
            Assert.IsNotNull(result.shortestSynchronizingWord);
            Assert.IsTrue(Verify.VerifyValidityOfSynchronizingWord(problem, result, 1));
            Assert.AreEqual(result.shortestSynchronizingWord.Length, 12 * 12);
        }

    }
}
