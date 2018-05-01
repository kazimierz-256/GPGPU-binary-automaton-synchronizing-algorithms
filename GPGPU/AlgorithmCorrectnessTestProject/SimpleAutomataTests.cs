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
        private IComputation GetLatestCPU() => new SlimCPU();

        [TestMethod]
        public void YoutubeExperimentalMathematicsSize3()
        {
            var problem = ProblemGenerator.generateWorstCase(3);

            var result = GetLatestCPU().ComputeOne(problem);

            Assert.IsNotNull(result);
            Assert.IsTrue(result.isSynchronizable);
            //Assert.IsNotNull(result.shortestSynchronizingWord);
            //Assert.IsTrue(Verify.VerifyValidityOfSynchronizingWord(problem, result, 1));
            Assert.AreEqual(4, result.shortestSynchronizingWordLength);
        }
        [TestMethod]
        public void YoutubeExperimentalMathematicsSizeLarge()
        {
            var n = 13;
            var problem = ProblemGenerator.generateWorstCase(n);

            var result = GetLatestCPU().ComputeOne(problem);

            Assert.IsNotNull(result);
            Assert.IsTrue(result.isSynchronizable);
            //Assert.IsNotNull(result.shortestSynchronizingWord);
            //Assert.IsTrue(Verify.VerifyValidityOfSynchronizingWord(problem, result, 1));
            Assert.AreEqual((n - 1) * (n - 1), result.shortestSynchronizingWordLength);
        }

    }
}
