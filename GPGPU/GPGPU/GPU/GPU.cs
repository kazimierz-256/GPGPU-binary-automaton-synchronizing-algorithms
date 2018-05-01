using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea.CSharp;



namespace GPGPU.Version_1._0
{
    class GPU : IComputation
    {
        public ComputationResult[] Compute(Problem[] problemsToSolve, int degreeOfParallelism) =>
            Enumerable.Range(0, problemsToSolve.Length)
            .Select(i => ComputeOne(problemsToSolve[i]))
            .ToArray();

        public ComputationResult ComputeOne(Problem problemToSolve)
        {
            var gpu = Alea.Gpu.Default;

            // transform this into streams

            // isDiscovered, distances, wasUsedB
            var launchParameters = new Alea.LaunchParam(1, 1, 4 * 8);

            var gpuA = gpu.Allocate(problemToSolve.stateTransitioningMatrixA);
            var gpuB = gpu.Allocate(problemToSolve.stateTransitioningMatrixB);
            var isCernyConjectureDisproved = gpu.Allocate<bool>(1);
            var isSynchronizing = gpu.Allocate<bool>(1);

            gpu.Launch(
                Kernel,
                launchParameters,
                gpuA,
                gpuB,
                isSynchronizing,
                isCernyConjectureDisproved
                );

            //resultingLeftRight = Gpu.CopyToHost(gpuLeftRight);
            Alea.Gpu.Free(gpuA);
            Alea.Gpu.Free(gpuB);
            Alea.Gpu.Free(isCernyConjectureDisproved);
            Alea.Gpu.Free(isSynchronizing);

            return new ComputationResult()
            {
                size = problemToSolve.size,
                // TODO issyncable, what is the sync word?
            };
        }
        public static void Kernel(int[] matrixA, int[] matrixB, bool[] isSynchronizing, bool[] isCernyConjectureDisproved)
        {

        }
    }
}
