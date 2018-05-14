#define benchmark
using GPGPU.Interfaces;
using GPGPU.Shared;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea.CSharp;
using Alea;
using Alea.FSharp;
using System.Diagnostics;

namespace GPGPU
{
    public class SlimGPUBreakthrough : IComputable
    {
        private static readonly GlobalVariableSymbol<int> problemSize = Gpu.DefineConstantVariableSymbol<int>();

        public ComputationResult ComputeOne(Problem problemToSolve)
        => Compute(new[] { problemToSolve }, 1).First();

        public ComputationResult[] Compute(
            IEnumerable<Problem> problemsToSolve,
            int streamCount)
            => Compute(problemsToSolve, streamCount, null);

        public ComputationResult[] Compute(
            IEnumerable<Problem> problemsToSolve,
            int streamCount,
            Action asyncAction = null,
            int warpCount = 16)
        // cannot be more warps since more memory should be allocated
        {
#if (benchmark)
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();
#endif
            var gpu = Gpu.Default;
            var n = problemsToSolve.First().size;

            var power = 1 << n;
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            var maximumWarps = gpu.Device.Attributes.MaxThreadsPerBlock / gpu.Device.Attributes.WarpSize;
            if (warpCount > maximumWarps)
                warpCount = maximumWarps;

            var problemsPerStream = (problemsToSolve.Count() + streamCount - 1) / streamCount;
            var problemsPartitioned = Enumerable.Range(0, streamCount)
                .Select(i => problemsToSolve.Skip(problemsPerStream * i)
                    .Take(problemsPerStream)
                    .ToArray())
                .Where(partition => partition.Length > 0)
                .ToArray();
            streamCount = problemsPartitioned.Length;
            var streams = Enumerable.Range(0, streamCount)
                .Select(_ => gpu.CreateStream()).ToArray();

            var gpuA = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var gpuB = problemsPartitioned.Select(problems => gpu.Allocate<int>(problems.Length * n)).ToArray();
            var isSynchronizable = problemsPartitioned.Select(problems => gpu.Allocate<bool>(problems.Length)).ToArray();
            gpu.Copy(n, problemSize);

            var launchParameters = new LaunchParam(
                new dim3(1, 1, 1),
                new dim3(gpu.Device.Attributes.WarpSize * warpCount, 1, 1),
                2 * n * sizeof(ushort)
            );
            var gpuResultsIsSynchronizable = problemsPartitioned
                .Select(problems => new bool[problems.Length])
                .ToArray();


            for (int stream = 0; stream < streamCount; stream++)
            {
                var problems = problemsPartitioned[stream];

                var matrixA = new int[problems.Length * n];
                var matrixB = new int[problems.Length * n];
                Parallel.For(0, problems.Length, problem =>
                {
                    Array.ConstrainedCopy(problems[problem].stateTransitioningMatrixA, 0, matrixA, problem * n, n);
                    Array.ConstrainedCopy(problems[problem].stateTransitioningMatrixB, 0, matrixB, problem * n, n);
                });

                streams[stream].Copy(matrixA, gpuA[stream]);
                streams[stream].Copy(matrixB, gpuB[stream]);

                // TODO: change this entirely
                // warning, this might not compute the length of a shortest synchronizing word but it will verify the Cerny conjecture
                streams[stream].Launch(
                    Kernel,
                    launchParameters,
                    gpuA[stream],
                    gpuB[stream],
                    isSynchronizable[stream]
                    );
            }

            asyncAction?.Invoke();

            for (int stream = 0; stream < streamCount; stream++)
            {
#if (benchmark)
                benchmarkTiming.Start();
#endif
                streams[stream].Synchronize();
#if (benchmark)
                benchmarkTiming.Stop();
#endif
                streams[stream].Copy(isSynchronizable[stream], gpuResultsIsSynchronizable[stream]);
            }

            gpu.Synchronize();

#if (benchmark)
#endif
            //Enumerable.Range(0, streamCount)
            //    .SelectMany(stream => gpuResultsIsSynchronizable[stream])
            //    .Select((result) =>
            //    {

            //    });
            var results = new ComputationResult[problemsToSolve.Count()];


            var slimCPU = new SlimCPU();
            var listOfCPUProblems = new List<Problem>();
            var cpuProblemIndex = new List<int>();
            int generalIndex = 0;
            for (int stream = 0; stream < streamCount; stream++)
            {
                for (int index = 0; index < gpuResultsIsSynchronizable[stream].Length; index++)
                {
                    if (gpuResultsIsSynchronizable[stream][index])
                    {
                        results[generalIndex] = new ComputationResult
                        {
                            isSynchronizable = true,
                            computationType = ComputationType.CPU_GPU_Combined,
                            size = n,
                            algorithmName = GetType().Name
                        };
                    }
                    else
                    {
                        listOfCPUProblems.Add(problemsPartitioned[stream][index]);
                        cpuProblemIndex.Add(generalIndex);
                    }
                    generalIndex++;
                }
            }

            var cpuResults = slimCPU.Compute(listOfCPUProblems, slimCPU.GetBestParallelism());

            for (int i = 0; i < listOfCPUProblems.Count; i++)
            {
                results[cpuProblemIndex[i]] = cpuResults[i];
            }

            if (cpuResults.Any(result => result.isSynchronizable && result.shortestSynchronizingWordLength > maximumPermissibleWordLength))
                throw new Exception("Cerny conjecture is false");

            foreach (var array in gpuA.AsEnumerable<Array>()
                .Concat(gpuB)
                .Concat(isSynchronizable))
                Gpu.Free(array);

            foreach (var stream in streams)
                stream.Dispose();

#if (benchmark)
            results[0].benchmarkResult = new BenchmarkResult
            {
                benchmarkedTime = benchmarkTiming.Elapsed,
                totalTime = totalTiming.Elapsed
            };
#endif
            return results;
        }

        public static void Kernel(
            int[] precomputedStateTransitioningMatrixA,
            int[] precomputedStateTransitioningMatrixB,
            bool[] statusOfSynchronization)
        {
            // the status might be YES, NO and DUNNO (aleaGPU enum???)
            // TODO: change this Kernel and computation!
            var n = problemSize.Value;
            var arrayCount = precomputedStateTransitioningMatrixA.Length / n;
            var power = 1 << n;

            #region Pointer setup
            var byteOffset = 0;

            var gpuA = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Volatile();
            byteOffset += n * sizeof(ushort);

            var gpuB = DeviceFunction.AddressOfArray(__shared__.ExternArray<ushort>())
                .Ptr(byteOffset / sizeof(ushort))
                .Volatile();
            byteOffset += n * sizeof(ushort);
            #endregion

            var acPart = (arrayCount + gridDim.x - 1) / gridDim.x;
            var acBegin = blockIdx.x * acPart;
            var acEnd = acBegin + acPart;
            if (arrayCount < acEnd)
                acEnd = arrayCount;
            var index = acBegin * n;
            for (int ac = acBegin; ac < acEnd; ac++, index += n)
            {
                DeviceFunction.SyncThreads();
                if (threadIdx.x == 0)
                    for (int i = 0; i < n; i++)
                    {
                        gpuA[i] = (ushort)(1 << precomputedStateTransitioningMatrixA[index + i]);
                        gpuB[i] = (ushort)(1 << precomputedStateTransitioningMatrixB[index + i]);
                    }
                var pathMask = threadIdx.x;
                int vertexAfterTransition;
                var consideringVertex = power - 1;
                DeviceFunction.SyncThreads();
                for (int iter = 0; iter < 9; iter++, pathMask >>= 1)
                {
                    vertexAfterTransition = 0;
                    if ((pathMask & 1) == 0)
                    {
                        for (int i = 0, mask = 1; i < n; i++, mask <<= 1)
                        {
                            if (0 != (mask & consideringVertex))
                            {
                                vertexAfterTransition |= gpuA[i];
                            }
                        }
                    }
                    else
                    {
                        for (int i = 0, mask = 1; i < n; i++, mask <<= 1)
                        {
                            if (0 != (mask & consideringVertex))
                            {
                                vertexAfterTransition |= gpuB[i];
                            }
                        }
                    }
                    consideringVertex = vertexAfterTransition;
                }
                var singleVertex = DeviceFunction.Any(0 == (consideringVertex & (consideringVertex - 1)));
                if (singleVertex && threadIdx.x % DeviceFunction.WarpSize == 0)
                {
                    statusOfSynchronization[ac] = true;
                }
            }
        }
        public int GetBestParallelism() => 4;
    }
}
