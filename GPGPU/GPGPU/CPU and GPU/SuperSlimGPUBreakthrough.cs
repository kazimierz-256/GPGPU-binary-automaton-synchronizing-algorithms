//#define benchmark
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
    public class SuperSlimGPUBreakthrough : IComputable
    {
        // the algorithm performs best when using 8 or 16 streams
        public int GetBestParallelism() => 8;

        private static readonly GlobalVariableSymbol<int> problemSize = Gpu.DefineConstantVariableSymbol<int>();

        public int Compute(
            Problem[] problemsToSolve,
            int problemsReadingIndex,
            ComputationResult[] computationResults,
            int resultsWritingIndex,
            int problemCount,
            int streamCount)
            => ComputeAction(problemsToSolve, problemsReadingIndex, computationResults, resultsWritingIndex, problemCount, streamCount);

        public int ComputeAction(
            Problem[] problemsToSolve,
            int problemsReadingIndex,
            ComputationResult[] computationResults,
            int resultsWritingIndex,
            int problemCount,
            int streamCount,
            Action asyncAction = null)
        // cannot be more warps since more memory should be allocated
        {
#if (benchmark)
            var totalTiming = new Stopwatch();
            totalTiming.Start();
            var benchmarkTiming = new Stopwatch();
#endif
            var CernyConjectureFailingIndex = -1;
            var gpu = Gpu.Default;
            var n = problemsToSolve[problemsReadingIndex].size;

            var power = 1 << n;
            var maximumPermissibleWordLength = (n - 1) * (n - 1);

            // in order to atomically add to a checking array (designed for queue consistency) one int is used by four threads, so in a reeeeaally pessimistic case 255 is the maximum number of threads (everyone go to the same vertex)
            // -1 for the already discovered vertex
            // at least 2*n+1 threads i.e. 27
            var threads = gpu.Device.Attributes.MaxThreadsPerBlock;
            var maximumThreads = Math.Min(
                gpu.Device.Attributes.MaxThreadsPerBlock,
               32 * 2
                );
            if (threads > maximumThreads)
                threads = maximumThreads;

            if (problemCount < streamCount)
                streamCount = problemCount;

            var problemsPerStream = (problemCount + streamCount - 1) / streamCount;
            var streams = Enumerable.Range(0, streamCount)
                .Select(_ => gpu.CreateStream()).ToArray();

            gpu.Copy(n, problemSize);

            var launchParameters = new LaunchParam(
                new dim3(1 << 10, 1, 1),
                new dim3(threads, 1, 1),
                2 * n * sizeof(ushort)
            );
            var gpuResultsIsSynchronizable = new bool[streamCount][];
            var gpuAs = new int[streamCount][];
            var gpuBs = new int[streamCount][];
            var isSynchronizable = new bool[streamCount][];

            for (int stream = 0; stream < streamCount; stream++)
            {
                var offset = stream * problemsPerStream;
                var localProblemsCount = Math.Min(problemsPerStream, problemCount - offset);

                gpuAs[stream] = gpu.Allocate<int>(localProblemsCount * n);
                gpuBs[stream] = gpu.Allocate<int>(localProblemsCount * n);
                isSynchronizable[stream] = gpu.Allocate<bool>(localProblemsCount);

                var matrixA = new int[localProblemsCount * n];
                var matrixB = new int[localProblemsCount * n];
                Parallel.For(0, localProblemsCount, problem =>
                {
                    Array.ConstrainedCopy(
                        problemsToSolve[problemsReadingIndex + offset + problem].stateTransitioningMatrixA,
                        0,
                        matrixA,
                        problem * n,
                        n
                        );
                    Array.ConstrainedCopy(
                        problemsToSolve[problemsReadingIndex + offset + problem].stateTransitioningMatrixB,
                        0,
                        matrixB,
                        problem * n,
                        n
                        );
                });

                streams[stream].Copy(matrixA, gpuAs[stream]);
                streams[stream].Copy(matrixB, gpuBs[stream]);

                gpuResultsIsSynchronizable[stream] = new bool[localProblemsCount];

                streams[stream].Launch(
                    Kernel,
                    launchParameters,
                    gpuAs[stream],
                    gpuBs[stream],
                    isSynchronizable[stream]
                    );
            }

            asyncAction?.Invoke();

            var streamId = 0;
            foreach (var stream in streams)
            {
#if (benchmark)
                benchmarkTiming.Start();
#endif
                stream.Synchronize();
#if (benchmark)
                benchmarkTiming.Stop();
#endif
                stream.Copy(isSynchronizable[streamId], gpuResultsIsSynchronizable[streamId]);

                var offset = streamId * problemsPerStream;
                var localProblemsCount = Math.Min(problemsPerStream, problemCount - offset);

                if (computationResults == null)
                {
                    resultsWritingIndex = 0;
                    computationResults = new ComputationResult[resultsWritingIndex + problemCount];
                }

                for (int i = 0; i < localProblemsCount; i++)
                {
                    computationResults[resultsWritingIndex + offset + i].isSynchronizable = gpuResultsIsSynchronizable[streamId][i];
                }

                streamId++;
            }

#if (benchmark)
            gpu.Synchronize();
#endif


            foreach (var arrays in new IEnumerable<Array>[] { gpuAs, gpuBs, isSynchronizable })
                foreach (var array in arrays)
                    Gpu.Free(array);

            foreach (var stream in streams)
                stream.Dispose();

            var cpu = new SlimCPUSkipper();
            var result = cpu.Compute(problemsToSolve, problemsReadingIndex, computationResults, resultsWritingIndex, problemCount, cpu.GetBestParallelism());

#if (benchmark)
            computationResults[resultsWritingIndex].benchmarkResult = new BenchmarkResult
            {
                benchmarkedTime = benchmarkTiming.Elapsed,
                totalTime = totalTiming.Elapsed
            };
#endif
            return result;
        }

        public static void Kernel(
            int[] precomputedStateTransitioningMatrixA,
            int[] precomputedStateTransitioningMatrixB,
            bool[] isSynchronizing)
        {
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
                for (int iter = 0; iter < 6; iter++, pathMask >>= 1)
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
                    isSynchronizing[ac] = true;
                }
            }
        }


        public int Verify(Problem[] problemsToSolve, int problemsReadingIndex, int problemCount, int degreeOfParallelism)
            => Compute(problemsToSolve, problemsReadingIndex, null, -1, problemCount, degreeOfParallelism);
    }
}
