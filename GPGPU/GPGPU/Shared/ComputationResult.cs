using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Shared
{
    public struct ComputationResult
    {
        public bool isSynchronizable;
        public bool[] shortestSynchronizingWord;
        public BenchmarkResult benchmarkResult;
        public int queueBreadth;
        public ComputationType computationType;
        public int size;
    }
}
