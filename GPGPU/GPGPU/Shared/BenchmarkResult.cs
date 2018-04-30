using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Shared
{
    public class BenchmarkResult
    {
        public int problemsPerSecond;
        public TimeSpan benchmarkedTime;
        public TimeSpan totalTime;
    }
}
