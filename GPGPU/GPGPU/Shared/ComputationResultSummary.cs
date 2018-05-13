using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GPGPU.Shared
{
    class ComputationResultSummary
    {
        TimeSpan elapsedComputationTime;
        TimeSpan elapsedMemoryTransferTime;
        TimeSpan elapsedTotalComputationTime;
        int n;
        double fractionOfSynchronizability;
    }
}
