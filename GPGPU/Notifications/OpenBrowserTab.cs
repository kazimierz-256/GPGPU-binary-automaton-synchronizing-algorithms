using System;
using System.Collections.Generic;
using System.Text;

namespace Notifications
{
    public class OpenBrowserTab : INotify
    {
        public void Notify(Problem problem)
        {
            System.Diagnostics.Process.Start($@"https://k256.bitbucket.io/cadcam/index.html?a=" + problem.ToString());
        }
    }
}
