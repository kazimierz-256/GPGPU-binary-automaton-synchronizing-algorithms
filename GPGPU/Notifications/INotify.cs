using System;
using System.Collections.Generic;
using System.Text;

namespace Notifications
{
    interface INotify
    {
        void Notify(Problem problem);
    }
}
