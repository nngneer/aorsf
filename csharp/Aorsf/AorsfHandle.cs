using System;
using System.Runtime.InteropServices;
using Aorsf.Native;

namespace Aorsf
{
    internal class ForestHandle : SafeHandle
    {
        public ForestHandle() : base(IntPtr.Zero, true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle()
        {
            if (!IsInvalid)
            {
                NativeMethods.aorsf_forest_destroy(handle);
            }
            return true;
        }
    }

    internal class DataHandle : SafeHandle
    {
        public DataHandle() : base(IntPtr.Zero, true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle()
        {
            if (!IsInvalid)
            {
                NativeMethods.aorsf_data_destroy(handle);
            }
            return true;
        }
    }
}
