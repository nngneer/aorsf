using System;
using Aorsf.Native;

namespace Aorsf
{
    public class AorsfException : Exception
    {
        public int ErrorCode { get; }

        public AorsfException(int errorCode, string message)
            : base(message)
        {
            ErrorCode = errorCode;
        }

        internal static void ThrowIfError(int errorCode)
        {
            if (errorCode == NativeMethods.AORSF_SUCCESS)
                return;

            string message = NativeMethods.GetLastError();

            throw errorCode switch
            {
                NativeMethods.AORSF_ERROR_NULL_POINTER =>
                    new ArgumentNullException(null, message),
                NativeMethods.AORSF_ERROR_INVALID_ARGUMENT =>
                    new ArgumentException(message),
                NativeMethods.AORSF_ERROR_NOT_FITTED =>
                    new InvalidOperationException(message),
                NativeMethods.AORSF_ERROR_OUT_OF_MEMORY =>
                    new OutOfMemoryException(message),
                _ => new AorsfException(errorCode, message)
            };
        }
    }
}
