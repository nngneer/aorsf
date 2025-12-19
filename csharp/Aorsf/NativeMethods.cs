using System;
using System.Runtime.InteropServices;

namespace Aorsf.Native
{
    internal static class NativeMethods
    {
        private const string LibraryName = "aorsf_c";

        // Error codes
        public const int AORSF_SUCCESS = 0;
        public const int AORSF_ERROR_NULL_POINTER = -1;
        public const int AORSF_ERROR_INVALID_ARGUMENT = -2;
        public const int AORSF_ERROR_NOT_FITTED = -3;
        public const int AORSF_ERROR_COMPUTATION = -4;
        public const int AORSF_ERROR_OUT_OF_MEMORY = -5;
        public const int AORSF_ERROR_IO = -6;
        public const int AORSF_ERROR_FORMAT = -7;
        public const int AORSF_ERROR_UNKNOWN = -99;

        // Serialization format
        public const int AORSF_FORMAT_BINARY = 0;
        public const int AORSF_FORMAT_JSON = 1;

        // Serialization flags
        public const uint AORSF_FLAG_HAS_IMPORTANCE = 0x01;
        public const uint AORSF_FLAG_HAS_OOB = 0x02;
        public const uint AORSF_FLAG_HAS_METADATA = 0x04;

        // Config struct
        [StructLayout(LayoutKind.Sequential)]
        public struct AorsfConfig
        {
            public int TreeType;
            public int NTree;
            public int Mtry;
            public int LeafMinObs;
            public int LeafMinEvents;
            public int SplitMinObs;
            public int SplitMinEvents;
            public int SplitMinStat;
            public int NSplit;
            public int NRetry;
            public int ViType;
            public int SplitRule;
            public int LincombType;
            public double LincombEps;
            public int LincombIterMax;
            public int LincombScale;
            public double LincombAlpha;
            public int LincombDfTarget;
            public int Oobag;
            public int OobagEvalEvery;
            public int NThread;
            public uint Seed;
            public int Verbosity;
        }

        // Lifecycle functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void aorsf_config_init(ref AorsfConfig config, int treeType);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_create(out IntPtr handle, ref AorsfConfig config);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void aorsf_forest_destroy(IntPtr handle);

        // Data functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_data_create(
            out IntPtr handle,
            [In] double[] x,
            int nRows,
            int nCols,
            [In] double[] y,
            int nYCols,
            [In] double[]? weights,
            int nClass
        );

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void aorsf_data_destroy(IntPtr handle);

        // Training functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_fit(IntPtr handle, IntPtr data);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_is_fitted(IntPtr handle);

        // Prediction functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_predict_get_dims(
            IntPtr handle,
            int nRows,
            int predType,
            out int outRows,
            out int outCols
        );

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_predict(
            IntPtr handle,
            [In] double[] xNew,
            int nRows,
            int nCols,
            int predType,
            [Out] double[] predictions,
            int predictionsSize
        );

        // Variable importance
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_get_importance(
            IntPtr handle,
            [Out] double[] importance,
            int importanceSize
        );

        // Model information
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_get_n_features(IntPtr handle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_get_n_tree(IntPtr handle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_get_tree_type(IntPtr handle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_get_n_class(IntPtr handle);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_get_oob_error(IntPtr handle, out double oobError);

        // Survival-specific
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_get_unique_times(
            IntPtr handle,
            [Out] double[]? times,
            ref int nTimes
        );

        // Error handling
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr aorsf_get_last_error();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr aorsf_get_version();

        // Serialization functions
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_get_save_size(
            IntPtr handle,
            int format,
            uint flags,
            out UIntPtr size
        );

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_save(
            IntPtr handle,
            int format,
            uint flags,
            [Out] byte[] buffer,
            UIntPtr bufferSize,
            out UIntPtr written
        );

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_load(
            out IntPtr handle,
            [In] byte[] buffer,
            UIntPtr bufferSize
        );

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_save_file(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string filepath,
            int format,
            uint flags
        );

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int aorsf_forest_load_file(
            out IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string filepath
        );

        // Helper to get error string
        public static string GetLastError()
        {
            IntPtr ptr = aorsf_get_last_error();
            return Marshal.PtrToStringAnsi(ptr) ?? "Unknown error";
        }

        public static string GetVersion()
        {
            IntPtr ptr = aorsf_get_version();
            return Marshal.PtrToStringAnsi(ptr) ?? "Unknown";
        }
    }
}
