using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Aorsf.Native
{
    internal static class NativeLibraryResolver
    {
        private static bool _initialized;

        internal static void Initialize()
        {
            if (_initialized) return;
            _initialized = true;

            NativeLibrary.SetDllImportResolver(
                typeof(NativeLibraryResolver).Assembly,
                ResolveDllImport);
        }

        private static IntPtr ResolveDllImport(
            string libraryName,
            Assembly assembly,
            DllImportSearchPath? searchPath)
        {
            if (libraryName != "aorsf_c")
                return IntPtr.Zero;

            // Try to load from runtimes folder first
            string rid = GetRuntimeIdentifier();
            string extension = GetLibraryExtension();
            string prefix = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "" : "lib";

            var assemblyDir = Path.GetDirectoryName(assembly.Location) ?? ".";
            var nativeLibPath = Path.Combine(
                assemblyDir, "runtimes", rid, "native", $"{prefix}aorsf_c{extension}");

            if (File.Exists(nativeLibPath) &&
                NativeLibrary.TryLoad(nativeLibPath, out IntPtr handle))
            {
                return handle;
            }

            // Also try in the assembly directory directly
            var directPath = Path.Combine(assemblyDir, $"{prefix}aorsf_c{extension}");
            if (File.Exists(directPath) &&
                NativeLibrary.TryLoad(directPath, out handle))
            {
                return handle;
            }

            // Fall back to system search path
            if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out handle))
            {
                return handle;
            }

            throw new DllNotFoundException(
                $"Unable to load native library 'aorsf_c'. " +
                $"Expected at: {nativeLibPath}. " +
                $"Make sure the native library is built for {rid}.");
        }

        private static string GetRuntimeIdentifier()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return "win-x64";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return "linux-x64";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return RuntimeInformation.ProcessArchitecture == Architecture.Arm64
                    ? "osx-arm64"
                    : "osx-x64";

            throw new PlatformNotSupportedException();
        }

        private static string GetLibraryExtension()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return ".dll";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return ".dylib";
            return ".so";
        }
    }
}
