using System.Runtime.CompilerServices;

namespace Aorsf
{
    internal static class ModuleInit
    {
        [ModuleInitializer]
        internal static void Initialize()
        {
            Native.NativeLibraryResolver.Initialize();
        }
    }
}
