#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/native-build"
OUTPUT_DIR="$SCRIPT_DIR/Aorsf/runtimes"

mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Build for current platform
build_native() {
    local rid=$1
    local output_name=$2

    echo "Building for $rid..."
    mkdir -p "$BUILD_DIR/$rid"
    cd "$BUILD_DIR/$rid"

    cmake "$ROOT_DIR/src/c_api" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED=ON

    cmake --build . --config Release

    mkdir -p "$OUTPUT_DIR/$rid/native"
    cp *aorsf_c* "$OUTPUT_DIR/$rid/native/" 2>/dev/null || true

    # Also handle the specific filenames
    if [ -f "libaorsf_c.dylib" ]; then
        cp libaorsf_c.dylib "$OUTPUT_DIR/$rid/native/$output_name"
    fi
    if [ -f "libaorsf_c.so" ]; then
        cp libaorsf_c.so "$OUTPUT_DIR/$rid/native/$output_name"
    fi
    if [ -f "aorsf_c.dll" ]; then
        cp aorsf_c.dll "$OUTPUT_DIR/$rid/native/$output_name"
    fi
}

# Detect platform and build
case "$(uname -s)" in
    Linux*)
        build_native "linux-x64" "libaorsf_c.so"
        ;;
    Darwin*)
        if [[ "$(uname -m)" == "arm64" ]]; then
            build_native "osx-arm64" "libaorsf_c.dylib"
        else
            build_native "osx-x64" "libaorsf_c.dylib"
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        build_native "win-x64" "aorsf_c.dll"
        ;;
esac

echo ""
echo "Native libraries built successfully!"
echo "Output directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*/native/ 2>/dev/null || echo "(No libraries found)"
