#!/usr/bin/env python3
"""Test script for Rust CUDA kernels via PyO3"""

try:
    import teenygrad._rs as rs
    print("✓ Successfully imported teenygrad._rs")
    print(f"Available functions: {dir(rs)}")
    print()

    # Test the smoke test
    print("Running CUDA smoke test...")
    rs.cuda_smoke_test()
    print("✓ CUDA smoke test passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
