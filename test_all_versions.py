#!/usr/bin/env python3
"""
MCML Multi-Version Test Runner

This script demonstrates testing all MCML versions with the unified test suite.
"""

import subprocess
import sys
import os

def run_version_test(version):
    """Test a specific MCML version"""
    print(f"\n{'='*60}")
    print(f"Testing MCML {version}")
    print(f"{'='*60}")
    
    if not os.path.exists(f"{version}/bin/mcml.exe"):
        print(f"[SKIP] MCML {version} not found - skipping")
        return False
        
    try:
        result = subprocess.run([
            sys.executable, "test_mcml.py", version
        ], capture_output=True, text=True, timeout=120)
        
        if "Success Rate: 100.0%" in result.stdout:
            print(f"[PASS] MCML {version}: ALL TESTS PASSED")
            return True
        else:
            print(f"[FAIL] MCML {version}: SOME TESTS FAILED")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] MCML {version}: TESTS TIMED OUT")
        return False
    except Exception as e:
        print(f"[ERROR] MCML {version}: ERROR - {e}")
        return False

def main():
    """Test all available MCML versions"""
    print("MCML Multi-Version Compatibility Test")
    print("Testing all available versions with the unified test suite...")
    
    versions = ["2.0.0", "2.1.0", "3.0.0"]
    results = {}
    
    for version in versions:
        results[version] = run_version_test(version)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    passed = 0
    for version, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"MCML {version}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(versions)} versions passed all tests")
    
    if passed == len(versions):
        print("\nSUCCESS: All MCML versions are fully compatible!")
        return 0
    else:
        print(f"\nWARNING: {len(versions)-passed} version(s) have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
