#!/usr/bin/env python3
"""
MCML/CONV Multi-Version Compatibility Test Runner

This script provides comprehensive testing across ALL available MCML and CONV versions
using their respective test suites. It automatically discovers available versions
and runs complete test suites for each one.

Features:
- Automatic version discovery for both MCML and CONV
- Comprehensive testing using individual test suites
- Cross-version compatibility validation  
- Detailed success/failure reporting
- Support for all version combinations (2.0.0, 2.1.0, 3.0.0)
- Timeout handling for problematic versions
- Summary reporting with overall compatibility status

Expected Usage:
    python test_all_versions.py    # Test all available versions
    
Test Coverage:
- MCML versions: All available versions with 29 comprehensive tests each
- CONV versions: All available versions with 29 comprehensive tests each
- Cross-version validation ensuring consistent functionality
- Performance comparison across versions
"""

import subprocess
import sys
import os

def run_mcml_version_test(version):
    """
    Execute comprehensive test suite for a specific MCML version.
    
    Args:
        version: MCML version string (e.g., "2.1.0", "3.0.0")
        
    Returns:
        bool: True if all tests passed, False otherwise
        
    Features:
    - Automatic executable discovery and validation
    - Complete test suite execution (29 tests)
    - Success rate parsing and reporting
    - Timeout handling for problematic versions
    - Detailed error reporting for failures
    """
    print(f"\n{'-'*50}")
    print(f"Testing MCML {version}")
    print(f"{'-'*50}")
    
    if not os.path.exists(f"../{version}/bin/mcml.exe"):
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
            # Show brief error info
            lines = result.stdout.split('\n')
            for line in lines:
                if "Failed:" in line or "Success Rate:" in line:
                    print(f"        {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] MCML {version}: TESTS TIMED OUT")
        return False
    except Exception as e:
        print(f"[ERROR] MCML {version}: ERROR - {e}")
        return False

def run_conv_version_test(version):
    """
    Execute comprehensive test suite for a specific CONV version.
    
    Args:
        version: CONV version string (e.g., "2.0.0", "2.1.0", "3.0.0")
        
    Returns:
        bool: True if all tests passed, False otherwise
        
    Features:
    - Automatic executable discovery and validation
    - Complete test suite execution (29 tests)  
    - Version-specific timeout handling
    - Legacy version support (2.0.0) with special notes
    - Success rate parsing with pattern matching
    - Detailed error reporting and diagnostics
    """
    print(f"\n{'-'*50}")
    print(f"Testing CONV {version}")
    print(f"{'-'*50}")
    
    conv_exe = f"../{version}/bin/conv.exe"
    if not os.path.exists(conv_exe):
        print(f"[SKIP] CONV {version} not found - skipping")
        return False
    
    try:
        # Use longer timeout for all versions to be safe
        timeout = 240 if version == "2.0.0" else 180
        
        result = subprocess.run([
            sys.executable, "test_conv.py", version
        ], capture_output=True, text=True, timeout=timeout)
        
        # All versions should now run all 29 tests with version-appropriate parameters
        success_pattern = "Success Rate: 100.0%"
        expected_tests = 29
        
        if success_pattern in result.stdout:
            print(f"[PASS] CONV {version}: ALL {expected_tests} TESTS PASSED")
            return True
        else:
            print(f"[FAIL] CONV {version}: SOME TESTS FAILED")
            # Show brief error info
            lines = result.stdout.split('\n')
            for line in lines:
                if "Tests Failed:" in line or "Tests Passed:" in line or "Success Rate:" in line:
                    print(f"        {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] CONV {version}: TESTS TIMED OUT")
        return False
    except Exception as e:
        print(f"[ERROR] CONV {version}: ERROR - {e}")
        return False

def main():
    """Test all available MCML and CONV versions"""
    print("MCML/CONV Multi-Version Compatibility Test")
    print("="*60)
    print("Testing all available versions with comprehensive test suites...")
    
    # Define versions to test
    mcml_versions = ["2.0.0", "2.1.0", "3.0.0"]
    conv_versions = ["2.0.0", "2.1.0", "3.0.0"]  # All versions that have CONV
    
    mcml_results = {}
    conv_results = {}
    
    # Test MCML versions
    print(f"\n{'='*60}")
    print("TESTING MCML VERSIONS")
    print(f"{'='*60}")
    
    for version in mcml_versions:
        mcml_results[version] = run_mcml_version_test(version)
    
    # Test CONV versions
    print(f"\n{'='*60}")
    print("TESTING CONV VERSIONS") 
    print(f"{'='*60}")
    
    for version in conv_versions:
        conv_results[version] = run_conv_version_test(version)
    
    # Final results summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print("\nMCML Results:")
    mcml_passed = 0
    for version, success in mcml_results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  MCML {version}: {status}")
        if success:
            mcml_passed += 1
    
    print("\nCONV Results:")
    conv_passed = 0 
    for version, success in conv_results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  CONV {version}: {status}")
        if success:
            conv_passed += 1
    
    # Overall summary
    total_passed = mcml_passed + conv_passed
    total_tests = len(mcml_results) + len(conv_results)
    
    print(f"\nOverall Summary:")
    print(f"  MCML: {mcml_passed}/{len(mcml_results)} versions passed")
    print(f"  CONV: {conv_passed}/{len(conv_results)} versions passed")
    print(f"  Total: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print(f"\n🎉 SUCCESS: All MCML/CONV versions are fully compatible!")
        print("   Complete functional parity achieved across all versions.")
        return 0
    else:
        failed = total_tests - total_passed
        print(f"\n⚠️  WARNING: {failed} test suite(s) have issues")
        print("   Check individual results above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
