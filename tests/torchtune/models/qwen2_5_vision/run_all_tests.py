#!/usr/bin/env python3
"""Main test runner for all Qwen2.5-VL model tests."""

import sys
import os
import importlib.util
from pathlib import Path


def import_and_run_test(test_file_path):
    """Import a test file and run its tests."""
    test_file = Path(test_file_path)
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Import the test module
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    test_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(test_module)
        
        # Run the test if it has a run_all_tests function
        if hasattr(test_module, 'run_all_tests'):
            return test_module.run_all_tests()
        else:
            print(f"⚠️  Test file {test_file.name} doesn't have a run_all_tests function")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run tests from {test_file.name}: {e}")
        return False


def main():
    """Run all Qwen2.5-VL tests."""
    print("=" * 70)
    print("🚀 Running All Qwen2.5-VL Model Tests")
    print("=" * 70)
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # Define test files in order of execution
    test_files = [
        "test_rotary_embeddings.py",  # Start with the most basic component
        "test_transform.py",          # Then test the transform
        "test_vision_encoder.py",     # Then the vision encoder
        "test_full_model.py",         # Finally the full model
    ]
    
    results = []
    total_tests = len(test_files)
    
    for i, test_file in enumerate(test_files, 1):
        test_path = test_dir / test_file
        
        print(f"\n📋 Test {i}/{total_tests}: {test_file}")
        print("=" * 50)
        
        try:
            result = import_and_run_test(test_path)
            results.append(result)
            
            if result:
                print(f"✅ {test_file} completed successfully!")
            else:
                print(f"❌ {test_file} failed!")
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error running {test_file}: {e}")
            results.append(False)
        
        print("-" * 50)
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    failed = total_tests - passed
    
    print(f"Total test files: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    for i, (test_file, result) in enumerate(zip(test_files, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {test_file:<25} {status}")
    
    if passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        exit_code = 0
    else:
        print(f"\n⚠️  {failed} TEST FILE(S) FAILED")
        exit_code = 1
    
    print("=" * 70)
    return exit_code


def run_specific_test(test_name):
    """Run a specific test by name."""
    test_dir = Path(__file__).parent
    test_path = test_dir / f"test_{test_name}.py"
    
    if not test_path.exists():
        test_path = test_dir / f"{test_name}.py"
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_name}")
        print("Available tests:")
        for test_file in test_dir.glob("test_*.py"):
            print(f"  - {test_file.stem}")
        return False
    
    print(f"🚀 Running specific test: {test_path.name}")
    return import_and_run_test(test_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        exit_code = main()
        sys.exit(exit_code) 