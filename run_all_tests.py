#!/usr/bin/env python3
"""
Comprehensive test runner for the modular genetic algorithm system.

This script runs all available tests and provides detailed reporting,
suitable for SLURM job execution and CI/CD pipelines.
"""

import os
import sys
import time
from typing import List, Dict, Any
import traceback

# Add current directory and src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


class TestResult:
    """Container for test results."""
    
    def __init__(self, test_name: str, passed: bool, duration: float, 
                 output: str = "", error: str = ""):
        self.test_name = test_name
        self.passed = passed
        self.duration = duration
        self.output = output
        self.error = error


class TestRunner:
    """Comprehensive test runner for the GA system."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def run_test(self, test_name: str, test_function) -> TestResult:
        """Run a single test and capture results."""
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        start_time = time.time()
        output_lines = []
        error_lines = []
        
        try:
            # Capture stdout
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            class OutputCapture:
                def __init__(self, original, lines_list):
                    self.original = original
                    self.lines = lines_list
                    
                def write(self, text):
                    self.original.write(text)
                    self.lines.append(text)
                    
                def flush(self):
                    self.original.flush()
            
            sys.stdout = OutputCapture(original_stdout, output_lines)
            sys.stderr = OutputCapture(original_stderr, error_lines)
            
            # Run the test
            success = test_function()
            
            # Restore stdout/stderr
            sys.stdout = original_stdout  
            sys.stderr = original_stderr
            
            duration = time.time() - start_time
            output = ''.join(output_lines)
            error = ''.join(error_lines)
            
            result = TestResult(test_name, success, duration, output, error)
            
            if success:
                print(f"PASS {test_name} PASSED ({duration:.2f}s)")
            else:
                print(f"FAIL {test_name} FAILED ({duration:.2f}s)")
                if error:
                    print(f"Error: {error}")
                    
        except Exception as e:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            duration = time.time() - start_time
            error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
            
            result = TestResult(test_name, False, duration, "", error_msg)
            print(f"FAIL {test_name} FAILED with exception ({duration:.2f}s)")
            print(f"Error: {str(e)}")
        
        self.results.append(result)
        return result
    
    def run_basic_modular_tests(self) -> bool:
        """Run the basic modular component tests."""
        try:
            # Import and run basic tests
            from test_basic_modular import (
                test_imports, test_parameter_encoding, test_population_management,
                test_selection_methods, test_genetic_operations, 
                test_duplicate_prevention, test_convergence_detection, test_reporting
            )
            
            tests = [
                ("Module Imports", test_imports),
                ("Parameter Encoding", test_parameter_encoding), 
                ("Population Management", test_population_management),
                ("Selection Methods", test_selection_methods),
                ("Genetic Operations", test_genetic_operations),
                ("Duplicate Prevention", test_duplicate_prevention),
                ("Convergence Detection", test_convergence_detection),
                ("Reporting System", test_reporting)
            ]
            
            all_passed = True
            for test_name, test_func in tests:
                result = self.run_test(f"Modular: {test_name}", test_func)
                if not result.passed:
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            print(f"Failed to run basic modular tests: {e}")
            return False
    
    def run_dependency_check(self) -> bool:
        """Check that all required dependencies are available."""
        print("\nChecking dependencies...")
        
        required_modules = [
            'random', 'csv', 'os', 'concurrent.futures', 'sys', 'time',
            'functools', 'typing', 'math', 'json', 'datetime', 'threading'
        ]
        
        optional_modules = [
            ('psutil', 'Memory monitoring'),
            ('tqdm', 'Progress bars'), 
            ('numpy', 'Numerical operations'),
            ('matplotlib', 'Plotting')
        ]
        
        missing_required = []
        missing_optional = []
        
        # Check required modules
        for module in required_modules:
            try:
                __import__(module)
                print(f"PASS {module}")
            except ImportError:
                print(f"FAIL {module} (REQUIRED)")
                missing_required.append(module)
        
        # Check optional modules
        for module, description in optional_modules:
            try:
                __import__(module)
                print(f"PASS {module} ({description})")
            except ImportError:
                print(f"WARN {module} ({description}) - OPTIONAL")
                missing_optional.append(module)
        
        if missing_required:
            print(f"\nFAIL Missing required dependencies: {missing_required}")
            return False
        
        if missing_optional:
            print(f"\nWARN Missing optional dependencies: {missing_optional}")
            print("   Some features may not work fully.")
        
        print("PASS All required dependencies available")
        return True
    
    def run_cache_tests(self) -> bool:
        """Test the caching system."""
        print("\nTesting cache system...")
        
        try:
            from cache import get_global_cache
            
            cache = get_global_cache()
            
            # Test cache stats (basic functionality test)
            stats = cache.get_stats()
            if len(stats) != 4:  # Should return (hits, misses, hit_rate, size)
                print("FAIL Cache stats not working")
                return False
                
            # Test cache file creation
            cache._ensure_cache_dir()
            if not cache.cache_dir.exists():
                print("FAIL Cache directory not created")
                return False
                
            print("PASS Cache system working correctly")
            return True
            
        except Exception as e:
            print(f"FAIL Cache test failed: {e}")
            return False
    
    def run_compressor_tests(self) -> bool:
        """Test compressor modules."""
        print("\nTesting compressor modules...")
        
        try:
            from Compressors.base_compressor import BaseCompressor
            
            # Test that base compressor can be imported
            print("PASS BaseCompressor imported successfully")
            
            # Try importing specific compressors
            compressor_modules = [
                'Compressors.brotli_compressor',
                'Compressors.lzma_compressor', 
                'Compressors.zstd_compressor'
            ]
            
            available_compressors = []
            for module_name in compressor_modules:
                try:
                    __import__(module_name)
                    compressor_name = module_name.split('.')[-1].replace('_compressor', '').upper()
                    available_compressors.append(compressor_name)
                    print(f"PASS {compressor_name} compressor available")
                except ImportError as e:
                    print(f"WARN {module_name} not available: {e}")
            
            if len(available_compressors) == 0:
                print("FAIL No compressors available")
                return False
            
            print(f"PASS {len(available_compressors)} compressor(s) available: {', '.join(available_compressors)}")
            return True
            
        except Exception as e:
            print(f"FAIL Compressor tests failed: {e}")
            return False
    
    def run_file_structure_check(self) -> bool:
        """Check that all required files and directories exist."""
        print("\nChecking file structure...")
        
        required_files = [
            'main.py',
            'src/genetic_algorithm.py',
            'src/main.py',
            'config/params.json',
            'src/cache.py',
            'src/ga_components/__init__.py',
            'src/ga_components/parameter_encoding.py',
            'src/ga_components/population_management.py',
            'src/ga_components/selection.py',
            'src/ga_components/genetic_operations.py',
            'src/ga_components/evaluation.py',
            'src/ga_components/duplicate_prevention.py',
            'src/ga_components/convergence_detection.py',
            'src/ga_components/reporting.py',
            'src/Compressors/base_compressor.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"PASS {file_path}")
            else:
                print(f"FAIL {file_path}")
                missing_files.append(file_path)
        
        if missing_files:
            print(f"\nFAIL Missing files: {missing_files}")
            return False
        
        print("PASS All required files present")
        return True
    
    def run_configuration_tests(self) -> bool:
        """Test configuration file parsing."""
        print("\nTesting configuration...")
        
        try:
            import json
            
            # Check params.json
            if os.path.exists('config/params.json'):
                with open('config/params.json', 'r') as f:
                    params = json.load(f)
                
                # Check for GA configuration section
                if 'ga_config' in params:
                    ga_config = params['ga_config']
                    required_keys = ['population_size', 'generations', 'mutation_rate', 'crossover_rate']
                    missing_keys = [key for key in required_keys if key not in ga_config]
                else:
                    missing_keys = ['ga_config section not found']
                
                if missing_keys:
                    print(f"FAIL config/params.json missing keys: {missing_keys}")
                    return False
                
                print("PASS config/params.json configuration valid")
            else:
                print("WARN config/params.json not found - using defaults")
            
            return True
            
        except Exception as e:
            print(f"FAIL Configuration test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all available tests."""
        print("COMPREHENSIVE GA SYSTEM TEST SUITE")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        
        # Define all test suites
        test_suites = [
            ("Dependency Check", self.run_dependency_check),
            ("File Structure Check", self.run_file_structure_check),
            ("Configuration Tests", self.run_configuration_tests),
            ("Cache System Tests", self.run_cache_tests),
            ("Compressor Tests", self.run_compressor_tests),
            ("Modular Component Tests", self.run_basic_modular_tests)
        ]
        
        # Run all test suites
        suite_results = {}
        for suite_name, suite_function in test_suites:
            result = self.run_test(suite_name, suite_function)
            suite_results[suite_name] = result.passed
        
        # Generate summary
        total_time = time.time() - self.start_time
        passed_tests = sum(1 for result in self.results if result.passed)
        total_tests = len(self.results)
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_time,
            'suite_results': suite_results
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"Total Tests Run: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        print("\nTest Suite Results:")
        for suite_name, passed in summary['suite_results'].items():
            status = "PASS" if passed else "FAIL"
            print(f"  {status} {suite_name}")
        
        if summary['failed_tests'] > 0:
            print("\nFailed Test Details:")
            for result in self.results:
                if not result.passed:
                    print(f"\nFAIL {result.test_name} ({result.duration:.2f}s)")
                    if result.error:
                        print(f"   Error: {result.error[:200]}...")
        
        print("\n" + "="*80)
        if summary['failed_tests'] == 0:
            print("ALL TESTS PASSED! The GA system is ready for use.")
        else:
            print(f"WARNING: {summary['failed_tests']} test(s) failed. Please review the issues above.")
        print("="*80)


def main():
    """Main entry point."""
    runner = TestRunner()
    
    try:
        summary = runner.run_all_tests()
        runner.print_summary(summary)
        
        # Return appropriate exit code
        return 0 if summary['failed_tests'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"\nTest runner crashed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)