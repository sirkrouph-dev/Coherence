#!/usr/bin/env python
"""
Test runner script with coverage enforcement.
Runs all tests and ensures coverage meets the required threshold.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode


def run_tests(test_type="all", coverage_threshold=90, verbose=False):
    """
    Run tests with coverage enforcement.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        coverage_threshold: Minimum coverage percentage required
        verbose: Whether to run tests in verbose mode
    """
    print(f"\n{'='*60}")
    print(f"Running {test_type} tests with coverage threshold: {coverage_threshold}%")
    print(f"{'='*60}\n")
    
    # Base pytest command
    pytest_cmd = ["pytest"]
    
    # Add test files based on type
    if test_type == "unit":
        pytest_cmd.extend([
            "tests/test_neurons.py",
            "tests/test_synapses.py",
            "tests/test_learning.py"
        ])
    elif test_type == "integration":
        pytest_cmd.extend(["tests/test_integration.py"])
    elif test_type == "all":
        pytest_cmd.extend(["tests/"])
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add coverage options
    pytest_cmd.extend([
        "--cov=core",
        "--cov=api",
        "--cov-report=term-missing",
        "--cov-report=html",
        f"--cov-fail-under={coverage_threshold}"
    ])
    
    # Add verbose flag if requested
    if verbose:
        pytest_cmd.append("-v")
    
    # Run tests
    return_code = run_command(pytest_cmd)
    
    if return_code == 0:
        print(f"\n✅ All tests passed with coverage ≥ {coverage_threshold}%")
    else:
        print(f"\n❌ Tests failed or coverage < {coverage_threshold}%")
    
    return return_code


def run_linting():
    """Run code quality checks."""
    print("\n" + "="*60)
    print("Running code quality checks")
    print("="*60 + "\n")
    
    checks_passed = True
    
    # Run black
    print("\n📝 Checking code formatting with black...")
    if run_command(["black", "--check", "core", "api", "tests"]) != 0:
        print("  ❌ Code formatting issues found. Run 'black core api tests' to fix.")
        checks_passed = False
    else:
        print("  ✅ Code formatting OK")
    
    # Run isort
    print("\n📦 Checking import sorting with isort...")
    if run_command(["isort", "--check-only", "core", "api", "tests"]) != 0:
        print("  ❌ Import sorting issues found. Run 'isort core api tests' to fix.")
        checks_passed = False
    else:
        print("  ✅ Import sorting OK")
    
    # Run flake8
    print("\n🔍 Checking code style with flake8...")
    if run_command([
        "flake8", "core", "api", "tests",
        "--max-line-length=88",
        "--extend-ignore=E203,W503"
    ]) != 0:
        print("  ❌ Code style issues found")
        checks_passed = False
    else:
        print("  ✅ Code style OK")
    
    # Run mypy
    print("\n🔎 Checking type hints with mypy...")
    if run_command(["mypy", "core", "api"]) != 0:
        print("  ❌ Type checking issues found")
        checks_passed = False
    else:
        print("  ✅ Type checking OK")
    
    return 0 if checks_passed else 1


def generate_coverage_report():
    """Generate detailed coverage report."""
    print("\n" + "="*60)
    print("Generating coverage report")
    print("="*60 + "\n")
    
    # Generate HTML report
    run_command(["coverage", "html"])
    
    # Generate XML report for CI
    run_command(["coverage", "xml"])
    
    # Print report to console
    run_command(["coverage", "report"])
    
    html_report = PROJECT_ROOT / "htmlcov" / "index.html"
    if html_report.exists():
        print(f"\n📊 HTML coverage report generated: {html_report}")
        print(f"   Open in browser: file://{html_report.as_posix()}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests with coverage enforcement"
    )
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        type=int,
        default=90,
        help="Minimum coverage percentage required (default: 90)"
    )
    parser.add_argument(
        "--no-lint",
        action="store_true",
        help="Skip linting checks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate coverage report from existing data"
    )
    
    args = parser.parse_args()
    
    if args.report_only:
        return generate_coverage_report()
    
    # Run linting first unless skipped
    if not args.no_lint:
        lint_result = run_linting()
        if lint_result != 0:
            print("\n⚠️  Linting checks failed. Fix issues before running tests.")
            return lint_result
    
    # Run tests
    test_result = run_tests(
        test_type=args.type,
        coverage_threshold=args.coverage,
        verbose=args.verbose
    )
    
    # Generate coverage report if tests passed
    if test_result == 0:
        generate_coverage_report()
    
    return test_result


if __name__ == "__main__":
    sys.exit(main())
