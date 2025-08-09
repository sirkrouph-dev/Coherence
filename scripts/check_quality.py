#!/usr/bin/env python3
"""
Local quality check script for developers.
Run this before pushing to ensure CI/CD will pass.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def run_command(cmd: List[str], check: bool = False) -> Tuple[int, str]:
    """Run a command and return its exit code and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout + e.stderr
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")


def print_result(name: str, success: bool, details: str = ""):
    """Print the result of a check."""
    status = f"{GREEN}✓ PASSED{RESET}" if success else f"{RED}✗ FAILED{RESET}"
    print(f"{name}: {status}")
    if details and not success:
        print(f"  {YELLOW}{details}{RESET}")


def check_ruff(fix: bool = False) -> bool:
    """Run Ruff linter."""
    print_section("Running Ruff Linter")
    cmd = ["ruff", "check", "."]
    if fix:
        cmd.append("--fix")
    
    returncode, output = run_command(cmd)
    success = returncode == 0
    
    if not success:
        print(output)
    
    print_result("Ruff", success)
    return success


def check_black(fix: bool = False) -> bool:
    """Run Black formatter."""
    print_section("Running Black Formatter")
    cmd = ["black", "core", "api", "tests", "benchmarks", "scripts"]
    if not fix:
        cmd.extend(["--check", "--diff"])
    
    returncode, output = run_command(cmd)
    success = returncode == 0
    
    if not success and not fix:
        print(output[:1000])  # Limit output
        print("Run with --fix to auto-format")
    
    print_result("Black", success)
    return success


def check_isort(fix: bool = False) -> bool:
    """Run isort import sorter."""
    print_section("Running isort Import Sorter")
    cmd = ["isort", "core", "api", "tests", "benchmarks", "scripts"]
    if not fix:
        cmd.extend(["--check-only", "--diff"])
    
    returncode, output = run_command(cmd)
    success = returncode == 0
    
    if not success and not fix:
        print(output[:1000])  # Limit output
        print("Run with --fix to auto-sort imports")
    
    print_result("isort", success)
    return success


def check_flake8() -> bool:
    """Run Flake8 linter."""
    print_section("Running Flake8 Linter")
    cmd = [
        "flake8", "core", "api", "tests",
        "--max-line-length=88",
        "--extend-ignore=E203,W503"
    ]
    
    returncode, output = run_command(cmd)
    success = returncode == 0
    
    if not success:
        print(output[:2000])  # Limit output
    
    print_result("Flake8", success)
    return success


def check_mypy() -> bool:
    """Run mypy type checker."""
    print_section("Running MyPy Type Checker")
    cmd = ["mypy", "core", "api", "--ignore-missing-imports"]
    
    returncode, output = run_command(cmd)
    success = returncode == 0
    
    if not success:
        print(output[:2000])  # Limit output
    
    print_result("MyPy", success)
    return success


def run_tests(coverage: bool = True) -> bool:
    """Run pytest tests."""
    print_section("Running Tests")
    cmd = ["pytest", "tests/", "-v"]
    
    if coverage:
        cmd.extend([
            "--cov=core",
            "--cov=api",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])
    
    returncode, output = run_command(cmd)
    success = returncode == 0
    
    if not success:
        # Print only failed test output
        lines = output.split("\n")
        for i, line in enumerate(lines):
            if "FAILED" in line or "ERROR" in line:
                print("\n".join(lines[max(0, i-5):min(len(lines), i+10)]))
                break
    
    print_result("Tests", success)
    
    if coverage and success:
        print(f"  {GREEN}Coverage report generated in htmlcov/index.html{RESET}")
    
    return success


def check_all(args: argparse.Namespace) -> int:
    """Run all quality checks."""
    print(f"{BLUE}Running Quality Checks...{RESET}")
    
    results = []
    
    # Linting checks
    if not args.no_lint:
        results.append(("Ruff", check_ruff(fix=args.fix)))
        results.append(("Black", check_black(fix=args.fix)))
        results.append(("isort", check_isort(fix=args.fix)))
        results.append(("Flake8", check_flake8()))
        
        if not args.no_type:
            results.append(("MyPy", check_mypy()))
    
    # Tests
    if not args.no_test:
        results.append(("Tests", run_tests(coverage=not args.no_coverage)))
    
    # Summary
    print_section("Summary")
    all_passed = True
    for name, passed in results:
        print_result(name, passed)
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n{GREEN}✓ All quality checks passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Some quality checks failed. Please fix before pushing.{RESET}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run quality checks for the neuromorphic system project"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues where possible (formatting, imports)"
    )
    parser.add_argument(
        "--no-lint",
        action="store_true",
        help="Skip linting checks"
    )
    parser.add_argument(
        "--no-type",
        action="store_true",
        help="Skip type checking"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip tests"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage report"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print(f"{RED}Error: pyproject.toml not found. Run from project root.{RESET}")
        sys.exit(1)
    
    # Install required tools if missing
    required_tools = ["ruff", "black", "isort", "flake8", "mypy", "pytest"]
    missing_tools = []
    
    for tool in required_tools:
        returncode, _ = run_command([tool, "--version"])
        if returncode != 0:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"{YELLOW}Missing tools: {', '.join(missing_tools)}{RESET}")
        print(f"Install with: pip install -e '.[dev]'")
        sys.exit(1)
    
    # Run checks
    exit_code = check_all(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
