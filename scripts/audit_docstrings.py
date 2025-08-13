#!/usr/bin/env python3
"""
Comprehensive docstring audit tool for Python codebase.

This script analyzes all Python files in the codebase to identify:
- Missing docstrings on public classes, functions, and methods
- Outdated parameter descriptions (parameters in signature but not in docstring)
- Missing return type descriptions
- Incorrect parameter names in docstrings
"""

import ast
import os
import sys
import json
import inspect
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse


@dataclass
class DocstringIssue:
    """Represents a documentation issue found in the code."""
    
    file_path: str
    line_number: int
    symbol_name: str
    symbol_type: str  # 'class', 'function', 'method'
    issue_type: str  # 'missing', 'outdated_params', 'missing_return', 'incorrect_params'
    details: str
    current_params: List[str] = None
    documented_params: List[str] = None
    missing_params: List[str] = None
    extra_params: List[str] = None
    has_return: bool = False
    needs_return_doc: bool = False


class DocstringParser:
    """Parse docstrings to extract parameter and return information."""
    
    @staticmethod
    def parse_google_style(docstring: str) -> Tuple[List[str], bool]:
        """Parse Google-style docstrings."""
        if not docstring:
            return [], False
            
        params = []
        has_return = False
        
        # Find Args/Arguments section
        args_pattern = r'(?:Args?|Arguments?|Parameters?):\s*\n((?:\s+\w+.*\n)*)'
        args_match = re.search(args_pattern, docstring, re.IGNORECASE)
        
        if args_match:
            args_section = args_match.group(1)
            # Extract parameter names
            param_pattern = r'^\s+(\w+)(?:\s*\([^)]+\))?:'
            for line in args_section.split('\n'):
                param_match = re.match(param_pattern, line)
                if param_match:
                    params.append(param_match.group(1))
        
        # Check for Returns section
        returns_pattern = r'(?:Returns?|Yields?):\s*\n'
        if re.search(returns_pattern, docstring, re.IGNORECASE):
            has_return = True
            
        return params, has_return
    
    @staticmethod
    def parse_numpy_style(docstring: str) -> Tuple[List[str], bool]:
        """Parse NumPy-style docstrings."""
        if not docstring:
            return [], False
            
        params = []
        has_return = False
        
        # Find Parameters section
        params_pattern = r'Parameters\s*\n\s*-+\s*\n((?:.*\n)*?)(?:\n\s*\n|Returns|Yields|Examples|Notes|$)'
        params_match = re.search(params_pattern, docstring, re.IGNORECASE | re.MULTILINE)
        
        if params_match:
            params_section = params_match.group(1)
            # Extract parameter names
            param_pattern = r'^(\w+)\s*:'
            for line in params_section.split('\n'):
                param_match = re.match(param_pattern, line.strip())
                if param_match:
                    params.append(param_match.group(1))
        
        # Check for Returns section
        returns_pattern = r'(?:Returns?|Yields?)\s*\n\s*-+'
        if re.search(returns_pattern, docstring, re.IGNORECASE):
            has_return = True
            
        return params, has_return
    
    @staticmethod
    def parse_sphinx_style(docstring: str) -> Tuple[List[str], bool]:
        """Parse Sphinx-style docstrings."""
        if not docstring:
            return [], False
            
        params = []
        has_return = False
        
        # Find :param directives
        param_pattern = r':param\s+(\w+):'
        params = re.findall(param_pattern, docstring)
        
        # Check for :return or :returns directive
        if re.search(r':returns?:', docstring, re.IGNORECASE):
            has_return = True
            
        return params, has_return
    
    @classmethod
    def parse_docstring(cls, docstring: str) -> Tuple[List[str], bool]:
        """Parse docstring and extract parameter names and return info."""
        if not docstring:
            return [], False
        
        # Try different docstring styles
        # Google style
        params, has_return = cls.parse_google_style(docstring)
        if params or has_return:
            return params, has_return
        
        # NumPy style
        params, has_return = cls.parse_numpy_style(docstring)
        if params or has_return:
            return params, has_return
        
        # Sphinx style
        params, has_return = cls.parse_sphinx_style(docstring)
        
        return params, has_return


class DocstringAuditor(ast.NodeVisitor):
    """AST visitor to audit docstrings in Python code."""
    
    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_code = source_code
        self.issues: List[DocstringIssue] = []
        self.current_class = None
        self.parser = DocstringParser()
        
    def is_public(self, name: str) -> bool:
        """Check if a symbol is public (doesn't start with underscore)."""
        return not name.startswith('_')
    
    def has_return_statement(self, node: ast.FunctionDef) -> bool:
        """Check if function has a return statement with value."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
            if isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                return True
        return False
    
    def get_function_params(self, node: ast.FunctionDef) -> List[str]:
        """Extract parameter names from function definition."""
        params = []
        for arg in node.args.args:
            # Skip 'self' and 'cls' parameters
            if arg.arg not in ['self', 'cls']:
                params.append(arg.arg)
        
        # Add *args if present
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        
        # Add keyword-only arguments
        for arg in node.args.kwonlyargs:
            params.append(arg.arg)
        
        # Add **kwargs if present
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")
        
        return params
    
    def check_function_docstring(self, node: ast.FunctionDef, symbol_type: str):
        """Check docstring for a function or method."""
        if not self.is_public(node.name):
            return
        
        docstring = ast.get_docstring(node)
        symbol_name = node.name
        if self.current_class and symbol_type == 'method':
            symbol_name = f"{self.current_class}.{node.name}"
        
        # Check for missing docstring
        if not docstring:
            self.issues.append(DocstringIssue(
                file_path=self.file_path,
                line_number=node.lineno,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                issue_type='missing',
                details='No docstring found'
            ))
            return
        
        # Get actual parameters and documented parameters
        actual_params = self.get_function_params(node)
        documented_params, has_return_doc = self.parser.parse_docstring(docstring)
        
        # Check for parameter mismatches
        actual_set = set(actual_params)
        documented_set = set(documented_params)
        
        # Remove special parameters from comparison
        actual_set_clean = {p.strip('*') for p in actual_set}
        documented_set_clean = {p.strip('*') for p in documented_set}
        
        missing_params = list(actual_set_clean - documented_set_clean)
        extra_params = list(documented_set_clean - actual_set_clean)
        
        if missing_params or extra_params:
            details = []
            if missing_params:
                details.append(f"Missing documentation for: {', '.join(missing_params)}")
            if extra_params:
                details.append(f"Documented but not in signature: {', '.join(extra_params)}")
            
            self.issues.append(DocstringIssue(
                file_path=self.file_path,
                line_number=node.lineno,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                issue_type='outdated_params',
                details='; '.join(details),
                current_params=actual_params,
                documented_params=documented_params,
                missing_params=missing_params,
                extra_params=extra_params
            ))
        
        # Check for missing return documentation
        if self.has_return_statement(node) and not has_return_doc:
            # Skip __init__ methods as they don't need return documentation
            if node.name != '__init__':
                self.issues.append(DocstringIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    issue_type='missing_return',
                    details='Function returns a value but no Returns section in docstring',
                    has_return=True,
                    needs_return_doc=True
                ))
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        if self.is_public(node.name):
            docstring = ast.get_docstring(node)
            if not docstring:
                self.issues.append(DocstringIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    symbol_name=node.name,
                    symbol_type='class',
                    issue_type='missing',
                    details='No docstring found'
                ))
        
        # Process methods within the class
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        if self.current_class:
            self.check_function_docstring(node, 'method')
        else:
            self.check_function_docstring(node, 'function')
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        if self.current_class:
            self.check_function_docstring(node, 'method')
        else:
            self.check_function_docstring(node, 'function')
        self.generic_visit(node)


def audit_file(file_path: Path) -> List[DocstringIssue]:
    """Audit a single Python file for docstring issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        tree = ast.parse(source_code, filename=str(file_path))
        auditor = DocstringAuditor(str(file_path), source_code)
        auditor.visit(tree)
        return auditor.issues
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return []


def find_python_files(root_dir: Path, exclude_dirs: Set[str] = None) -> List[Path]:
    """Find all Python files in the directory tree."""
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', '.tox', 'build', 'dist'}
    
    python_files = []
    for path in root_dir.rglob('*.py'):
        # Skip if any parent directory is in exclude list
        if any(part in exclude_dirs for part in path.parts):
            continue
        python_files.append(path)
    
    return python_files


def generate_report(issues: List[DocstringIssue], output_format: str = 'json') -> str:
    """Generate a report of all docstring issues."""
    if output_format == 'json':
        return json.dumps([asdict(issue) for issue in issues], indent=2)
    
    elif output_format == 'markdown':
        report = ["# Docstring Audit Report", ""]
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total issues found: {len(issues)}")
        report.append("")
        
        # Group issues by file
        issues_by_file: Dict[str, List[DocstringIssue]] = {}
        for issue in issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        # Statistics
        report.append("## Summary Statistics")
        report.append("")
        
        issue_counts = {'missing': 0, 'outdated_params': 0, 'missing_return': 0}
        symbol_counts = {'class': 0, 'function': 0, 'method': 0}
        
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
            symbol_counts[issue.symbol_type] = symbol_counts.get(issue.symbol_type, 0) + 1
        
        report.append("### By Issue Type")
        for issue_type, count in issue_counts.items():
            report.append(f"- {issue_type.replace('_', ' ').title()}: {count}")
        report.append("")
        
        report.append("### By Symbol Type")
        for symbol_type, count in symbol_counts.items():
            report.append(f"- {symbol_type.title()}: {count}")
        report.append("")
        
        # Detailed issues by file
        report.append("## Detailed Issues by File")
        report.append("")
        
        for file_path in sorted(issues_by_file.keys()):
            file_issues = issues_by_file[file_path]
            report.append(f"### {file_path}")
            report.append(f"Issues: {len(file_issues)}")
            report.append("")
            
            for issue in sorted(file_issues, key=lambda x: x.line_number):
                report.append(f"- **Line {issue.line_number}**: `{issue.symbol_name}` ({issue.symbol_type})")
                report.append(f"  - Issue: {issue.issue_type.replace('_', ' ').title()}")
                report.append(f"  - Details: {issue.details}")
                
                if issue.missing_params:
                    report.append(f"  - Missing params: {', '.join(issue.missing_params)}")
                if issue.extra_params:
                    report.append(f"  - Extra params: {', '.join(issue.extra_params)}")
                
                report.append("")
        
        return "\n".join(report)
    
    elif output_format == 'csv':
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['File', 'Line', 'Symbol', 'Type', 'Issue', 'Details'])
        
        for issue in sorted(issues, key=lambda x: (x.file_path, x.line_number)):
            writer.writerow([
                issue.file_path,
                issue.line_number,
                issue.symbol_name,
                issue.symbol_type,
                issue.issue_type,
                issue.details
            ])
        
        return output.getvalue()
    
    else:  # text format
        report = []
        report.append("=" * 80)
        report.append("DOCSTRING AUDIT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total issues: {len(issues)}")
        report.append("")
        
        for issue in sorted(issues, key=lambda x: (x.file_path, x.line_number)):
            report.append(f"File: {issue.file_path}")
            report.append(f"Line: {issue.line_number}")
            report.append(f"Symbol: {issue.symbol_name} ({issue.symbol_type})")
            report.append(f"Issue: {issue.issue_type}")
            report.append(f"Details: {issue.details}")
            report.append("-" * 40)
        
        return "\n".join(report)


def main():
    """Main function to run the docstring audit."""
    parser = argparse.ArgumentParser(
        description='Audit Python codebase for missing or outdated docstrings'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Path to Python file or directory to audit (default: current directory)'
    )
    parser.add_argument(
        '-o', '--output',
        default='docstring_audit_report.json',
        help='Output file path (default: docstring_audit_report.json)'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'markdown', 'csv', 'text'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        help='Additional directories to exclude from scanning'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output during processing'
    )
    
    args = parser.parse_args()
    
    # Convert path to Path object
    target_path = Path(args.path).resolve()
    
    # Determine if it's a file or directory
    if target_path.is_file():
        if target_path.suffix == '.py':
            python_files = [target_path]
        else:
            print(f"Error: {target_path} is not a Python file", file=sys.stderr)
            sys.exit(1)
    elif target_path.is_dir():
        exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', '.tox', 'build', 'dist'}
        exclude_dirs.update(args.exclude)
        python_files = find_python_files(target_path, exclude_dirs)
    else:
        print(f"Error: {target_path} not found", file=sys.stderr)
        sys.exit(1)
    
    if not python_files:
        print("No Python files found to audit", file=sys.stderr)
        sys.exit(1)
    
    print(f"Auditing {len(python_files)} Python files...")
    
    # Audit all files
    all_issues = []
    for file_path in python_files:
        if args.verbose:
            print(f"Processing: {file_path}")
        issues = audit_file(file_path)
        all_issues.extend(issues)
    
    # Generate report
    report = generate_report(all_issues, args.format)
    
    # Determine output extension based on format
    output_path = Path(args.output)
    if output_path.suffix == '':
        extensions = {'json': '.json', 'markdown': '.md', 'csv': '.csv', 'text': '.txt'}
        output_path = output_path.with_suffix(extensions[args.format])
    
    # Write report to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Print summary
    print(f"\nAudit complete!")
    print(f"Total issues found: {len(all_issues)}")
    print(f"Report written to: {output_path}")
    
    # Print issue breakdown
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
    
    if issue_counts:
        print("\nIssue breakdown:")
        for issue_type, count in sorted(issue_counts.items()):
            print(f"  - {issue_type.replace('_', ' ').title()}: {count}")
    
    # Exit with error code if issues found
    sys.exit(1 if all_issues else 0)


if __name__ == '__main__':
    main()
