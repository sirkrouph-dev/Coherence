#!/usr/bin/env python3
"""
Analyze docstring audit results and provide prioritized recommendations.

This script reads the audit report and provides:
- Priority rankings for files needing attention
- Summary statistics by module
- Recommendations for documentation improvements
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


def load_audit_report(report_path: str) -> List[Dict]:
    """Load the JSON audit report containing docstring issues.
    
    Args:
        report_path: Path to the JSON file containing audit results.
    
    Returns:
        List of dictionaries containing issue details.
    
    Raises:
        FileNotFoundError: If the report file doesn't exist.
        json.JSONDecodeError: If the report file contains invalid JSON.
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_by_module(issues: List[Dict]) -> Dict[str, Dict]:
    """Group and analyze documentation issues by module or directory.
    
    Aggregates issue statistics per module and calculates priority scores
    based on issue severity (missing docstrings are highest priority).
    
    Args:
        issues: List of issue dictionaries from the audit report.
    
    Returns:
        Dictionary mapping module names to their statistics including:
        - total_issues: Total number of documentation issues
        - missing: Count of missing docstrings
        - outdated_params: Count of outdated parameter documentation
        - missing_return: Count of missing return documentation
        - file_count: Number of files affected in the module
        - priority_score: Calculated priority score (higher = more urgent)
    
    Example:
        >>> issues = load_audit_report('audit.json')
        >>> stats = analyze_by_module(issues)
        >>> print(stats['core']['total_issues'])
    """
    module_stats = defaultdict(lambda: {
        'total_issues': 0,
        'missing': 0,
        'outdated_params': 0,
        'missing_return': 0,
        'files': set(),
        'priority_score': 0
    })
    
    for issue in issues:
        file_path = Path(issue['file_path'])
        # Get module name (first directory after project root)
        parts = file_path.parts
        
        # Find module name
        module = 'root'
        for i, part in enumerate(parts):
            if part == 'neuron' and i < len(parts) - 1:
                if parts[i + 1] != '__pycache__':
                    module = parts[i + 1] if parts[i + 1].endswith('.py') else parts[i + 1]
                break
        
        stats = module_stats[module]
        stats['total_issues'] += 1
        stats['files'].add(str(file_path))
        stats[issue['issue_type']] += 1
        
        # Calculate priority score
        # Missing docstrings are highest priority
        if issue['issue_type'] == 'missing':
            stats['priority_score'] += 10
        elif issue['issue_type'] == 'outdated_params':
            stats['priority_score'] += 5
        elif issue['issue_type'] == 'missing_return':
            stats['priority_score'] += 2
    
    # Convert sets to counts
    for module in module_stats:
        module_stats[module]['file_count'] = len(module_stats[module]['files'])
        del module_stats[module]['files']
    
    return dict(module_stats)


def get_critical_files(issues: List[Dict], top_n: int = 10) -> List[Tuple[str, int, Dict]]:
    """Identify files with the most critical documentation issues.
    
    Analyzes all issues and ranks files by a priority score that weights
    different issue types: missing docstrings (10 points), outdated params
    (5 points), and missing returns (2 points).
    
    Args:
        issues: List of issue dictionaries from the audit report.
        top_n: Number of top critical files to return (default: 10).
    
    Returns:
        List of tuples containing:
        - File path (str)
        - Priority score (int)
        - Statistics dictionary with counts and affected symbols
    
    Example:
        >>> critical = get_critical_files(issues, top_n=5)
        >>> for file_path, score, stats in critical:
        ...     print(f"{file_path}: priority={score}")
    """
    file_issues = defaultdict(lambda: {
        'count': 0,
        'missing': 0,
        'outdated_params': 0,
        'missing_return': 0,
        'priority_score': 0,
        'symbols': []
    })
    
    for issue in issues:
        file_path = issue['file_path']
        file_stat = file_issues[file_path]
        file_stat['count'] += 1
        file_stat[issue['issue_type']] += 1
        file_stat['symbols'].append({
            'name': issue['symbol_name'],
            'type': issue['symbol_type'],
            'issue': issue['issue_type'],
            'line': issue['line_number']
        })
        
        # Priority scoring
        if issue['issue_type'] == 'missing':
            file_stat['priority_score'] += 10
        elif issue['issue_type'] == 'outdated_params':
            file_stat['priority_score'] += 5
        elif issue['issue_type'] == 'missing_return':
            file_stat['priority_score'] += 2
    
    # Sort by priority score
    sorted_files = sorted(
        [(path, stats['priority_score'], stats) for path, stats in file_issues.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_files[:top_n]


def generate_summary_report(issues: List[Dict]) -> str:
    """Generate a comprehensive summary report of documentation issues.
    
    Creates a human-readable report with overall statistics, module analysis,
    critical files listing, and actionable recommendations for improving
    documentation.
    
    Args:
        issues: List of issue dictionaries from the audit report.
    
    Returns:
        Formatted string containing the complete summary report.
    
    Example:
        >>> issues = load_audit_report('audit.json')
        >>> report = generate_summary_report(issues)
        >>> print(report)
    """
    report = []
    report.append("=" * 80)
    report.append("DOCSTRING AUDIT SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    total_issues = len(issues)
    issue_types = defaultdict(int)
    symbol_types = defaultdict(int)
    
    for issue in issues:
        issue_types[issue['issue_type']] += 1
        symbol_types[issue['symbol_type']] += 1
    
    report.append("OVERALL STATISTICS")
    report.append("-" * 40)
    report.append(f"Total documentation issues: {total_issues}")
    report.append("")
    
    report.append("By Issue Type:")
    for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_issues) * 100
        report.append(f"  - {issue_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    report.append("")
    
    report.append("By Symbol Type:")
    for symbol_type, count in sorted(symbol_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_issues) * 100
        report.append(f"  - {symbol_type.title()}: {count} ({percentage:.1f}%)")
    report.append("")
    
    # Module analysis
    module_stats = analyze_by_module(issues)
    report.append("MODULE ANALYSIS")
    report.append("-" * 40)
    report.append("Modules ranked by priority (files with most critical issues):")
    report.append("")
    
    sorted_modules = sorted(
        module_stats.items(),
        key=lambda x: x[1]['priority_score'],
        reverse=True
    )
    
    for module, stats in sorted_modules[:10]:
        report.append(f"{module}:")
        report.append(f"  Files affected: {stats['file_count']}")
        report.append(f"  Total issues: {stats['total_issues']}")
        report.append(f"  - Missing docstrings: {stats['missing']}")
        report.append(f"  - Outdated parameters: {stats['outdated_params']}")
        report.append(f"  - Missing return docs: {stats['missing_return']}")
        report.append(f"  Priority score: {stats['priority_score']}")
        report.append("")
    
    # Critical files
    report.append("TOP 10 CRITICAL FILES")
    report.append("-" * 40)
    report.append("Files requiring immediate attention:")
    report.append("")
    
    critical_files = get_critical_files(issues, top_n=10)
    for i, (file_path, score, stats) in enumerate(critical_files, 1):
        # Shorten file path for readability
        short_path = str(Path(file_path).relative_to(Path.cwd()))
        report.append(f"{i}. {short_path}")
        report.append(f"   Priority score: {score}")
        report.append(f"   Total issues: {stats['count']}")
        report.append(f"   - Missing: {stats['missing']}")
        report.append(f"   - Outdated params: {stats['outdated_params']}")
        report.append(f"   - Missing returns: {stats['missing_return']}")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("Based on the analysis, here are the recommended actions:")
    report.append("")
    
    if issue_types['missing'] > 0:
        report.append("1. HIGH PRIORITY - Add missing docstrings:")
        report.append(f"   {issue_types['missing']} public symbols lack any documentation.")
        report.append("   Focus on public APIs and frequently used classes/functions first.")
        report.append("")
    
    if issue_types['outdated_params'] > 0:
        report.append("2. MEDIUM PRIORITY - Update parameter documentation:")
        report.append(f"   {issue_types['outdated_params']} functions have mismatched parameters.")
        report.append("   This can cause confusion and errors when using the API.")
        report.append("")
    
    if issue_types['missing_return'] > 0:
        report.append("3. LOW PRIORITY - Document return values:")
        report.append(f"   {issue_types['missing_return']} functions return values without documentation.")
        report.append("   Add Returns sections to clarify what functions return.")
        report.append("")
    
    # Module-specific recommendations
    report.append("4. MODULE-SPECIFIC ACTIONS:")
    for module, stats in sorted_modules[:3]:
        if stats['total_issues'] > 10:
            report.append(f"   - {module}: Critical need for documentation review")
            report.append(f"     ({stats['total_issues']} issues across {stats['file_count']} files)")
    report.append("")
    
    report.append("5. SUGGESTED WORKFLOW:")
    report.append("   a. Start with the top 10 critical files listed above")
    report.append("   b. Focus on public API methods and classes first")
    report.append("   c. Use automated tools to generate docstring templates")
    report.append("   d. Review and update existing docstrings for accuracy")
    report.append("   e. Implement docstring validation in CI/CD pipeline")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def export_prioritized_list(issues: List[Dict], output_path: str):
    """Export a prioritized list of symbols needing documentation.
    
    Categorizes issues into priority levels (critical, high, medium, low)
    based on issue type and context (e.g., public APIs are critical).
    Exports the categorized list as JSON for processing tools.
    
    Args:
        issues: List of issue dictionaries from the audit report.
        output_path: Path where the prioritized JSON file will be saved.
    
    Returns:
        None
    
    Raises:
        IOError: If unable to write to the output path.
    
    Example:
        >>> export_prioritized_list(issues, 'priorities.json')
        Prioritized issue list exported to: priorities.json
    """
    # Group by priority
    priority_groups = {
        'critical': [],  # Missing docstrings on public APIs
        'high': [],      # Outdated parameters
        'medium': [],    # Missing return documentation
        'low': []        # Other issues
    }
    
    for issue in issues:
        # Determine priority based on issue type and symbol type
        if issue['issue_type'] == 'missing':
            if issue['symbol_type'] == 'class' or 'api' in issue['file_path'].lower():
                priority_groups['critical'].append(issue)
            else:
                priority_groups['high'].append(issue)
        elif issue['issue_type'] == 'outdated_params':
            priority_groups['high'].append(issue)
        elif issue['issue_type'] == 'missing_return':
            priority_groups['medium'].append(issue)
        else:
            priority_groups['low'].append(issue)
    
    # Create prioritized JSON output
    prioritized = {
        'summary': {
            'total_issues': len(issues),
            'critical': len(priority_groups['critical']),
            'high': len(priority_groups['high']),
            'medium': len(priority_groups['medium']),
            'low': len(priority_groups['low'])
        },
        'issues_by_priority': priority_groups
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prioritized, f, indent=2)
    
    print(f"Prioritized issue list exported to: {output_path}")


def main():
    """Main function to analyze docstring audit results.
    
    Parses command-line arguments, loads the audit report, generates
    analysis reports, and exports prioritized issue lists.
    
    Returns:
        Exit code (0 for success, 1 for error).
    
    Example:
        Command line usage:
        $ python analyze_audit_results.py audit.json -o summary.txt
    """
    parser = argparse.ArgumentParser(
        description='Analyze docstring audit results and provide recommendations'
    )
    parser.add_argument(
        'report',
        nargs='?',
        default='docstring_audit_report.json',
        help='Path to the JSON audit report (default: docstring_audit_report.json)'
    )
    parser.add_argument(
        '-o', '--output',
        default='docstring_audit_summary.txt',
        help='Output file for summary report (default: docstring_audit_summary.txt)'
    )
    parser.add_argument(
        '-p', '--prioritized',
        default='docstring_priorities.json',
        help='Output file for prioritized issue list (default: docstring_priorities.json)'
    )
    
    args = parser.parse_args()
    
    # Load the audit report
    try:
        issues = load_audit_report(args.report)
    except FileNotFoundError:
        print(f"Error: Audit report '{args.report}' not found.")
        print("Please run the audit_docstrings.py script first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{args.report}'")
        sys.exit(1)
    
    if not issues:
        print("No issues found in the audit report!")
        sys.exit(0)
    
    # Generate summary report
    summary = generate_summary_report(issues)
    
    # Write summary to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Also print to console
    print(summary)
    
    # Export prioritized list
    export_prioritized_list(issues, args.prioritized)
    
    print(f"\nSummary report written to: {args.output}")


if __name__ == '__main__':
    main()
