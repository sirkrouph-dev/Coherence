#!/usr/bin/env python3
"""
Interactive browser for docstring audit results.

This script provides an easy way to:
- Filter issues by type, module, or file
- View specific issues with context
- Generate fix templates for common issues
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_report(report_path: str) -> List[Dict]:
    """Load the JSON audit report."""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_issues(issues: List[Dict], 
                  issue_type: Optional[str] = None,
                  module: Optional[str] = None,
                  file_path: Optional[str] = None) -> List[Dict]:
    """Filter issues based on criteria."""
    filtered = issues
    
    if issue_type:
        filtered = [i for i in filtered if i['issue_type'] == issue_type]
    
    if module:
        filtered = [i for i in filtered if module.lower() in i['file_path'].lower()]
    
    if file_path:
        filtered = [i for i in filtered if file_path in i['file_path']]
    
    return filtered


def display_issue(issue: Dict, verbose: bool = False):
    """Display a single issue in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"File: {issue['file_path']}")
    print(f"Line: {issue['line_number']}")
    print(f"Symbol: {issue['symbol_name']} ({issue['symbol_type']})")
    print(f"Issue: {issue['issue_type'].replace('_', ' ').title()}")
    print(f"Details: {issue['details']}")
    
    if verbose:
        if issue.get('current_params'):
            print(f"Current params: {', '.join(issue['current_params'])}")
        if issue.get('documented_params'):
            print(f"Documented params: {', '.join(issue['documented_params'])}")
        if issue.get('missing_params'):
            print(f"Missing params: {', '.join(issue['missing_params'])}")
        if issue.get('extra_params'):
            print(f"Extra params: {', '.join(issue['extra_params'])}")


def generate_docstring_template(issue: Dict) -> str:
    """Generate a docstring template for fixing the issue."""
    template = []
    
    if issue['issue_type'] == 'missing':
        template.append('"""')
        template.append(f'Brief description of {issue["symbol_name"]}.')
        template.append('')
        
        if issue['symbol_type'] in ['function', 'method']:
            if issue.get('current_params'):
                template.append('Args:')
                for param in issue['current_params']:
                    if param not in ['self', 'cls']:
                        clean_param = param.strip('*')
                        template.append(f'    {clean_param}: Description of {clean_param}.')
                template.append('')
            
            template.append('Returns:')
            template.append('    Description of return value.')
        
        template.append('"""')
    
    elif issue['issue_type'] == 'outdated_params':
        template.append('# Update the docstring to include:')
        if issue.get('missing_params'):
            template.append('# Missing parameters:')
            for param in issue['missing_params']:
                template.append(f'#     {param}: Description needed')
        if issue.get('extra_params'):
            template.append('# Remove these parameters (not in signature):')
            for param in issue['extra_params']:
                template.append(f'#     {param}')
    
    elif issue['issue_type'] == 'missing_return':
        template.append('# Add to docstring:')
        template.append('Returns:')
        template.append('    Description of what this function returns.')
    
    return '\n'.join(template)


def interactive_mode(issues: List[Dict]):
    """Interactive mode for browsing issues."""
    print("\n=== DOCSTRING ISSUE BROWSER ===")
    print(f"Total issues: {len(issues)}")
    
    while True:
        print("\nOptions:")
        print("1. Filter by issue type")
        print("2. Filter by module")
        print("3. Show statistics")
        print("4. Show critical files")
        print("5. Generate fix templates")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            print("\nIssue types:")
            print("1. missing")
            print("2. outdated_params")
            print("3. missing_return")
            type_choice = input("Select type (1-3): ").strip()
            
            type_map = {'1': 'missing', '2': 'outdated_params', '3': 'missing_return'}
            if type_choice in type_map:
                filtered = filter_issues(issues, issue_type=type_map[type_choice])
                print(f"\nFound {len(filtered)} issues of type '{type_map[type_choice]}'")
                
                if filtered and input("Show issues? (y/n): ").lower() == 'y':
                    for i, issue in enumerate(filtered[:10], 1):
                        display_issue(issue)
                        if i % 5 == 0 and i < len(filtered):
                            if input("\nContinue? (y/n): ").lower() != 'y':
                                break
        
        elif choice == '2':
            module = input("Enter module name to filter: ").strip()
            filtered = filter_issues(issues, module=module)
            print(f"\nFound {len(filtered)} issues in module '{module}'")
            
            if filtered and input("Show issues? (y/n): ").lower() == 'y':
                for issue in filtered[:10]:
                    display_issue(issue)
        
        elif choice == '3':
            show_statistics(issues)
        
        elif choice == '4':
            show_critical_files(issues)
        
        elif choice == '5':
            issue_num = input("Enter issue number to generate template (or 'back'): ").strip()
            if issue_num.isdigit() and 0 < int(issue_num) <= len(issues):
                issue = issues[int(issue_num) - 1]
                display_issue(issue, verbose=True)
                print("\n--- SUGGESTED FIX TEMPLATE ---")
                print(generate_docstring_template(issue))
        
        elif choice == '6':
            break
        
        else:
            print("Invalid option. Please try again.")


def show_statistics(issues: List[Dict]):
    """Show statistics about the issues."""
    from collections import defaultdict
    
    issue_types = defaultdict(int)
    symbol_types = defaultdict(int)
    files_affected = set()
    
    for issue in issues:
        issue_types[issue['issue_type']] += 1
        symbol_types[issue['symbol_type']] += 1
        files_affected.add(issue['file_path'])
    
    print("\n=== STATISTICS ===")
    print(f"Total issues: {len(issues)}")
    print(f"Files affected: {len(files_affected)}")
    
    print("\nBy issue type:")
    for itype, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {itype.replace('_', ' ').title()}: {count}")
    
    print("\nBy symbol type:")
    for stype, count in sorted(symbol_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {stype.title()}: {count}")


def show_critical_files(issues: List[Dict], top_n: int = 5):
    """Show files with the most issues."""
    from collections import defaultdict
    
    file_counts = defaultdict(int)
    for issue in issues:
        file_counts[issue['file_path']] += 1
    
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n=== TOP {top_n} FILES WITH MOST ISSUES ===")
    for i, (file_path, count) in enumerate(sorted_files[:top_n], 1):
        short_path = str(Path(file_path).name)
        print(f"{i}. {short_path}: {count} issues")


def main():
    """Main function for the issue browser."""
    parser = argparse.ArgumentParser(
        description='Browse and filter docstring audit issues'
    )
    parser.add_argument(
        'report',
        nargs='?',
        default='docstring_audit_report.json',
        help='Path to the JSON audit report'
    )
    parser.add_argument(
        '--type',
        choices=['missing', 'outdated_params', 'missing_return'],
        help='Filter by issue type'
    )
    parser.add_argument(
        '--module',
        help='Filter by module name'
    )
    parser.add_argument(
        '--file',
        help='Filter by file path'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics only'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Load report
    try:
        issues = load_report(args.report)
    except FileNotFoundError:
        print(f"Error: Report file '{args.report}' not found.")
        sys.exit(1)
    
    # Apply filters
    filtered = filter_issues(issues, args.type, args.module, args.file)
    
    if args.interactive:
        interactive_mode(filtered)
    elif args.stats:
        show_statistics(filtered)
    else:
        # Display filtered issues
        print(f"Found {len(filtered)} issues")
        
        if len(filtered) > 0:
            print("\nShowing first 10 issues:")
            for i, issue in enumerate(filtered[:10], 1):
                display_issue(issue)
            
            if len(filtered) > 10:
                print(f"\n... and {len(filtered) - 10} more issues")


if __name__ == '__main__':
    main()
