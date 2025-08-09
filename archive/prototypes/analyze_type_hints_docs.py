import ast
import os
from typing import List, Tuple, Dict, Any
from pathlib import Path

class TypeHintAndDocAnalyzer(ast.NodeVisitor):
    """Analyze Python code for missing type hints and docstrings."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.issues = {
            'missing_type_hints': [],
            'missing_docstrings': [],
            'undocumented_public_apis': []
        }
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions for type hints and docstrings."""
        # Check if public API (doesn't start with _)
        is_public = not node.name.startswith('_')
        
        # Check for docstring
        has_docstring = (
            node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, (ast.Str, ast.Constant))
        )
        
        if is_public and not has_docstring:
            self.issues['missing_docstrings'].append(f"Function: {node.name} (line {node.lineno})")
            self.issues['undocumented_public_apis'].append(f"Function: {node.name} (line {node.lineno})")
        
        # Check for return type hint
        if not node.returns and node.name != '__init__':
            self.issues['missing_type_hints'].append(f"Function {node.name} missing return type (line {node.lineno})")
        
        # Check for parameter type hints
        for arg in node.args.args:
            if arg.arg != 'self' and not arg.annotation:
                self.issues['missing_type_hints'].append(
                    f"Function {node.name}, parameter '{arg.arg}' missing type hint (line {node.lineno})"
                )
        
        self.generic_visit(node)
    
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check class definitions for docstrings."""
        is_public = not node.name.startswith('_')
        
        # Check for docstring
        has_docstring = (
            node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, (ast.Str, ast.Constant))
        )
        
        if is_public and not has_docstring:
            self.issues['missing_docstrings'].append(f"Class: {node.name} (line {node.lineno})")
            self.issues['undocumented_public_apis'].append(f"Class: {node.name} (line {node.lineno})")
        
        self.generic_visit(node)

def analyze_file(filepath: str) -> Dict[str, List[str]]:
    """Analyze a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        analyzer = TypeHintAndDocAnalyzer(filepath)
        analyzer.visit(tree)
        return analyzer.issues
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return {'missing_type_hints': [], 'missing_docstrings': [], 'undocumented_public_apis': []}

def main():
    """Main analysis function."""
    modules = ['api', 'core', 'demo', 'scripts']
    
    all_issues = {
        'missing_type_hints': {},
        'missing_docstrings': {},
        'undocumented_public_apis': {}
    }
    
    print("=== TYPE HINTS AND DOCUMENTATION ANALYSIS ===\n")
    
    # Analyze each module
    for module in modules:
        if not os.path.exists(module):
            continue
            
        for root, dirs, files in os.walk(module):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', 'venv_neuron']]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    issues = analyze_file(filepath)
                    
                    relative_path = filepath.replace('\\', '/')
                    
                    if issues['missing_type_hints']:
                        all_issues['missing_type_hints'][relative_path] = issues['missing_type_hints']
                    if issues['missing_docstrings']:
                        all_issues['missing_docstrings'][relative_path] = issues['missing_docstrings']
                    if issues['undocumented_public_apis']:
                        all_issues['undocumented_public_apis'][relative_path] = issues['undocumented_public_apis']
    
    # Report missing type hints
    print("Missing Type Hints:")
    print("-" * 50)
    total_missing_hints = 0
    for filepath, hints in sorted(all_issues['missing_type_hints'].items()):
        if hints:
            print(f"\n{filepath}:")
            for hint in hints[:5]:  # Show first 5 issues per file
                print(f"  • {hint}")
            if len(hints) > 5:
                print(f"  ... and {len(hints) - 5} more")
            total_missing_hints += len(hints)
    
    print(f"\nTotal missing type hints: {total_missing_hints}")
    
    # Report missing docstrings
    print("\n" + "=" * 50)
    print("Missing Docstrings:")
    print("-" * 50)
    total_missing_docs = 0
    for filepath, docs in sorted(all_issues['missing_docstrings'].items()):
        if docs:
            print(f"\n{filepath}:")
            for doc in docs[:5]:  # Show first 5 issues per file
                print(f"  • {doc}")
            if len(docs) > 5:
                print(f"  ... and {len(docs) - 5} more")
            total_missing_docs += len(docs)
    
    print(f"\nTotal missing docstrings: {total_missing_docs}")
    
    # Report undocumented public APIs
    print("\n" + "=" * 50)
    print("Undocumented Public APIs:")
    print("-" * 50)
    total_undocumented = 0
    for filepath, apis in sorted(all_issues['undocumented_public_apis'].items()):
        if apis:
            print(f"\n{filepath}:")
            for api in apis[:3]:  # Show first 3 issues per file
                print(f"  • {api}")
            if len(apis) > 3:
                print(f"  ... and {len(apis) - 3} more")
            total_undocumented += len(apis)
    
    print(f"\nTotal undocumented public APIs: {total_undocumented}")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS:")
    print("-" * 50)
    print(f"Files analyzed: {len(all_issues['missing_type_hints']) + len(all_issues['missing_docstrings'])}")
    print(f"Total missing type hints: {total_missing_hints}")
    print(f"Total missing docstrings: {total_missing_docs}")
    print(f"Total undocumented public APIs: {total_undocumented}")

if __name__ == "__main__":
    main()
