import ast
import os
from collections import defaultdict
from pathlib import Path

def analyze_imports(filepath):
    """Extract imports from a Python file."""
    imports = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(('api', 'core', 'demo', 'scripts')):
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(('api', 'core', 'demo', 'scripts')):
                        imports.append(alias.name)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return imports

def main():
    # Map module dependencies
    dependencies = defaultdict(list)
    modules = ['api', 'core', 'demo', 'scripts']
    
    print("=== MODULE DEPENDENCY ANALYSIS ===\n")
    
    for module in modules:
        if not os.path.exists(module):
            continue
            
        for root, dirs, files in os.walk(module):
            # Skip venv and __pycache__
            dirs[:] = [d for d in dirs if d not in ['__pycache__', 'venv_neuron']]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    relative_path = filepath.replace('\\', '/')
                    imports = analyze_imports(filepath)
                    if imports:
                        dependencies[relative_path] = imports
    
    # Print dependency structure
    print("Module Dependencies:")
    print("-" * 50)
    for filepath, deps in sorted(dependencies.items()):
        if deps:
            print(f"\n{filepath}:")
            for dep in sorted(set(deps)):
                print(f"  → {dep}")
    
    # Check for circular dependencies
    print("\n" + "=" * 50)
    print("Checking for circular dependencies...")
    print("-" * 50)
    
    circular_found = False
    checked_pairs = set()
    
    for file1, deps1 in dependencies.items():
        module1 = file1.split('/')[0]
        for dep in deps1:
            dep_module = dep.split('.')[0]
            
            # Check if the dependency also imports from the original module
            for file2, deps2 in dependencies.items():
                if file2.startswith(dep_module):
                    for dep2 in deps2:
                        if dep2.startswith(module1):
                            pair = tuple(sorted([file1, file2]))
                            if pair not in checked_pairs:
                                checked_pairs.add(pair)
                                print(f"Potential circular dependency:")
                                print(f"  {file1} → {dep}")
                                print(f"  {file2} → {dep2}")
                                print()
                                circular_found = True
    
    if not circular_found:
        print("No circular dependencies detected!")
    
    # Count modules without external dependencies
    print("\n" + "=" * 50)
    print("Module Independence Analysis:")
    print("-" * 50)
    
    for module in modules:
        independent_files = []
        dependent_files = []
        
        for filepath, deps in dependencies.items():
            if filepath.startswith(module):
                if not deps:
                    independent_files.append(filepath)
                else:
                    # Check if dependencies are only within same module
                    external_deps = [d for d in deps if not d.startswith(module)]
                    if not external_deps:
                        independent_files.append(filepath)
                    else:
                        dependent_files.append(filepath)
        
        if independent_files or dependent_files:
            print(f"\n{module.upper()} module:")
            print(f"  Independent files: {len(independent_files)}")
            print(f"  Files with external deps: {len(dependent_files)}")

if __name__ == "__main__":
    main()
