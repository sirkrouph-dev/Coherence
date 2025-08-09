import ast
import os

def get_classes(filepath):
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

core_dir = 'core'
for filename in os.listdir(core_dir):
    if filename.endswith('.py'):
        filepath = os.path.join(core_dir, filename)
        classes = get_classes(filepath)
        if classes:
            print(f"\n{filename}:")
            for cls in classes:
                print(f"  - {cls}")
