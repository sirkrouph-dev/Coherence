#!/usr/bin/env python3
"""
Root directory organization script for the neuromorphic project.
Moves files from root to appropriate directories based on type and purpose.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Define file type mappings
FILE_MAPPINGS = {
    # Python files that should be moved to src/
    "src_files": [
        "test_engine.py",
        "test_memory_subsystem.py",
        "verify_installation.py",
    ],
    
    # Documentation files that should be moved to docs/
    "docs_files": [
        "ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "DOCKER.md",
        "JETSON_DEPLOYMENT.md",
        "LICENSE",
        "MANIFEST.in",
        "Makefile",
        "setup.cfg",
        "setup.py",
        # Additional markdown files to move
        "*.md",  # All other markdown files except README.md
    ],
    
    # Configuration files that should be moved to configs/
    "config_files": [
        "pyproject.toml",
        "ruff.toml",
        "requirements.txt",
        "requirements_gpu.txt",
        "requirements_jetson.txt",
    ],
    
    # Data and log files
    "data_files": [
        "installed_packages.txt",
    ],
    
    # Files to keep in root (core project files)
    "keep_in_root": [
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.override.yml",
    ]
}

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["src", "docs", "configs", "data", "logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}/")

def get_file_type(filename: str) -> str:
    """Determine the type of file based on extension and name."""
    ext = Path(filename).suffix.lower()
    
    if ext == ".py":
        return "src"
    elif ext == ".md" and filename != "README.md":
        return "docs"
    elif ext in [".json", ".yaml", ".yml", ".toml"]:
        return "configs"
    elif ext in [".log", ".txt"] and filename != "requirements.txt":
        return "data"
    else:
        return "unknown"

def organize_files():
    """Organize files in the root directory."""
    root = Path(".")
    moved_count = 0
    errors = []
    
    print("🔧 Organizing root directory...")
    
    # Process files based on predefined mappings
    for category, files in FILE_MAPPINGS.items():
        if category == "keep_in_root":
            continue
            
        target_dir = category.replace("_files", "")
        
        for filename in files:
            source = root / filename
            if source.exists():
                target = root / target_dir / filename
                try:
                    shutil.move(str(source), str(target))
                    print(f"✓ Moved {filename} → {target_dir}/")
                    moved_count += 1
                except Exception as e:
                    errors.append(f"Failed to move {filename}: {e}")
    
    # Process remaining files by type
    for file_path in root.iterdir():
        if file_path.is_file() and file_path.name not in FILE_MAPPINGS["keep_in_root"]:
            file_type = get_file_type(file_path.name)
            
            if file_type != "unknown":
                target_dir = root / file_type
                target_path = target_dir / file_path.name
                
                # Skip if already processed
                if file_path.name in [f for files in FILE_MAPPINGS.values() for f in files]:
                    continue
                    
                try:
                    shutil.move(str(file_path), str(target_path))
                    print(f"✓ Moved {file_path.name} → {file_type}/")
                    moved_count += 1
                except Exception as e:
                    errors.append(f"Failed to move {file_path.name}: {e}")
    
    return moved_count, errors

def create_organization_report():
    """Create a report of the organization."""
    report = """
# Root Directory Organization Report

## Summary
- Files moved: {moved_count}
- Errors: {error_count}

## New Directory Structure
```
neuron/
├── src/                    # Main source code
├── docs/                   # Documentation
├── configs/                # Configuration files
├── data/                   # Data files
├── logs/                   # Log files
├── tests/                  # Test files
├── tools/                  # Analysis tools
├── scripts/                # Utility scripts
├── core/                   # Core neuromorphic components
├── api/                    # API interfaces
├── engine/                 # Neural simulation engine
├── demo/                   # Demonstration scripts
├── examples/               # Example implementations
├── benchmarks/             # Performance benchmarks
├── archive/                # Historical files
└── README.md              # Project overview
```

## Files Kept in Root
- README.md (project overview)
- Dockerfile (containerization)
- docker-compose.yml (orchestration)
- docker-compose.override.yml (local overrides)

## Next Steps
1. Review moved files for appropriateness
2. Update import statements if needed
3. Update documentation references
4. Test that all functionality still works
"""
    return report

def main():
    """Main organization function."""
    print("🧹 Neuromorphic Project Root Organization")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Organize files
    moved_count, errors = organize_files()
    
    # Report results
    print(f"\n📊 Organization Complete!")
    print(f"✓ Files moved: {moved_count}")
    if errors:
        print(f"⚠️  Errors: {len(errors)}")
        for error in errors:
            print(f"   - {error}")
    
    # Create report
    report = create_organization_report().format(
        moved_count=moved_count,
        error_count=len(errors)
    )
    
    with open("archive/reports/latest_claims_assessment/organization_report.md", "w") as f:
        f.write(report)
    
    print(f"\n📋 Organization report saved to: archive/reports/latest_claims_assessment/organization_report.md")
    print("\n🎯 Next steps:")
    print("1. Review the moved files")
    print("2. Update any import statements")
    print("3. Test that functionality still works")
    print("4. Update documentation references")

if __name__ == "__main__":
    main()
