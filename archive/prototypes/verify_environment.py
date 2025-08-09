#!/usr/bin/env python3
"""
Environment Verification Script
Tests that all required components are properly installed and functional
"""

import sys
import platform

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def test_imports():
    """Test all critical imports"""
    print_header("Testing Package Imports")
    
    packages = {
        'numpy': None,
        'scipy': None,
        'pandas': None,
        'matplotlib': None,
        'seaborn': None,
        'sklearn': None,
        'cv2': None,
        'torch': None,
        'torchvision': None,
        'cupy': None,
        'psutil': None
    }
    
    for package_name in packages:
        try:
            if package_name == 'sklearn':
                import sklearn
                packages[package_name] = sklearn.__version__
            elif package_name == 'cv2':
                import cv2
                packages[package_name] = cv2.__version__
            else:
                module = __import__(package_name)
                packages[package_name] = module.__version__
            print(f"✓ {package_name:15} {packages[package_name]:>20}")
        except ImportError as e:
            print(f"✗ {package_name:15} {'FAILED':>20}")
            print(f"  Error: {e}")
    
    return packages

def test_cuda():
    """Test CUDA availability"""
    print_header("Testing CUDA Support")
    
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
            
            # Test basic CUDA operation
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print(f"✓ Basic CUDA tensor operation successful")
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
    
    try:
        import cupy
        print(f"\nCuPy version: {cupy.__version__}")
        # Test basic CuPy operation
        x = cupy.random.randn(100, 100)
        y = cupy.random.randn(100, 100)
        z = cupy.dot(x, y)
        print(f"✓ Basic CuPy operation successful")
    except Exception as e:
        print(f"✗ CuPy test failed: {e}")

def test_system_info():
    """Display system information"""
    print_header("System Information")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")
    
    try:
        import psutil
        print(f"\nMemory Information:")
        mem = psutil.virtual_memory()
        print(f"  Total: {mem.total / (1024**3):.2f} GB")
        print(f"  Available: {mem.available / (1024**3):.2f} GB")
        print(f"  Used: {mem.percent}%")
        
        print(f"\nCPU Information:")
        print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"  Logical cores: {psutil.cpu_count(logical=True)}")
        print(f"  Current frequency: {psutil.cpu_freq().current:.2f} MHz")
    except Exception as e:
        print(f"Could not get system stats: {e}")

def test_neuromorphic_imports():
    """Test local neuromorphic module imports"""
    print_header("Testing Neuromorphic Module Imports")
    
    modules = [
        'core.neurons',
        'core.synapses',
        'core.network',
        'core.encoding',
        'core.neuromodulation',
        'api.neuromorphic_api'
    ]
    
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}")
            print(f"  Error: {e}")

def main():
    print("\n" + "="*60)
    print(" NEUROMORPHIC COMPUTING ENVIRONMENT VERIFICATION")
    print("="*60)
    
    # Run all tests
    test_system_info()
    packages = test_imports()
    test_cuda()
    test_neuromorphic_imports()
    
    print("\n" + "="*60)
    print(" VERIFICATION COMPLETE")
    print("="*60)
    
    # Summary
    if all(v is not None for v in packages.values()):
        print("\n✓ All required packages are installed")
    else:
        failed = [k for k, v in packages.items() if v is None]
        print(f"\n✗ Failed packages: {', '.join(failed)}")
    
    print("\n✓ Environment is ready for neuromorphic computing experiments")

if __name__ == "__main__":
    main()
