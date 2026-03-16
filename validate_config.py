#!/usr/bin/env python3
"""
Validate pyproject.toml and setup.py configuration.
"""

import sys
import tomli


def validate_pyproject():
    """Validate pyproject.toml structure."""
    print("Validating pyproject.toml...")
    
    try:
        with open("pyproject.toml", "rb") as f:
            config = tomli.load(f)
        
        # Check required sections
        assert "build-system" in config, "Missing [build-system]"
        assert "project" in config, "Missing [project]"
        
        # Check project metadata
        project = config["project"]
        assert "name" in project, "Missing project.name"
        assert "version" in project, "Missing project.version"
        assert "dependencies" in project, "Missing project.dependencies"
        
        # Check dependencies is a list
        assert isinstance(project["dependencies"], list), "project.dependencies must be a list"
        
        print("✓ pyproject.toml is valid")
        return True
        
    except FileNotFoundError:
        print("❌ pyproject.toml not found")
        return False
    except Exception as e:
        print(f"❌ pyproject.toml validation failed: {e}")
        return False


def validate_setup():
    """Validate setup.py exists and is importable."""
    print("\nValidating setup.py...")
    
    try:
        with open("setup.py", "r") as f:
            content = f.read()
        
        # Basic checks
        assert "from setuptools import setup" in content or "import setuptools" in content
        assert "CUDAExtension" in content
        assert "get_extensions" in content
        
        print("✓ setup.py exists and contains required components")
        return True
        
    except FileNotFoundError:
        print("❌ setup.py not found")
        return False
    except Exception as e:
        print(f"❌ setup.py validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("=" * 60)
    print("Configuration Validation")
    print("=" * 60)
    
    pyproject_ok = validate_pyproject()
    setup_ok = validate_setup()
    
    print("\n" + "=" * 60)
    if pyproject_ok and setup_ok:
        print("✓ All configuration files are valid")
        return 0
    else:
        print("❌ Configuration validation failed")
        return 1


if __name__ == "__main__":
    # Try to import tomli, fall back to tomllib for Python 3.11+
    try:
        import tomli
    except ImportError:
        try:
            import tomllib as tomli
        except ImportError:
            print("Error: tomli or tomllib required for validation")
            print("Install with: pip install tomli")
            sys.exit(1)
    
    sys.exit(main())
