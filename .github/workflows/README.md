# GitHub Actions Workflows

This directory contains the CI/CD workflows for the Neuromorphic System project.

## Workflows

### 1. **lint.yml** - Code Quality Checks
- **Trigger**: Push to main/develop, Pull Requests
- **Tools**: 
  - Ruff (fast Python linter)
  - Black (code formatter)
  - Flake8 (style guide enforcement)
  - isort (import sorting)
  - mypy (type checking)
- **Purpose**: Ensures code quality and consistency

### 2. **test.yml** - Test and Coverage
- **Trigger**: Push to main/develop, Pull Requests
- **Matrix**: 
  - OS: Linux, Windows, macOS
  - Python: 3.9, 3.10, 3.11, 3.12
- **Features**:
  - Unit and integration tests
  - Coverage reporting to Codecov
  - GPU tests (on main branch only)
  - Coverage threshold enforcement (90%)

### 3. **publish.yml** - Build and Publish
- **Trigger**: Version tags (v*), Manual dispatch
- **Features**:
  - Build wheels for multiple platforms
  - Test wheels across Python versions
  - Publish to TestPyPI (automatic)
  - Publish to PyPI (requires manual approval)
  - Create GitHub releases

### 4. **ci.yml** - Comprehensive CI/CD Pipeline
- **Trigger**: All branches, Weekly schedule
- **Quality Gates**:
  1. Code Quality (linting, security checks)
  2. Test Matrix (full test suite)
  3. Documentation (build and coverage)
  4. Performance Benchmarks (main branch only)
  5. Dependency Security Check
- **Features**:
  - Parallel execution
  - Comprehensive reporting
  - Quality gate summary

## Setup Requirements

### Secrets Configuration
Add these secrets to your GitHub repository:

1. **TEST_PYPI_API_TOKEN**: API token for TestPyPI
   - Get from: https://test.pypi.org/manage/account/token/
   
2. **PYPI_API_TOKEN**: API token for PyPI
   - Get from: https://pypi.org/manage/account/token/
   
3. **CODECOV_TOKEN**: Token for Codecov integration
   - Get from: https://app.codecov.io/gh/your-org/your-repo/settings

### Repository Settings

1. **Branch Protection Rules** (for main branch):
   - Require pull request reviews
   - Require status checks to pass:
     - `quality-check`
     - `test-matrix`
     - `codecov/project`
   - Require branches to be up to date
   - Include administrators

2. **Environments**:
   - Create `testpypi` environment (auto-deploy)
   - Create `pypi` environment (requires manual approval)

## Usage

### Running Tests Locally
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check .
black --check .
flake8 .

# Run tests with coverage
pytest --cov=core --cov=api --cov-report=html

# Run specific test matrix
pytest tests/ -v -m "not slow and not gpu"
```

### Creating a Release
1. Update version in `pyproject.toml`
2. Commit changes: `git commit -am "Bump version to x.y.z"`
3. Create tag: `git tag -a vx.y.z -m "Release version x.y.z"`
4. Push tag: `git push origin vx.y.z`
5. Workflows will automatically:
   - Build and test wheels
   - Publish to TestPyPI
   - Create GitHub release
   - Await manual approval for PyPI

### Manual Workflow Dispatch
You can manually trigger workflows from GitHub Actions tab:
- Select workflow
- Click "Run workflow"
- Choose branch and parameters

## Workflow Badges

Add these badges to your README:

```markdown
[![CI/CD](https://github.com/your-org/neuromorphic-system/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/neuromorphic-system/actions/workflows/ci.yml)
[![Tests](https://github.com/your-org/neuromorphic-system/actions/workflows/test.yml/badge.svg)](https://github.com/your-org/neuromorphic-system/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/your-org/neuromorphic-system/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/neuromorphic-system)
[![PyPI](https://img.shields.io/pypi/v/neuromorphic-system.svg)](https://pypi.org/project/neuromorphic-system/)
```

## Troubleshooting

### Common Issues

1. **Codecov not reporting**:
   - Ensure CODECOV_TOKEN is set in repository secrets
   - Check that coverage.xml is generated

2. **Publishing fails**:
   - Verify API tokens are valid
   - Check package name availability on PyPI

3. **Tests timeout**:
   - Increase timeout in workflow: `timeout-minutes: 30`
   - Check for infinite loops in tests

4. **Cache issues**:
   - Clear cache from Actions → Caches
   - Update cache key in workflow

## Maintenance

### Updating Dependencies
- Dependabot is configured for automatic updates
- Review and merge dependency PRs regularly

### Updating Python Versions
When new Python version is released:
1. Add to test matrix in workflows
2. Update `pyproject.toml` classifiers
3. Test locally before pushing

### Performance Optimization
- Use caching for dependencies
- Run jobs in parallel where possible
- Use `fail-fast: false` for comprehensive testing
- Consider using self-hosted runners for heavy workloads
