# Contributing to Neuron

Thank you for your interest in contributing to Neuron! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branching Model](#branching-model)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/neuron.git
   cd neuron
   ```
3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/original-owner/neuron.git
   ```
4. Create a new branch for your feature or bug fix

## Development Setup

### Prerequisites

- Go 1.21 or higher
- Git
- Make (optional but recommended)

### Installation

1. Install dependencies:
   ```bash
   go mod download
   ```

2. Build the project:
   ```bash
   go build ./...
   ```

3. Run tests:
   ```bash
   go test ./...
   ```

## Branching Model

We follow a modified Git Flow branching model:

### Main Branches

- **`main`** - Production-ready code. All releases are tagged from this branch.
- **`develop`** - Integration branch for features. This is the default branch for PRs.

### Supporting Branches

- **Feature branches** (`feature/description`)
  - Branch from: `develop`
  - Merge into: `develop`
  - Naming: `feature/short-description` (e.g., `feature/add-memory-module`)

- **Bugfix branches** (`bugfix/description`)
  - Branch from: `develop`
  - Merge into: `develop`
  - Naming: `bugfix/short-description` (e.g., `bugfix/fix-memory-leak`)

- **Hotfix branches** (`hotfix/description`)
  - Branch from: `main`
  - Merge into: `main` and `develop`
  - Naming: `hotfix/short-description` (e.g., `hotfix/critical-security-fix`)

- **Release branches** (`release/version`)
  - Branch from: `develop`
  - Merge into: `main` and `develop`
  - Naming: `release/v1.2.0`

### Branch Creation Examples

```bash
# Feature branch
git checkout -b feature/add-new-component develop

# Bugfix branch
git checkout -b bugfix/fix-memory-issue develop

# Hotfix branch
git checkout -b hotfix/security-patch main

# Release branch
git checkout -b release/v1.2.0 develop
```

## Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for clear and structured commit messages.

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that don't affect code meaning (formatting, missing semicolons, etc.)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Performance improvement
- **test**: Adding or updating tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Scope

The scope should be the name of the package or module affected (e.g., `network`, `memory`, `core`, `cli`).

### Subject

- Use imperative mood ("add feature" not "added feature")
- Don't capitalize first letter
- No period at the end
- Limit to 50 characters

### Examples

```bash
# Feature
git commit -m "feat(network): add WebSocket support for real-time updates"

# Bug fix
git commit -m "fix(memory): resolve memory leak in cache cleanup"

# Documentation
git commit -m "docs(api): update REST API documentation"

# Breaking change
git commit -m "feat(core)!: redesign plugin architecture

BREAKING CHANGE: Plugin API has been completely redesigned.
Existing plugins will need to be updated to the new interface."

# Multi-line commit
git commit -m "fix(network): handle connection timeout gracefully

- Add retry logic with exponential backoff
- Improve error messages for timeout scenarios
- Add configuration option for timeout duration

Fixes #123"
```

## Pull Request Process

1. **Before Creating a PR:**
   - Ensure your branch is up to date with the target branch
   - Run all tests locally: `go test ./...`
   - Run linting: `golangci-lint run`
   - Update documentation if needed
   - Add/update tests for new functionality

2. **Creating a PR:**
   - Use a clear, descriptive title following the commit message format
   - Fill out the PR template completely
   - Link related issues using keywords (Fixes #123, Closes #456)
   - Add appropriate labels
   - Request reviews from maintainers

3. **PR Requirements:**
   - All CI checks must pass
   - Code coverage should not decrease
   - At least one approving review from a maintainer
   - No unresolved conversations
   - Branch must be up to date with target branch

4. **After PR Approval:**
   - Squash commits if requested
   - Ensure commit message follows guidelines
   - Delete branch after merge (if it's a fork branch)

## Testing Guidelines

### Test Requirements

- All new features must include unit tests
- Bug fixes should include a test that reproduces the issue
- Maintain or improve code coverage (minimum 80%)
- Tests should be deterministic and not rely on external services

### Running Tests

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run tests for specific package
go test ./pkg/network

# Run tests with race detection
go test -race ./...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Test Structure

- Follow Go testing conventions
- Use table-driven tests where appropriate
- Mock external dependencies
- Use meaningful test names that describe the scenario

Example:
```go
func TestModuleName_FunctionName_Scenario(t *testing.T) {
    // Test implementation
}
```

## Documentation

### Code Documentation

- All exported functions, types, and packages must have GoDoc comments
- Comments should start with the name of the element
- Include examples in documentation where helpful

### Project Documentation

- Update README.md for user-facing changes
- Update technical documentation in `/docs`
- Include inline code comments for complex logic
- Update CHANGELOG.md following Keep a Changelog format

## Community

### Getting Help

- Check existing issues and discussions
- Join our community chat/forum
- Attend community meetings (if applicable)

### Reporting Issues

- Use issue templates
- Provide minimal reproducible examples
- Include system information and versions
- Search existing issues before creating new ones

### Feature Requests

- Use the feature request template
- Explain the use case and motivation
- Provide examples of how it would work
- Be open to alternative solutions

## Recognition

Contributors will be recognized in:
- The CONTRIBUTORS file
- Release notes
- Project documentation

Thank you for contributing to Neuron! 🎉
