# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python utility that analyzes both Claude Code and OpenCode session files to provide token usage summaries and cost estimates. The tool parses:
- **Claude Code**: `.jsonl` session files from `~/.claude/projects/`
- **OpenCode**: JSON session files from `~/.local/share/opencode/project/`

The analyzer generates comprehensive reports with token usage breakdowns, cost estimates, and usage trends. The project uses uv for dependency management and Python 3.12.

## Architecture

### Core Components

- **ClaudeUsageAnalyzer** (main.py:47): Main class that handles Claude Code session file discovery, parsing, and analysis
- **OpenCodeUsageAnalyzer** (main.py:400): Main class that handles OpenCode session file discovery, parsing, and analysis
- **SessionSummary** (main.py:30): Data class representing a single session with token usage metrics
- **TokenUsage** (main.py:22): Data class for tracking different types of token usage (input, output, cache read/write)

### Data Processing Flow

#### Claude Code Processing
1. **Session Discovery**: Recursively finds `.jsonl` files in `~/.claude/projects/` directory
2. **JSONL Parsing**: Processes each line of session files to extract usage data from assistant messages
3. **Cost Calculation**: Applies current Claude pricing from models.dev API to estimate costs per session

#### OpenCode Processing
1. **Project Discovery**: Finds project directories in `~/.local/share/opencode/project/`
2. **Session File Discovery**: Locates session info files and corresponding message files
3. **JSON Parsing**: Extracts token usage and cost data directly from OpenCode session files
4. **Provider/Model Analysis**: Groups usage by provider (Groq, Anthropic) and model

#### Combined Analysis
3. **Aggregation**: Groups usage by model type, provider, and project for summary reporting
4. **Unified Reporting**: Provides separate and combined views of Claude Code and OpenCode usage

### Key Features

#### Claude Code Support
- Supports multiple Claude models (Sonnet 4, 3.5 Sonnet, 3.5 Haiku) with accurate pricing from models.dev API
- Handles cache tokens (creation and read) for precise cost calculation
- JSONL parsing with robust error handling for malformed lines

#### OpenCode Support  
- Supports multiple AI providers (Groq, Anthropic) and models (GPT, Claude)
- Direct cost extraction from OpenCode session files
- Provider and model-specific breakdowns
- JSON parsing of session metadata and message files

#### Universal Features
- Provides breakdowns by project, model, provider, and recent session activity
- Combined analysis showing usage across both platforms
- Flexible filtering by project, model, or platform
- Comprehensive cost analysis and projections

## Usage

### Running the Analyzer

#### Analyze Both Platforms (Default)
```bash
uv run main.py                              # Analyze both Claude Code and OpenCode
```

#### Platform-Specific Analysis
```bash
uv run main.py --claude-only                # Analyze only Claude Code sessions
uv run main.py --opencode-only              # Analyze only OpenCode sessions
```

#### Custom Directories
```bash
uv run main.py --claude-dir /path/to/claude # Custom Claude directory
uv run main.py --opencode-dir /path/to/oc   # Custom OpenCode directory
```

#### Filtering Options
```bash
uv run main.py --recent 5                   # Show only last 5 sessions
uv run main.py --project my-project         # Filter by specific project
uv run main.py --model claude-sonnet-4      # Filter by specific model
```

The script automatically discovers:
- Claude Code directory at `~/.claude` 
- OpenCode directory at `~/.local/share/opencode`

### Expected Output

#### Claude Code Summary
- Overall statistics (total sessions, messages, tokens)
- Total token usage breakdown by type
- Estimated costs with model-specific pricing from models.dev API
- Usage breakdowns by model and project
- Recent session activity

#### OpenCode Summary  
- Overall statistics with provider breakdown
- Token usage and direct cost data
- Usage breakdowns by provider, model, and project
- Recent session activity with cost information

#### Combined Summary
- Unified view of total usage across both platforms
- Combined token counts and session statistics
- Recent sessions from both platforms chronologically sorted

## Development

### Setup
```bash
uv sync                                     # Install dependencies
```

### Running Tests and Validation
```bash
uv run main.py --help                       # Verify installation and see all options
uv run main.py --list-providers             # List available providers
uv run main.py --list-models anthropic      # List models for specific provider
```

### Dependencies
Project uses uv for dependency management with Python 3.12 as specified in `.python-version`. Main dependency is `requests>=2.31.0` for API calls.

## Pricing Configuration

### Claude Code Pricing
- Fetched dynamically from models.dev API with 1-hour caching
- Supports all available Claude models with current pricing
- Fallback to default Claude models if specific model pricing unavailable

### OpenCode Pricing  
- Uses cost data directly from OpenCode session files
- No separate pricing configuration needed
- Supports multiple providers (Groq, Anthropic) with their respective pricing