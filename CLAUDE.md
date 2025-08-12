# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python utility that analyzes Claude Code session files to provide token usage summaries and cost estimates. The tool parses `.jsonl` session files from `~/.claude/projects/` and generates comprehensive reports.

## Architecture

### Core Components

- **ClaudeUsageAnalyzer** (claude_usage_analyzer.py:42): Main class that handles session file discovery, parsing, and analysis
- **SessionSummary** (claude_usage_analyzer.py:28): Data class representing a single Claude session with token usage metrics
- **TokenUsage** (claude_usage_analyzer.py:20): Data class for tracking different types of token usage (input, output, cache read/write)

### Data Processing Flow

1. **Session Discovery**: Recursively finds `.jsonl` files in `~/.claude/projects/` directory
2. **JSON Parsing**: Processes each line of session files to extract usage data from assistant messages
3. **Aggregation**: Groups usage by model type and project for summary reporting
4. **Cost Calculation**: Applies current Claude pricing to estimate costs per session and overall

### Key Features

- Supports multiple Claude models (Sonnet 4, 3.5 Sonnet, 3.5 Haiku) with accurate pricing
- Handles cache tokens (creation and read) for precise cost calculation  
- Provides breakdowns by project, model, and recent session activity
- Robust error handling for malformed JSON lines

## Usage

### Running the Analyzer

```bash
python3 claude_usage_analyzer.py
```

The script automatically discovers the Claude directory at `~/.claude` and processes all session files.

### Expected Output

- Overall statistics (total sessions, messages, tokens)
- Total token usage breakdown by type
- Estimated costs with model-specific pricing
- Usage breakdowns by model and project
- Recent session activity (last 10 sessions)

## Pricing Configuration

Current pricing is hardcoded in `ClaudeUsageAnalyzer.PRICING` (claude_usage_analyzer.py:44) and reflects January 2025 rates. Update these values when Claude pricing changes.