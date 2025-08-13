# Claude Usage Analyzer

A Python utility that analyzes AI coding session files to provide comprehensive token usage summaries and cost estimates. Supports both [Claude Code](https://claude.ai/code) and [OpenCode](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter) platforms.

## Features

- **Multi-Platform Support**: Analyze sessions from both Claude Code and OpenCode
- **Comprehensive Analysis**: Token usage breakdowns, cost estimates, and usage trends
- **Dynamic Pricing**: Real-time pricing data from models.dev API for accurate cost calculations
- **Flexible Filtering**: Filter by project, model, platform, or recent activity
- **Detailed Reporting**: Per-project, per-model, and combined usage summaries

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and requires Python 3.12+.

```bash
# Clone the repository
git clone <repository-url>
cd claude-usage-analyzer

# Install dependencies
uv sync
```

## Usage

### Basic Analysis

```bash
# Analyze both Claude Code and OpenCode sessions
uv run main.py

# Analyze only Claude Code sessions
uv run main.py --claude-only

# Analyze only OpenCode sessions  
uv run main.py --opencode-only
```

### Filtering Options

```bash
# Show only recent sessions
uv run main.py --recent 5

# Filter by specific project
uv run main.py --project my-project

# Filter by specific model
uv run main.py --model claude-sonnet-4

# Custom directories
uv run main.py --claude-dir /path/to/claude --opencode-dir /path/to/opencode
```

### Model and Provider Information

```bash
# List all available providers
uv run main.py --list-providers

# List models for a specific provider
uv run main.py --list-models anthropic
uv run main.py --list-models groq
```

## Default Locations

The tool automatically discovers session files in these default locations:

- **Claude Code**: `~/.claude/projects/` (`.jsonl` files)
- **OpenCode**: `~/.local/share/opencode/project/` (JSON files)

## Output Format

The analyzer provides detailed reports including:

### Session Statistics
- Total sessions analyzed
- Total messages processed
- Overall token usage by type (input, output, cache)

### Cost Analysis
- Real-time pricing from models.dev API
- Per-session and total cost estimates
- Cost breakdowns by model and provider

### Usage Breakdowns
- **By Model**: Token usage and costs per AI model
- **By Project**: Usage patterns across different projects
- **By Provider**: Comparison between different AI providers
- **Recent Activity**: Timeline of recent sessions

### Example Output

```
=== Claude Code Usage Summary ===
Total Sessions: 15
Total Messages: 127
Total Tokens: 89,432

Token Usage Breakdown:
  Input Tokens: 45,123
  Output Tokens: 32,109
  Cache Creation: 8,456
  Cache Read: 3,744

Estimated Total Cost: $2.34

=== Usage by Model ===
claude-sonnet-4: 65,234 tokens ($1.89)
claude-3-5-sonnet: 24,198 tokens ($0.45)
```

## Data Sources

### Claude Code Sessions
- Parses `.jsonl` files containing session data
- Extracts token usage from assistant messages
- Applies current pricing from models.dev API

### OpenCode Sessions  
- Processes JSON session files and message data
- Supports multiple providers (Groq, Anthropic, etc.)
- Uses direct cost data when available

## Development

```bash
# Install dependencies
uv sync

# Run with development options
uv run main.py --help

# Validate installation
uv run main.py --list-providers
```

## Requirements

- Python 3.12+
- uv package manager
- requests library (installed via uv sync)

## License

[Add your license here]