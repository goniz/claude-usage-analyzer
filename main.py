#!/usr/bin/env python3
"""
Claude Code Usage Analyzer

Parses Claude Code session files and displays token usage summary with cost estimates.
Session files are stored in ~/.claude/projects/ as .jsonl files.
"""

import json
import os
import glob
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class SessionSummary:
    session_id: str
    project_path: str
    messages_count: int = 0
    token_usage: TokenUsage = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    model: str = ""
    
    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = TokenUsage()


class ClaudeUsageAnalyzer:
    # Claude pricing (as of January 2025) - prices per 1M tokens
    PRICING = {
        # Latest Claude 4 Models
        'claude-4.1-opus': {
            'input': 15.00,
            'output': 75.00,
            'cache_write': 18.75,
            'cache_read': 1.50,
        },
        'claude-opus-4.1': {
            'input': 15.00,
            'output': 75.00,
            'cache_write': 18.75,
            'cache_read': 1.50,
        },
        'claude-4-opus': {
            'input': 15.00,
            'output': 75.00,
            'cache_write': 18.75,
            'cache_read': 1.50,
        },
        'claude-sonnet-4-20250514': {
            'input': 3.00,  # ≤200K tokens
            'output': 15.00,  # ≤200K tokens
            'cache_write': 3.75,
            'cache_read': 0.30,
        },
        'claude-4-sonnet': {
            'input': 3.00,  # ≤200K tokens
            'output': 15.00,  # ≤200K tokens
            'cache_write': 3.75,
            'cache_read': 0.30,
        },
        'claude-3-5-haiku-20241022': {
            'input': 0.80,
            'output': 4.00,
            'cache_write': 1.00,
            'cache_read': 0.08,
        },
        'claude-haiku-3-5': {
            'input': 0.80,
            'output': 4.00,
            'cache_write': 1.00,
            'cache_read': 0.08,
        },
        
        # Legacy Models
        'claude-3-5-sonnet-20241022': {
            'input': 3.00,
            'output': 15.00,
            'cache_write': 3.75,
            'cache_read': 0.30,
        },
        'claude-3-opus': {
            'input': 15.00,
            'output': 75.00,
            'cache_write': 18.75,
            'cache_read': 1.50,
        },
        'claude-3-haiku': {
            'input': 0.25,
            'output': 1.25,
            'cache_write': 0.30,
            'cache_read': 0.025,
        },
        
        # Common model name variations
        'claude-4': {  # Default to Sonnet 4
            'input': 3.00,
            'output': 15.00,
            'cache_write': 3.75,
            'cache_read': 0.30,
        },
        'claude-3.5-sonnet': {
            'input': 3.00,
            'output': 15.00,
            'cache_write': 3.75,
            'cache_read': 0.30,
        },
        'claude-3.5-haiku': {
            'input': 0.80,
            'output': 4.00,
            'cache_write': 1.00,
            'cache_read': 0.08,
        },
    }
    
    def __init__(self, claude_dir: str = None):
        self.claude_dir = claude_dir or os.path.expanduser('~/.claude')
        self.projects_dir = os.path.join(self.claude_dir, 'projects')
        
    def find_session_files(self) -> List[str]:
        """Find all .jsonl session files in the projects directory."""
        pattern = os.path.join(self.projects_dir, '**', '*.jsonl')
        return glob.glob(pattern, recursive=True)
    
    def parse_session_file(self, filepath: str) -> SessionSummary:
        """Parse a single session file and extract usage information."""
        filename = os.path.basename(filepath)
        session_id = filename.replace('.jsonl', '')
        project_path = os.path.basename(os.path.dirname(filepath))
        
        summary = SessionSummary(
            session_id=session_id,
            project_path=project_path
        )
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        summary.messages_count += 1
                        
                        # Extract timestamp
                        if 'timestamp' in data:
                            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                            if summary.start_time is None or timestamp < summary.start_time:
                                summary.start_time = timestamp
                            if summary.end_time is None or timestamp > summary.end_time:
                                summary.end_time = timestamp
                        
                        # Extract usage information from assistant messages
                        if (data.get('type') == 'assistant' and 
                            'message' in data and 
                            'usage' in data['message']):
                            usage = data['message']['usage']
                            
                            summary.token_usage.input_tokens += usage.get('input_tokens', 0)
                            summary.token_usage.output_tokens += usage.get('output_tokens', 0)
                            summary.token_usage.cache_creation_input_tokens += usage.get('cache_creation_input_tokens', 0)
                            summary.token_usage.cache_read_input_tokens += usage.get('cache_read_input_tokens', 0)
                            
                            # Extract model information
                            if 'model' in data['message'] and not summary.model:
                                summary.model = data['message']['model']
                                
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in {filepath} line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
        return summary
    
    def calculate_cost(self, token_usage: TokenUsage, model: str) -> float:
        """Calculate estimated cost based on token usage and model."""
        if model not in self.PRICING:
            # Use Sonnet 4 pricing as default
            model = 'claude-sonnet-4-20250514'
            
        pricing = self.PRICING[model]
        
        cost = 0.0
        cost += (token_usage.input_tokens / 1_000_000) * pricing['input']
        cost += (token_usage.output_tokens / 1_000_000) * pricing['output']
        cost += (token_usage.cache_creation_input_tokens / 1_000_000) * pricing['cache_write']
        cost += (token_usage.cache_read_input_tokens / 1_000_000) * pricing['cache_read']
        
        return cost
    
    def analyze_all_sessions(self) -> List[SessionSummary]:
        """Analyze all session files and return summaries."""
        session_files = self.find_session_files()
        summaries = []
        
        print(f"Found {len(session_files)} session files to analyze...")
        
        for filepath in session_files:
            summary = self.parse_session_file(filepath)
            summaries.append(summary)
            
        return summaries
    
    def print_summary(self, summaries: List[SessionSummary]):
        """Print a comprehensive summary of token usage and costs."""
        if not summaries:
            print("No session data found.")
            return
            
        # Overall statistics
        total_sessions = len(summaries)
        total_messages = sum(s.messages_count for s in summaries)
        
        # Aggregate token usage
        total_tokens = TokenUsage()
        model_usage = defaultdict(TokenUsage)
        project_usage = defaultdict(TokenUsage)
        total_cost = 0.0
        model_costs = defaultdict(float)
        
        for summary in summaries:
            usage = summary.token_usage
            total_tokens.input_tokens += usage.input_tokens
            total_tokens.output_tokens += usage.output_tokens
            total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
            total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens
            
            # Group by model
            model = summary.model or 'unknown'
            model_usage[model].input_tokens += usage.input_tokens
            model_usage[model].output_tokens += usage.output_tokens
            model_usage[model].cache_creation_input_tokens += usage.cache_creation_input_tokens
            model_usage[model].cache_read_input_tokens += usage.cache_read_input_tokens
            
            # Group by project
            project_usage[summary.project_path].input_tokens += usage.input_tokens
            project_usage[summary.project_path].output_tokens += usage.output_tokens
            project_usage[summary.project_path].cache_creation_input_tokens += usage.cache_creation_input_tokens
            project_usage[summary.project_path].cache_read_input_tokens += usage.cache_read_input_tokens
            
            # Calculate costs
            session_cost = self.calculate_cost(usage, summary.model or 'claude-sonnet-4-20250514')
            total_cost += session_cost
            model_costs[model] += session_cost
        
        # Print results
        print("\n" + "=" * 80)
        print("CLAUDE CODE USAGE SUMMARY")
        print("=" * 80)
        
        print(f"\nOverall Statistics:")
        print(f"  Total Sessions: {total_sessions:,}")
        print(f"  Total Messages: {total_messages:,}")
        
        print(f"\nTotal Token Usage:")
        print(f"  Input Tokens:              {total_tokens.input_tokens:,}")
        print(f"  Output Tokens:             {total_tokens.output_tokens:,}")
        print(f"  Cache Creation Tokens:     {total_tokens.cache_creation_input_tokens:,}")
        print(f"  Cache Read Tokens:         {total_tokens.cache_read_input_tokens:,}")
        print(f"  Total Tokens:              {total_tokens.input_tokens + total_tokens.output_tokens + total_tokens.cache_creation_input_tokens + total_tokens.cache_read_input_tokens:,}")
        
        print(f"\nEstimated Total Cost: ${total_cost:.2f}")
        
        # Model breakdown
        if len(model_usage) > 1:
            print(f"\nUsage by Model:")
            for model, usage in sorted(model_usage.items()):
                cost = model_costs[model]
                print(f"  {model}:")
                print(f"    Input: {usage.input_tokens:,}, Output: {usage.output_tokens:,}")
                print(f"    Cache Write: {usage.cache_creation_input_tokens:,}, Cache Read: {usage.cache_read_input_tokens:,}")
                print(f"    Estimated Cost: ${cost:.2f}")
        
        # Project breakdown
        if len(project_usage) > 1:
            print(f"\nUsage by Project:")
            for project, usage in sorted(project_usage.items(), key=lambda x: x[1].input_tokens + x[1].output_tokens, reverse=True):
                total_project_tokens = usage.input_tokens + usage.output_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
                print(f"  {project}: {total_project_tokens:,} total tokens")
        
        # Recent sessions
        recent_sessions = sorted([s for s in summaries if s.start_time], 
                               key=lambda x: x.start_time, reverse=True)[:10]
        
        if recent_sessions:
            print(f"\nRecent Sessions (Last 10):")
            for session in recent_sessions:
                total_session_tokens = (session.token_usage.input_tokens + 
                                      session.token_usage.output_tokens + 
                                      session.token_usage.cache_creation_input_tokens + 
                                      session.token_usage.cache_read_input_tokens)
                cost = self.calculate_cost(session.token_usage, session.model or 'claude-sonnet-4-20250514')
                time_str = session.start_time.strftime('%Y-%m-%d %H:%M') if session.start_time else 'Unknown'
                print(f"  {time_str} - {session.project_path}: {total_session_tokens:,} tokens (${cost:.2f})")


def main():
    analyzer = ClaudeUsageAnalyzer()
    
    if not os.path.exists(analyzer.projects_dir):
        print(f"Error: Claude projects directory not found at {analyzer.projects_dir}")
        print("Make sure Claude Code has been used and session data exists.")
        return
    
    summaries = analyzer.analyze_all_sessions()
    analyzer.print_summary(summaries)


if __name__ == '__main__':
    main()