#!/usr/bin/env python3
"""
Claude Code Usage Analyzer

Parses Claude Code session files and displays token usage summary with cost estimates.
Session files are stored in ~/.claude/projects/ as .jsonl files.
"""

import argparse
import json
import os
import glob
import requests
import time
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
    # Models.dev API endpoint
    MODELS_API_URL = "https://models.dev/api.json"
    
    def __init__(self, claude_dir: str = None, quiet: bool = False):
        self.claude_dir = claude_dir or os.path.expanduser('~/.claude')
        self.projects_dir = os.path.join(self.claude_dir, 'projects')
        self._pricing_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 3600  # 1 hour cache TTL
        self._quiet = quiet
        self._recent_count = 10
    
    def _fetch_pricing_from_api(self) -> Dict:
        """Fetch pricing data from models.dev API."""
        try:
            response = requests.get(self.MODELS_API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Transform API data to our pricing format
            pricing = {}
            
            # Check multiple possible sections for Claude models
            sections_to_check = ['anthropic', 'google-vertex-anthropic']
            
            for section_name in sections_to_check:
                section_data = data.get(section_name, {})
                if section_data and isinstance(section_data, dict):
                    # Check if there's a 'models' key containing the actual model data
                    models_data = section_data.get('models', {})
                    if isinstance(models_data, dict):
                        for model_id, model_data in models_data.items():
                            if not isinstance(model_data, dict):
                                continue
                                
                            cost_data = model_data.get('cost', {})
                            
                            if model_id and 'claude' in model_id.lower() and cost_data:
                                input_price = cost_data.get('input')
                                output_price = cost_data.get('output')
                                cache_write_price = cost_data.get('cache_write')
                                cache_read_price = cost_data.get('cache_read')
                                
                                if input_price is not None and output_price is not None:
                                    pricing[model_id] = {
                                        'input': input_price,
                                        'output': output_price,
                                        'cache_write': cache_write_price or input_price * 1.25,
                                        'cache_read': cache_read_price or input_price * 0.1,
                                    }
            
            if pricing and not self._quiet:
                print(f"Fetched pricing for {len(pricing)} Claude models from API")
            return pricing
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch pricing from models.dev API: {e}")
    
    def _get_pricing(self) -> Dict:
        """Get pricing data with caching."""
        current_time = time.time()
        
        # Check if cache is valid
        if (self._pricing_cache is not None and 
            self._cache_timestamp is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._pricing_cache
        
        # Fetch fresh pricing data (required)
        try:
            api_pricing = self._fetch_pricing_from_api()
            if not api_pricing:
                raise RuntimeError("No Claude models found in API response")
                
            self._pricing_cache = api_pricing
            self._cache_timestamp = current_time
            if not self._quiet:
                print(f"Using pricing data from models.dev API ({len(api_pricing)} models)")
            return self._pricing_cache
            
        except Exception as e:
            if self._pricing_cache is not None:
                if not self._quiet:
                    print(f"Warning: Failed to refresh pricing data, using cached data: {e}")
                return self._pricing_cache
            else:
                raise RuntimeError(f"Failed to fetch initial pricing data: {e}")
        
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
        pricing_data = self._get_pricing()
        
        pricing = pricing_data.get(model)
        
        if pricing is None:
            # Try to find a default Claude model for fallback
            default_models = ['claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022']
            for default_model in default_models:
                if default_model in pricing_data:
                    pricing = pricing_data[default_model]
                    if not self._quiet:
                        print(f"Warning: No pricing found for {model}, using {default_model} pricing")
                    break
            
            if pricing is None:
                # Use any available Claude model as last resort
                available_models = [m for m in pricing_data.keys() if 'claude' in m.lower()]
                if available_models:
                    fallback_model = available_models[0]
                    pricing = pricing_data[fallback_model]
                    if not self._quiet:
                        print(f"Warning: No pricing found for {model}, using {fallback_model} pricing")
                else:
                    raise RuntimeError(f"No pricing data available for model {model} and no Claude models found in API data")
        
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
        
        if not self._quiet:
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
        
        # Calculate date range and cost averages
        sessions_with_time = [s for s in summaries if s.start_time]
        if sessions_with_time:
            earliest_date = min(s.start_time for s in sessions_with_time).date()
            latest_date = max(s.start_time for s in sessions_with_time).date()
            total_days = (latest_date - earliest_date).days + 1  # +1 to include both start and end dates
            
            # Calculate averages
            avg_daily_cost = total_cost / total_days if total_days > 0 else 0
            avg_weekly_cost = avg_daily_cost * 7
            avg_monthly_cost = avg_daily_cost * 30.44  # Average days per month (365.25/12)
            
            print(f"\nCost Averages:")
            print(f"  Date Range: {earliest_date} to {latest_date} ({total_days} days)")
            print(f"  Average Daily Cost: ${avg_daily_cost:.2f}")
            print(f"  Average Weekly Cost: ${avg_weekly_cost:.2f}")
            print(f"  Average Monthly Cost: ${avg_monthly_cost:.2f}")
            
            # Add disclaimer for limited data
            if total_days < 7:
                print(f"  ⚠️  Projections based on limited data ({total_days} days) - actual costs may vary significantly")
            elif total_days < 30:
                print(f"  ⚠️  Monthly projection based on {total_days} days - consider seasonal usage patterns")
            
            # Show actual breakdown by period if we have enough data
            daily_costs = defaultdict(float)
            monthly_costs = defaultdict(float)
            
            for summary in summaries:
                if summary.start_time:
                    date_key = summary.start_time.date()
                    month_key = summary.start_time.strftime('%Y-%m')
                    session_cost = self.calculate_cost(summary.token_usage, summary.model or 'claude-sonnet-4-20250514')
                    daily_costs[date_key] += session_cost
                    monthly_costs[month_key] += session_cost
            
            # Show daily breakdown if reasonable number of days
            if 1 < total_days <= 14:
                print(f"\nDaily Breakdown:")
                for date in sorted(daily_costs.keys()):
                    print(f"  {date}: ${daily_costs[date]:.2f}")
            
            # Show monthly breakdown if we have multiple months
            if len(monthly_costs) > 1:
                print(f"\nMonthly Breakdown:")
                for month in sorted(monthly_costs.keys()):
                    print(f"  {month}: ${monthly_costs[month]:.2f}")
                actual_monthly_avg = sum(monthly_costs.values()) / len(monthly_costs)
                print(f"  Actual Monthly Average: ${actual_monthly_avg:.2f}")
            
            # Show cost trend if we have multiple days
            if total_days > 1:
                sorted_dates = sorted(daily_costs.keys())
                first_day_cost = daily_costs[sorted_dates[0]]
                last_day_cost = daily_costs[sorted_dates[-1]]
                
                if first_day_cost > 0 and last_day_cost > 0:
                    trend_change = ((last_day_cost - first_day_cost) / first_day_cost) * 100
                    trend_direction = "📈 increasing" if trend_change > 10 else "📉 decreasing" if trend_change < -10 else "📊 stable"
                    print(f"\nCost Trend: {trend_direction} ({trend_change:+.1f}% from first to last day)")
        else:
            print(f"\nCost Averages: Unable to calculate (no sessions with timestamps)")
        
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
                               key=lambda x: x.start_time, reverse=True)[:self._recent_count]
        
        if recent_sessions:
            print(f"\nRecent Sessions (Last {min(self._recent_count, len(recent_sessions))}):")
            for session in recent_sessions:
                total_session_tokens = (session.token_usage.input_tokens + 
                                      session.token_usage.output_tokens + 
                                      session.token_usage.cache_creation_input_tokens + 
                                      session.token_usage.cache_read_input_tokens)
                cost = self.calculate_cost(session.token_usage, session.model or 'claude-sonnet-4-20250514')
                time_str = session.start_time.strftime('%Y-%m-%d %H:%M') if session.start_time else 'Unknown'
                print(f"  {time_str} - {session.project_path}: {total_session_tokens:,} tokens (${cost:.2f})")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze Claude Code session files and display token usage summary with cost estimates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Analyze default Claude directory
  %(prog)s --claude-dir /path/to/claude # Analyze custom Claude directory
  %(prog)s --recent 5                   # Show only last 5 sessions
  %(prog)s --project my-project         # Filter by specific project
  %(prog)s --model claude-sonnet-4      # Filter by specific model
        """
    )
    
    parser.add_argument(
        '--claude-dir',
        type=str,
        default=None,
        help='Path to Claude directory (default: ~/.claude)'
    )
    
    parser.add_argument(
        '--recent',
        type=int,
        default=10,
        help='Number of recent sessions to display (default: 10)'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='Filter sessions by project name'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Filter sessions by model name'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force refresh pricing data from API (ignore cache)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages and warnings'
    )
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        analyzer = ClaudeUsageAnalyzer(claude_dir=args.claude_dir, quiet=args.quiet)
        
        # Force cache refresh if requested
        if args.no_cache:
            analyzer._pricing_cache = None
            analyzer._cache_timestamp = None
        
        if not os.path.exists(analyzer.projects_dir):
            print(f"Error: Claude projects directory not found at {analyzer.projects_dir}")
            print("Make sure Claude Code has been used and session data exists.")
            return 1
        
        summaries = analyzer.analyze_all_sessions()
        
        # Apply filters
        if args.project:
            summaries = [s for s in summaries if args.project.lower() in s.project_path.lower()]
            if not summaries:
                print(f"No sessions found for project '{args.project}'")
                return 0
        
        if args.model:
            summaries = [s for s in summaries if args.model.lower() in s.model.lower()]
            if not summaries:
                print(f"No sessions found for model '{args.model}'")
                return 0
        
        # Update recent sessions count
        analyzer._recent_count = args.recent
        
        analyzer.print_summary(summaries)
        
    except RuntimeError as e:
        if not args.quiet:
            print(f"Error: {e}")
            print("Make sure you have internet connectivity to fetch pricing data from models.dev")
        return 1
    except Exception as e:
        if not args.quiet:
            print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    main()