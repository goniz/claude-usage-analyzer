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
    provider: str = ""
    cost: float = 0.0
    
    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = TokenUsage()


class ModelsDotDev:
    """Handles interaction with models.dev API for pricing data."""
    
    # Models.dev API endpoint
    API_URL = "https://models.dev/api.json"
    
    def __init__(self, quiet: bool = False):
        self._quiet = quiet
    
    def get_pricing(self) -> Dict:
        """Get pricing data from models.dev API."""
        try:
            response = requests.get(self.API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Transform API data to our pricing format with provider info
            pricing = {}
            
            # Iterate through all provider sections
            for provider_id, provider_data in data.items():
                if not isinstance(provider_data, dict):
                    continue
                    
                # Get provider display name, fallback to ID
                provider_name = provider_data.get('name', provider_id)
                models_data = provider_data.get('models', {})
                
                if isinstance(models_data, dict):
                    for model_id, model_data in models_data.items():
                        if not isinstance(model_data, dict):
                            continue
                            
                        cost_data = model_data.get('cost', {})
                        
                        if model_id and cost_data:
                            input_price = cost_data.get('input')
                            output_price = cost_data.get('output')
                            cache_write_price = cost_data.get('cache_write')
                            cache_read_price = cost_data.get('cache_read')
                            
                            if input_price is not None and output_price is not None:
                                pricing[model_id] = {
                                    'input': input_price,
                                    'output': output_price,
                                    'cache_write': cache_write_price if cache_write_price is not None else 0,
                                    'cache_read': cache_read_price if cache_read_price is not None else 0,
                                    'provider_id': provider_id,
                                    'provider_name': provider_name,
                                }
            
            if pricing and not self._quiet:
                print(f"Fetched pricing for {len(pricing)} models from API")
            return pricing
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch pricing from models.dev API: {e}")
    
    def get_providers_and_models(self) -> Dict:
        """Get organized provider and model data from models.dev API."""
        pricing_data = self.get_pricing()
        
        # Organize by provider
        providers = {}
        for model_id, model_info in pricing_data.items():
            provider_id = model_info['provider_id']
            provider_name = model_info['provider_name']
            
            if provider_id not in providers:
                providers[provider_id] = {
                    'name': provider_name,
                    'models': [],
                    'count': 0
                }
            
            providers[provider_id]['models'].append({
                'id': model_id,
                'pricing': {
                    'input': model_info['input'],
                    'output': model_info['output'],
                    'cache_read': model_info['cache_read'],
                    'cache_write': model_info['cache_write'],
                }
            })
            providers[provider_id]['count'] += 1
        
        return providers


class ClaudeUsageAnalyzer:
    def __init__(self, claude_dir: str = None, quiet: bool = False):
        self.claude_dir = claude_dir or os.path.expanduser('~/.claude')
        self.projects_dir = os.path.join(self.claude_dir, 'projects')
        self._quiet = quiet
        self._recent_count = 10
        self.models_api = ModelsDotDev(quiet=quiet)
    
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
                            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
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
                                model_name = data['message']['model']
                                # Add provider prefix for Claude Code models (always Anthropic)
                                summary.model = f"anthropic/{model_name}"
                                
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in {filepath} line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
        return summary
    
    def calculate_cost(self, token_usage: TokenUsage, model: str) -> float:
        """Calculate estimated cost based on token usage and model."""
        pricing_data = self.models_api.get_pricing()
        
        # Extract model name without provider prefix for pricing lookup
        model_for_pricing = model
        if '/' in model:
            model_for_pricing = model.split('/', 1)[1]
        
        pricing = pricing_data.get(model_for_pricing)
        
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
            session_cost = self.calculate_cost(usage, summary.model or 'anthropic/claude-sonnet-4-20250514')
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
                    session_cost = self.calculate_cost(summary.token_usage, summary.model or 'anthropic/claude-sonnet-4-20250514')
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
                cost = self.calculate_cost(session.token_usage, session.model or 'anthropic/claude-sonnet-4-20250514')
                time_str = session.start_time.strftime('%Y-%m-%d %H:%M') if session.start_time else 'Unknown'
                print(f"  {time_str} - {session.project_path}: {total_session_tokens:,} tokens (${cost:.2f})")


class OpenCodeUsageAnalyzer:
    def __init__(self, opencode_dir: str = None, quiet: bool = False):
        self.opencode_dir = opencode_dir or os.path.expanduser('~/.local/share/opencode')
        self.projects_dir = os.path.join(self.opencode_dir, 'project')
        self._quiet = quiet
        self._recent_count = 10
    
    def find_session_projects(self) -> List[str]:
        """Find all project directories containing session data."""
        if not os.path.exists(self.projects_dir):
            return []
        
        projects = []
        for item in os.listdir(self.projects_dir):
            project_path = os.path.join(self.projects_dir, item)
            if os.path.isdir(project_path):
                session_info_dir = os.path.join(project_path, 'storage', 'session', 'info')
                if os.path.exists(session_info_dir):
                    projects.append(project_path)
        return projects
    
    def find_session_files(self, project_path: str) -> List[str]:
        """Find all session info files in a project directory."""
        session_info_dir = os.path.join(project_path, 'storage', 'session', 'info')
        if not os.path.exists(session_info_dir):
            return []
        
        session_files = []
        for filename in os.listdir(session_info_dir):
            if filename.startswith('ses_') and filename.endswith('.json'):
                session_files.append(os.path.join(session_info_dir, filename))
        return session_files
    
    def parse_session_file(self, info_filepath: str, project_path: str) -> SessionSummary:
        """Parse a single OpenCode session and extract usage information."""
        session_id = os.path.basename(info_filepath).replace('.json', '')
        project_name = os.path.basename(project_path)
        
        summary = SessionSummary(
            session_id=session_id,
            project_path=project_name
        )
        
        try:
            # Parse session info file
            with open(info_filepath, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
            
            # Extract timestamps from info
            if 'time' in info_data:
                time_data = info_data['time']
                if 'created' in time_data:
                    summary.start_time = datetime.fromtimestamp(time_data['created'] / 1000).replace(tzinfo=None)
                if 'updated' in time_data:
                    summary.end_time = datetime.fromtimestamp(time_data['updated'] / 1000).replace(tzinfo=None)
            
            # Find message files for this session
            message_dir = os.path.join(project_path, 'storage', 'session', 'message', session_id)
            if os.path.exists(message_dir):
                for msg_filename in os.listdir(message_dir):
                    if msg_filename.startswith('msg_') and msg_filename.endswith('.json'):
                        msg_filepath = os.path.join(message_dir, msg_filename)
                        try:
                            with open(msg_filepath, 'r', encoding='utf-8') as f:
                                msg_data = json.load(f)
                            
                            summary.messages_count += 1
                            
                            # Extract usage information from assistant messages
                            if msg_data.get('role') == 'assistant':
                                # Extract token usage
                                if 'tokens' in msg_data:
                                    tokens = msg_data['tokens']
                                    summary.token_usage.input_tokens += tokens.get('input', 0)
                                    summary.token_usage.output_tokens += tokens.get('output', 0)
                                    # Note: OpenCode uses different field names for cache tokens
                                    cache_data = tokens.get('cache', {})
                                    summary.token_usage.cache_creation_input_tokens += cache_data.get('write', 0)
                                    summary.token_usage.cache_read_input_tokens += cache_data.get('read', 0)
                                
                                # Extract cost information (OpenCode provides cost directly)
                                if 'cost' in msg_data:
                                    summary.cost += msg_data['cost']
                                
                                # Extract model and provider information
                                if 'modelID' in msg_data and not summary.model:
                                    model_name = msg_data['modelID']
                                    provider_name = msg_data.get('providerID', 'unknown')
                                    # Format with provider prefix
                                    summary.model = f"{provider_name}/{model_name}"
                                if 'providerID' in msg_data and not summary.provider:
                                    summary.provider = msg_data['providerID']
                                
                                # Update timestamps from message if available
                                if 'time' in msg_data:
                                    time_data = msg_data['time']
                                    if 'created' in time_data:
                                        msg_time = datetime.fromtimestamp(time_data['created'] / 1000).replace(tzinfo=None)
                                        if summary.start_time is None or msg_time < summary.start_time:
                                            summary.start_time = msg_time
                                    if 'completed' in time_data:
                                        msg_time = datetime.fromtimestamp(time_data['completed'] / 1000).replace(tzinfo=None)
                                        if summary.end_time is None or msg_time > summary.end_time:
                                            summary.end_time = msg_time
                        
                        except (json.JSONDecodeError, KeyError) as e:
                            if not self._quiet:
                                print(f"Warning: Error parsing message file {msg_filepath}: {e}")
                            continue
                        except Exception as e:
                            if not self._quiet:
                                print(f"Warning: Unexpected error reading {msg_filepath}: {e}")
                            continue
        
        except Exception as e:
            if not self._quiet:
                print(f"Error reading session info {info_filepath}: {e}")
        
        # Ensure OpenCode sessions are marked as such (set default provider if none found)
        if not summary.provider:
            summary.provider = "unknown"
        
        return summary
    
    def analyze_all_sessions(self) -> List[SessionSummary]:
        """Analyze all OpenCode session files and return summaries."""
        project_paths = self.find_session_projects()
        summaries = []
        
        if not self._quiet:
            print(f"Found {len(project_paths)} OpenCode projects to analyze...")
        
        for project_path in project_paths:
            session_files = self.find_session_files(project_path)
            if not self._quiet and session_files:
                project_name = os.path.basename(project_path)
                print(f"  {project_name}: {len(session_files)} sessions")
            
            for filepath in session_files:
                summary = self.parse_session_file(filepath, project_path)
                summaries.append(summary)
        
        return summaries
    
    def print_summary(self, summaries: List[SessionSummary]):
        """Print a comprehensive summary of OpenCode token usage and costs."""
        if not summaries:
            print("No OpenCode session data found.")
            return
            
        # Overall statistics
        total_sessions = len(summaries)
        total_messages = sum(s.messages_count for s in summaries)
        
        # Aggregate token usage
        total_tokens = TokenUsage()
        model_usage = defaultdict(TokenUsage)
        provider_usage = defaultdict(TokenUsage)
        project_usage = defaultdict(TokenUsage)
        total_cost = sum(s.cost for s in summaries)
        model_costs = defaultdict(float)
        provider_costs = defaultdict(float)
        
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
            model_costs[model] += summary.cost
            
            # Group by provider
            provider = summary.provider or 'unknown'
            provider_usage[provider].input_tokens += usage.input_tokens
            provider_usage[provider].output_tokens += usage.output_tokens
            provider_usage[provider].cache_creation_input_tokens += usage.cache_creation_input_tokens
            provider_usage[provider].cache_read_input_tokens += usage.cache_read_input_tokens
            provider_costs[provider] += summary.cost
            
            # Group by project
            project_usage[summary.project_path].input_tokens += usage.input_tokens
            project_usage[summary.project_path].output_tokens += usage.output_tokens
            project_usage[summary.project_path].cache_creation_input_tokens += usage.cache_creation_input_tokens
            project_usage[summary.project_path].cache_read_input_tokens += usage.cache_read_input_tokens
        
        # Print results
        print("\n" + "=" * 80)
        print("OPENCODE USAGE SUMMARY")
        print("=" * 80)
        
        print(f"\nOverall Statistics:")
        print(f"  Total Sessions: {total_sessions:,}")
        print(f"  Total Messages: {total_messages:,}")
        
        print(f"\nTotal Token Usage:")
        print(f"  Input Tokens:              {total_tokens.input_tokens:,}")
        print(f"  Output Tokens:             {total_tokens.output_tokens:,}")
        print(f"  Cache Creation Tokens:     {total_tokens.cache_creation_input_tokens:,}")
        print(f"  Cache Read Tokens:         {total_tokens.cache_read_input_tokens:,}")
        total_all_tokens = total_tokens.input_tokens + total_tokens.output_tokens + total_tokens.cache_creation_input_tokens + total_tokens.cache_read_input_tokens
        print(f"  Total Tokens:              {total_all_tokens:,}")
        
        print(f"\nTotal Cost: ${total_cost:.4f}")
        
        # Calculate date range and cost averages
        sessions_with_time = [s for s in summaries if s.start_time]
        if sessions_with_time:
            earliest_date = min(s.start_time for s in sessions_with_time).date()
            latest_date = max(s.start_time for s in sessions_with_time).date()
            total_days = (latest_date - earliest_date).days + 1
            
            avg_daily_cost = total_cost / total_days if total_days > 0 else 0
            avg_weekly_cost = avg_daily_cost * 7
            avg_monthly_cost = avg_daily_cost * 30.44
            
            print(f"\nCost Averages:")
            print(f"  Date Range: {earliest_date} to {latest_date} ({total_days} days)")
            print(f"  Average Daily Cost: ${avg_daily_cost:.4f}")
            print(f"  Average Weekly Cost: ${avg_weekly_cost:.2f}")
            print(f"  Average Monthly Cost: ${avg_monthly_cost:.2f}")
        
        # Provider breakdown
        if len(provider_usage) > 1:
            print(f"\nUsage by Provider:")
            for provider, usage in sorted(provider_usage.items()):
                cost = provider_costs[provider]
                print(f"  {provider}:")
                print(f"    Input: {usage.input_tokens:,}, Output: {usage.output_tokens:,}")
                print(f"    Cache Write: {usage.cache_creation_input_tokens:,}, Cache Read: {usage.cache_read_input_tokens:,}")
                print(f"    Total Cost: ${cost:.4f}")
        
        # Model breakdown
        if len(model_usage) > 1:
            print(f"\nUsage by Model:")
            for model, usage in sorted(model_usage.items()):
                cost = model_costs[model]
                print(f"  {model}:")
                print(f"    Input: {usage.input_tokens:,}, Output: {usage.output_tokens:,}")
                print(f"    Cache Write: {usage.cache_creation_input_tokens:,}, Cache Read: {usage.cache_read_input_tokens:,}")
                print(f"    Total Cost: ${cost:.4f}")
        
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
                time_str = session.start_time.strftime('%Y-%m-%d %H:%M') if session.start_time else 'Unknown'
                print(f"  {time_str} - {session.project_path}: {total_session_tokens:,} tokens (${session.cost:.4f})")


def list_providers():
    """List all available providers from the models.dev API."""
    try:
        # Create a temporary ModelsDotDev instance to fetch provider data
        models_api = ModelsDotDev(quiet=True)
        providers = models_api.get_providers_and_models()
        
        if not providers:
            print("❌ Unable to fetch provider data from models.dev API")
            return 1
        
        print("🌐 Available AI Providers")
        print("=" * 60)
        
        # Sort providers by model count (descending)
        sorted_providers = sorted(providers.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for provider_id, provider_data in sorted_providers:
            count = provider_data['count']
            name = provider_data['name']
            print(f"  {provider_id:<15} {count:3d} models - {name}")
        
        print(f"\n💡 Usage:")
        print(f"  uv run main.py --list-models <provider>")
        print(f"  uv run main.py --compare-model <provider/model-name>")
        print(f"\n📊 Examples:")
        # Show top 3 providers as examples
        top_providers = [provider_id for provider_id, _ in sorted_providers[:3]]
        for provider_id in top_providers:
            print(f"  uv run main.py --list-models {provider_id}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error fetching provider data: {e}")
        print("💡 Make sure you have internet connectivity to access models.dev API")
        return 1


def list_available_models(provider=None):
    """List models for a specific provider from the models.dev API."""
    if not provider:
        print("❌ Provider argument is required")
        print("💡 Use --list-providers to see available providers")
        print("📊 Example: uv run main.py --list-models anthropic")
        return 1
        
    try:
        # Create a temporary ModelsDotDev instance to fetch provider data
        models_api = ModelsDotDev(quiet=True)
        providers = models_api.get_providers_and_models()
        
        if not providers:
            print("❌ Unable to fetch provider data from models.dev API")
            return 1
        
        # Validate provider argument
        if provider not in providers:
            print(f"❌ Unknown provider: {provider}")
            print(f"💡 Use --list-providers to see available providers")
            return 1
            
        # Get provider data
        provider_data = providers[provider]
        models = provider_data['models']
        provider_name = provider_data['name']
        
        if not models:
            print(f"❌ No models found for provider: {provider}")
            return 1
        
        print(f"🤖 {provider.upper()} Models ({len(models)} total)")
        print(f"{provider_name}")
        print("=" * 70)
        
        # Display models for the specified provider
        for model in models:
            model_id = model['id']
            pricing = model['pricing']
            
            # Format with provider prefix for consistency
            model_with_prefix = f"{provider}/{model_id}"
            
            # Compact pricing format
            price_str = f"${pricing['input']:.3f}→${pricing['output']:.3f}/M"
            if pricing.get('cache_read', 0) > 0 or pricing.get('cache_write', 0) > 0:
                price_str += f" (cache: ${pricing.get('cache_read', 0):.3f}→${pricing.get('cache_write', 0):.3f})"
            
            print(f"  {model_with_prefix:<50} {price_str}")
        
        print(f"\n💡 Usage: uv run main.py --compare-model <model-name>")
        print(f"📊 Examples for {provider} models:")
        
        # Show a few example models for comparison
        example_models = models[:3]  # First 3 models as examples
        for model in example_models:
            model_with_prefix = f"{provider}/{model['id']}"
            print(f"  uv run main.py --compare-model {model_with_prefix}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error fetching model data: {e}")
        print("💡 Make sure you have internet connectivity to access models.dev API")
        return 1


def calculate_alternative_model_cost(summaries: List[SessionSummary], alternative_model: str, claude_analyzer=None) -> float:
    """Calculate what the cost would have been using an alternative model (Claude or non-Claude)."""
    if not claude_analyzer:
        return 0.0
    
    try:
        # Check if the specific model is available in pricing data
        pricing_data = claude_analyzer.models_api.get_pricing()
        
        # Extract model name without provider prefix for pricing lookup
        model_for_pricing = alternative_model
        if '/' in alternative_model:
            model_for_pricing = alternative_model.split('/', 1)[1]
        
        if model_for_pricing not in pricing_data:
            return 0.0  # Model not found
        
        # Aggregate total token usage across all sessions
        total_usage = TokenUsage()
        for summary in summaries:
            usage = summary.token_usage
            total_usage.input_tokens += usage.input_tokens
            total_usage.output_tokens += usage.output_tokens
            total_usage.cache_creation_input_tokens += usage.cache_creation_input_tokens
            total_usage.cache_read_input_tokens += usage.cache_read_input_tokens
        
        # Calculate cost using the specific model pricing
        pricing = pricing_data[model_for_pricing]
        cost = 0.0
        
        # Basic input/output tokens (all models have these)
        cost += (total_usage.input_tokens / 1_000_000) * pricing['input']
        cost += (total_usage.output_tokens / 1_000_000) * pricing['output']
        
        # Cache tokens (only Claude models typically have cache pricing)
        if 'cache_write' in pricing and 'cache_read' in pricing:
            cost += (total_usage.cache_creation_input_tokens / 1_000_000) * pricing['cache_write']
            cost += (total_usage.cache_read_input_tokens / 1_000_000) * pricing['cache_read']
        else:
            # For non-Claude models, treat cache tokens as regular input tokens
            # since they don't have cache-specific pricing
            cost += (total_usage.cache_creation_input_tokens / 1_000_000) * pricing['input']
            cost += (total_usage.cache_read_input_tokens / 1_000_000) * pricing['input']
        
        return cost
    except Exception as e:
        return 0.0


def print_unified_summary(summaries: List[SessionSummary], recent_count: int = 10, claude_analyzer=None, compare_model: str = None):
    """Print a clean, unified summary of both Claude Code and OpenCode sessions."""
    if not summaries:
        print("No session data found.")
        return
    
    # Separate by source (based on the actual filtered summaries)
    claude_summaries = [s for s in summaries if not s.provider]
    opencode_summaries = [s for s in summaries if s.provider]
    
    # Debug: check provider field values
    # print(f"Debug: Total summaries: {len(summaries)}")
    # print(f"Debug: Sessions without provider: {len(claude_summaries)}")  
    # print(f"Debug: Sessions with provider: {len(opencode_summaries)}")
    # for s in summaries[:5]:
    #     print(f"Debug: session {s.session_id}: provider='{s.provider}', model='{s.model}'")  
    
    # Calculate stats
    total_sessions = len(summaries)
    total_messages = sum(s.messages_count for s in summaries)
    
    # Aggregate token usage
    total_tokens = TokenUsage()
    for summary in summaries:
        usage = summary.token_usage
        total_tokens.input_tokens += usage.input_tokens
        total_tokens.output_tokens += usage.output_tokens
        total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
        total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens
    
    total_all_tokens = (total_tokens.input_tokens + total_tokens.output_tokens + 
                       total_tokens.cache_creation_input_tokens + total_tokens.cache_read_input_tokens)
    
    # Calculate costs
    opencode_cost = sum(s.cost for s in opencode_summaries)
    claude_cost = 0.0
    if claude_summaries and claude_analyzer:
        try:
            claude_total_usage = TokenUsage()
            for s in claude_summaries:
                claude_total_usage.input_tokens += s.token_usage.input_tokens
                claude_total_usage.output_tokens += s.token_usage.output_tokens
                claude_total_usage.cache_creation_input_tokens += s.token_usage.cache_creation_input_tokens
                claude_total_usage.cache_read_input_tokens += s.token_usage.cache_read_input_tokens
            claude_cost = claude_analyzer.calculate_cost(claude_total_usage, 'anthropic/claude-sonnet-4-20250514')
        except:
            claude_cost = 0.0
    
    total_cost = claude_cost + opencode_cost
    
    # Clean project names
    def clean_project_name(project_path):
        if project_path.startswith('-home-'):
            return project_path.replace('-home-goniz-dev-', '').replace('-home-goniz-', '').replace('-root-', '')
        return project_path
    
    # Print clean summary
    print(f"\n📊 AI Usage Summary")
    print(f"━" * 50)
    
    # Basic stats
    if len(claude_summaries) > 0 and len(opencode_summaries) > 0:
        print(f"💬 {total_sessions:,} sessions ({len(claude_summaries):,} Claude Code, {len(opencode_summaries):,} OpenCode)")
    elif len(claude_summaries) > 0:
        print(f"💬 {total_sessions:,} Claude Code sessions")
    elif len(opencode_summaries) > 0:
        print(f"💬 {total_sessions:,} OpenCode sessions")
    else:
        print(f"💬 {total_sessions:,} sessions")
    
    print(f"📝 {total_messages:,} messages")
    print(f"🔤 {total_all_tokens:,} tokens total")
    
    if total_cost > 0:
        print(f"💰 ${total_cost:.2f} estimated cost")
        if claude_cost > 0 and opencode_cost > 0:
            print(f"   └─ Claude Code: ${claude_cost:.2f}, OpenCode: ${opencode_cost:.4f}")
        
        # Cost comparison with alternative model if requested
        if compare_model and claude_analyzer and claude_summaries:
            try:
                alt_cost = calculate_alternative_model_cost(claude_summaries, compare_model, claude_analyzer)
                alt_total_cost = alt_cost + opencode_cost
                
                if alt_cost > 0 and abs(alt_cost - claude_cost) > 0.001:  # Avoid floating point comparison issues
                    cost_diff = claude_cost - alt_cost
                    percentage_diff = (cost_diff / claude_cost) * 100 if claude_cost > 0 else 0
                    
                    # Clean up model name for display
                    clean_model_name = compare_model
                    model_lower = compare_model.lower()
                    
                    # Claude models
                    if 'claude' in model_lower:
                        if 'haiku' in model_lower:
                            clean_model_name = 'Claude 3.5 Haiku'
                        elif 'sonnet' in model_lower:
                            if '4' in model_lower:
                                clean_model_name = 'Claude Sonnet 4'
                            else:
                                clean_model_name = 'Claude 3.5 Sonnet'
                        elif 'opus' in model_lower:
                            clean_model_name = 'Claude 3 Opus'
                    
                    # OpenAI models
                    elif 'gpt' in model_lower or 'o1' in model_lower:
                        if 'gpt-4o' in model_lower:
                            clean_model_name = 'GPT-4 Omni'
                        elif 'gpt-4' in model_lower:
                            clean_model_name = 'GPT-4'
                        elif 'gpt-3.5' in model_lower:
                            clean_model_name = 'GPT-3.5'
                        elif 'o1-preview' in model_lower:
                            clean_model_name = 'GPT-o1 Preview'
                        elif 'o1-mini' in model_lower:
                            clean_model_name = 'GPT-o1 Mini'
                    
                    # Groq models
                    elif 'llama' in model_lower:
                        if '70b' in model_lower:
                            clean_model_name = 'Llama 3 70B (Groq)'
                        elif '8b' in model_lower:
                            clean_model_name = 'Llama 3 8B (Groq)'
                        else:
                            clean_model_name = 'Llama (Groq)'
                    elif 'mixtral' in model_lower:
                        clean_model_name = 'Mixtral 8x7B (Groq)'
                    elif 'gemma' in model_lower:
                        clean_model_name = 'Gemma 2 (Groq)'
                    
                    # Show comparison with appropriate emoji based on model type
                    model_emoji = "📊"
                    if 'gpt' in model_lower or 'o1' in model_lower:
                        model_emoji = "🔵"
                    elif 'groq' in model_lower or 'llama' in model_lower or 'mixtral' in model_lower:
                        model_emoji = "⚡"
                    elif 'claude' in model_lower:
                        model_emoji = "🟣"
                    
                    if cost_diff > 0:
                        print(f"{model_emoji} Alternative: {clean_model_name} would cost ${alt_total_cost:.2f} (save ${cost_diff:.2f}, -{percentage_diff:.1f}%)")
                    elif cost_diff < 0:
                        print(f"{model_emoji} Alternative: {clean_model_name} would cost ${alt_total_cost:.2f} (cost ${-cost_diff:.2f} more, +{-percentage_diff:.1f}%)")
                elif alt_cost > 0:
                    print(f"📊 Alternative: {clean_model_name} would cost ${alt_total_cost:.2f} (same cost)")
                else:
                    print(f"⚠️  Could not calculate cost for '{compare_model}' (model not found in pricing data)")
            except Exception as e:
                print(f"⚠️  Error comparing with {compare_model}: {str(e)}")
    
    # Token breakdown (simplified)
    if total_tokens.cache_read_input_tokens > 0 or total_tokens.cache_creation_input_tokens > 0:
        print(f"\n📈 Token breakdown:")
        print(f"   Input: {total_tokens.input_tokens:,} | Output: {total_tokens.output_tokens:,}")
        if total_tokens.cache_read_input_tokens > 0:
            print(f"   Cache: {total_tokens.cache_read_input_tokens:,} read, {total_tokens.cache_creation_input_tokens:,} created")
    
    # Top projects (simplified)
    project_usage = defaultdict(int)
    for summary in summaries:
        clean_name = clean_project_name(summary.project_path)
        usage = summary.token_usage
        project_tokens = usage.input_tokens + usage.output_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        project_usage[clean_name] += project_tokens
    
    if len(project_usage) > 1:
        print(f"\n🗂️  Top projects:")
        sorted_projects = sorted(project_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        for project, tokens in sorted_projects:
            percentage = (tokens / total_all_tokens) * 100 if total_all_tokens > 0 else 0
            print(f"   {project}: {tokens:,} tokens ({percentage:.1f}%)")
    
    # Model breakdown (if multiple models)
    model_usage = defaultdict(int)
    for summary in summaries:
        model = summary.model or 'unknown'
        if model != 'unknown':
            # Clean up model names for display
            if 'claude' in model.lower():
                model = 'Claude ' + model.split('-')[-1] if '-' in model else 'Claude'
            elif 'gpt' in model.lower():
                if '120b' in model:
                    model = 'GPT-4 (120B)'
                elif '20b' in model:
                    model = 'GPT-4 (20B)'
                else:
                    model = 'GPT-4'
        
        usage = summary.token_usage
        tokens = usage.input_tokens + usage.output_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        model_usage[model] += tokens
    
    # Only show model breakdown if there are multiple models
    valid_models = {k: v for k, v in model_usage.items() if k != 'unknown' and v > 0}
    if len(valid_models) > 1:
        print(f"\n🤖 Models used:")
        sorted_models = sorted(valid_models.items(), key=lambda x: x[1], reverse=True)
        for model, tokens in sorted_models:
            percentage = (tokens / total_all_tokens) * 100 if total_all_tokens > 0 else 0
            print(f"   {model}: {tokens:,} tokens ({percentage:.1f}%)")
    
    # Recent activity (simplified)
    recent_sessions = sorted([s for s in summaries if s.start_time], 
                           key=lambda x: x.start_time, reverse=True)[:recent_count]
    
    if recent_sessions:
        print(f"\n⏱️  Recent activity:")
        for session in recent_sessions:
            total_session_tokens = (session.token_usage.input_tokens + 
                                  session.token_usage.output_tokens + 
                                  session.token_usage.cache_creation_input_tokens + 
                                  session.token_usage.cache_read_input_tokens)
            
            time_str = session.start_time.strftime('%m/%d %H:%M') if session.start_time else 'Unknown'
            clean_project = clean_project_name(session.project_path)
            source_icon = "🔵" if session.provider else "🟣"
            
            if total_session_tokens >= 1_000_000:
                token_str = f"{total_session_tokens/1_000_000:.1f}M"
            elif total_session_tokens >= 1_000:
                token_str = f"{total_session_tokens/1_000:.1f}K"
            else:
                token_str = str(total_session_tokens)
            
            print(f"   {source_icon} {time_str} {clean_project} ({token_str} tokens)")
    
    print()


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze Claude Code and OpenCode session files and display token usage summary with cost estimates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Clean unified summary (both platforms)
  %(prog)s --claude-only                # Analyze only Claude Code sessions
  %(prog)s --opencode-only              # Analyze only OpenCode sessions
  %(prog)s --verbose                    # Show detailed platform breakdowns
  %(prog)s --recent 5                   # Show only last 5 recent sessions
  %(prog)s --project my-project         # Filter by specific project
  %(prog)s --model claude-sonnet-4      # Filter by specific model
  %(prog)s --compare-model anthropic/claude-3-5-haiku-20241022  # Compare costs with alternative model
  %(prog)s --list-providers             # List all available providers
  %(prog)s --list-models anthropic      # List all models for anthropic provider
        """
    )
    
    parser.add_argument(
        '--claude-dir',
        type=str,
        default=None,
        help='Path to Claude directory (default: ~/.claude)'
    )
    
    parser.add_argument(
        '--opencode-dir',
        type=str,
        default=None,
        help='Path to OpenCode directory (default: ~/.local/share/opencode)'
    )
    
    parser.add_argument(
        '--claude-only',
        action='store_true',
        help='Analyze only Claude Code sessions'
    )
    
    parser.add_argument(
        '--opencode-only',
        action='store_true',
        help='Analyze only OpenCode sessions'
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
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed breakdowns for each platform (default: unified summary only)'
    )
    
    parser.add_argument(
        '--compare-model',
        type=str,
        default=None,
        help='Compare costs with an alternative model (e.g., claude-3-5-haiku-20241022)'
    )
    
    parser.add_argument(
        '--list-providers',
        action='store_true',
        help='List all available providers from models.dev API and exit'
    )
    
    parser.add_argument(
        '--list-models',
        type=str,
        metavar='PROVIDER',
        help='List models for a specific provider (e.g., anthropic, openai, groq) and exit'
    )
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle list-providers flag and exit early
    if args.list_providers:
        return list_providers()
    
    # Handle list-models flag and exit early
    if args.list_models:
        return list_available_models(args.list_models)
    
    # Validate mutually exclusive options
    if args.claude_only and args.opencode_only:
        print("Error: --claude-only and --opencode-only are mutually exclusive")
        return 1
    
    try:
        claude_summaries = []
        opencode_summaries = []
        
        # Analyze Claude Code sessions unless --opencode-only is specified
        if not args.opencode_only:
            try:
                # Make it quiet by default unless verbose is requested
                claude_quiet = args.quiet or not args.verbose
                claude_analyzer = ClaudeUsageAnalyzer(claude_dir=args.claude_dir, quiet=claude_quiet)
                
                # Note: --no-cache flag is no longer needed as caching was removed
                
                if os.path.exists(claude_analyzer.projects_dir):
                    claude_summaries = claude_analyzer.analyze_all_sessions()
                    claude_analyzer._recent_count = args.recent
                elif args.verbose:
                    print(f"Claude Code projects directory not found at {claude_analyzer.projects_dir}")
            except Exception as e:
                if not args.quiet:
                    print(f"Error analyzing Claude Code sessions: {e}")
                if args.claude_only:
                    return 1
        
        # Analyze OpenCode sessions unless --claude-only is specified
        if not args.claude_only:
            try:
                # Make it quiet by default unless verbose is requested
                opencode_quiet = args.quiet or not args.verbose
                opencode_analyzer = OpenCodeUsageAnalyzer(opencode_dir=args.opencode_dir, quiet=opencode_quiet)
                
                if os.path.exists(opencode_analyzer.projects_dir):
                    opencode_summaries = opencode_analyzer.analyze_all_sessions()
                    opencode_analyzer._recent_count = args.recent
                elif args.verbose:
                    print(f"OpenCode projects directory not found at {opencode_analyzer.projects_dir}")
            except Exception as e:
                if not args.quiet:
                    print(f"Error analyzing OpenCode sessions: {e}")
                if args.opencode_only:
                    return 1
        
        # Apply filters to appropriate summaries based on flags
        all_summaries = []
        if not args.opencode_only:
            all_summaries.extend(claude_summaries)
        if not args.claude_only:
            all_summaries.extend(opencode_summaries)
        
        if args.project:
            all_summaries = [s for s in all_summaries if args.project.lower() in s.project_path.lower()]
            if not all_summaries:
                print(f"No sessions found for project '{args.project}'")
                return 0
        
        if args.model:
            all_summaries = [s for s in all_summaries if args.model.lower() in s.model.lower()]
            if not all_summaries:
                print(f"No sessions found for model '{args.model}'")
                return 0
        
        # Print unified summary
        if not all_summaries:
            print("No session data found.")
            return 0
        
        # Always use the unified summary for a cleaner experience
        analyzer_for_cost = None
        if claude_summaries and not args.opencode_only:
            analyzer_for_cost = claude_analyzer
        
        print_unified_summary(all_summaries, args.recent, analyzer_for_cost, args.compare_model)
        
        # Add verbose option for detailed breakdowns if requested
        if hasattr(args, 'verbose') and args.verbose:
            if claude_summaries and not args.opencode_only:
                filtered_claude = [s for s in claude_summaries if s in all_summaries]
                if filtered_claude:
                    claude_analyzer._recent_count = args.recent
                    claude_analyzer.print_summary(filtered_claude)
            
            if opencode_summaries and not args.claude_only:
                filtered_opencode = [s for s in opencode_summaries if s in all_summaries]
                if filtered_opencode:
                    opencode_analyzer._recent_count = args.recent
                    opencode_analyzer.print_summary(filtered_opencode)
        
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