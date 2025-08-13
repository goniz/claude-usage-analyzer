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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast


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
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    model: str = ""
    provider: str = ""
    cost: float = 0.0
    
    def __post_init__(self) -> None:
        # token_usage is guaranteed by default_factory
        return None


class ModelsDotDev:
    """Handles interaction with models.dev API for pricing data."""
    
    # Models.dev API endpoint
    API_URL = "https://models.dev/api.json"
    
    def __init__(self, quiet: bool = False):
        self._quiet = quiet
        self._pricing_cache: Optional[Dict[str, Dict[str, Any]]] = None
    
    def get_pricing(self) -> Dict[str, Dict[str, Any]]:
        """Get pricing data from models.dev API."""
        # Return cached data if available
        if self._pricing_cache is not None:
            return self._pricing_cache
            
        try:
            response = requests.get(self.API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Transform API data to our pricing format with provider info
            pricing: Dict[str, Dict[str, Any]] = {}
            
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
            
            # Cache the result
            self._pricing_cache = pricing
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
    
    def calculate_cost(self, token_usage: TokenUsage, model: str) -> float:
        """Calculate estimated cost based on token usage and model."""
        pricing_data = self.get_pricing()
        
        pricing: Optional[Dict[str, Any]] = pricing_data.get(model)
        
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
        
        assert pricing is not None
        p_in = cast(float, pricing['input'])
        p_out = cast(float, pricing['output'])
        p_cache_w = cast(float, pricing.get('cache_write', 0.0))
        p_cache_r = cast(float, pricing.get('cache_read', 0.0))
        cost = 0.0
        cost += (token_usage.input_tokens / 1_000_000) * p_in
        cost += (token_usage.output_tokens / 1_000_000) * p_out
        cost += (token_usage.cache_creation_input_tokens / 1_000_000) * p_cache_w
        cost += (token_usage.cache_read_input_tokens / 1_000_000) * p_cache_r
        
        return cost


class ClaudeUsageAnalyzer:
    def __init__(self, claude_dir: Optional[str] = None, quiet: bool = False):
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
        pricing_data: Dict[str, Dict[str, Any]] = self.models_api.get_pricing()
        
        # Extract model name without provider prefix for pricing lookup
        model_for_pricing = model
        if '/' in model:
            model_for_pricing = model.split('/', 1)[1]
        
        pricing: Optional[Dict[str, Any]] = pricing_data.get(model_for_pricing)
        
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
        
        assert pricing is not None
        p_in = cast(float, pricing['input'])
        p_out = cast(float, pricing['output'])
        p_cache_w = cast(float, pricing.get('cache_write', 0.0))
        p_cache_r = cast(float, pricing.get('cache_read', 0.0))
        cost = 0.0
        cost += (token_usage.input_tokens / 1_000_000) * p_in
        cost += (token_usage.output_tokens / 1_000_000) * p_out
        cost += (token_usage.cache_creation_input_tokens / 1_000_000) * p_cache_w
        cost += (token_usage.cache_read_input_tokens / 1_000_000) * p_cache_r
        
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
    
    def print_summary(self, summaries: List[SessionSummary]) -> None:
        """Print a comprehensive summary of token usage and costs."""
        if not summaries:
            print("No session data found.")
            return
            
        # Overall statistics
        total_sessions = len(summaries)
        total_messages = sum(s.messages_count for s in summaries)
        
        # Aggregate token usage
        total_tokens = TokenUsage()
        model_usage: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        project_usage: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        total_cost = 0.0
        model_costs: Dict[str, float] = defaultdict(float)
        
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
            project_name = os.path.basename(summary.project_path)
            project_usage[project_name].input_tokens += usage.input_tokens
            project_usage[project_name].output_tokens += usage.output_tokens
            project_usage[project_name].cache_creation_input_tokens += usage.cache_creation_input_tokens
            project_usage[project_name].cache_read_input_tokens += usage.cache_read_input_tokens
            
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
        
        # Calculate costs per token type
        pricing_data = self.models_api.get_pricing()
        default_model = 'claude-sonnet-4-20250514'
        pricing = pricing_data.get(default_model)
        if pricing is None:
            # Find any available Claude model for pricing
            available_models = [m for m in pricing_data.keys() if 'claude' in m.lower()]
            if available_models:
                default_model = available_models[0]
                pricing = pricing_data[default_model]
        
        if pricing:
            input_cost = (total_tokens.input_tokens / 1_000_000) * pricing['input']
            output_cost = (total_tokens.output_tokens / 1_000_000) * pricing['output']
            cache_creation_cost = (total_tokens.cache_creation_input_tokens / 1_000_000) * pricing['cache_write']
            cache_read_cost = (total_tokens.cache_read_input_tokens / 1_000_000) * pricing['cache_read']
            
            print(f"  Input Tokens:              {total_tokens.input_tokens:,} (${input_cost:.4f})")
            print(f"  Output Tokens:             {total_tokens.output_tokens:,} (${output_cost:.4f})")
            print(f"  Cache Creation Tokens:     {total_tokens.cache_creation_input_tokens:,} (${cache_creation_cost:.4f})")
            print(f"  Cache Read Tokens:         {total_tokens.cache_read_input_tokens:,} (${cache_read_cost:.4f})")
        else:
            print(f"  Input Tokens:              {total_tokens.input_tokens:,}")
            print(f"  Output Tokens:             {total_tokens.output_tokens:,}")
            print(f"  Cache Creation Tokens:     {total_tokens.cache_creation_input_tokens:,}")
            print(f"  Cache Read Tokens:         {total_tokens.cache_read_input_tokens:,}")
            
        print(f"  Total Tokens:              {total_tokens.input_tokens + total_tokens.output_tokens + total_tokens.cache_creation_input_tokens + total_tokens.cache_read_input_tokens:,}")
        
        print(f"\nEstimated Total Cost: ${total_cost:.2f}")
        
# Calculate date range and cost averages
        sessions_with_time = [s for s in summaries if s.start_time is not None]
        if sessions_with_time:
            times = [cast(datetime, s.start_time) for s in sessions_with_time]
            earliest_date = min(times).date()
            latest_date = max(times).date()
            total_days = (latest_date - earliest_date).days + 1
            
            avg_daily_cost = total_cost / total_days if total_days > 0 else 0
            avg_weekly_cost = avg_daily_cost * 7
            avg_monthly_cost = avg_daily_cost * 30.44
            
            print(f"\nCost Averages:")
            print(f"  Date Range: {earliest_date} to {latest_date} ({total_days} days)")
            print(f"  Average Daily Cost: ${avg_daily_cost:.4f}")
            print(f"  Average Weekly Cost: ${avg_weekly_cost:.2f}")
            print(f"  Average Monthly Cost: ${avg_monthly_cost:.2f}")
        
        
        
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
        sessions_with_time = [s for s in summaries if s.start_time is not None]
        recent_sessions = sorted(sessions_with_time, 
                               key=lambda x: cast(datetime, x.start_time), reverse=True)[:self._recent_count]
        
        if recent_sessions:
            print(f"\nRecent Sessions (Last {min(self._recent_count, len(recent_sessions))}):")
            for session in recent_sessions:
                total_session_tokens = (session.token_usage.input_tokens + 
                                      session.token_usage.output_tokens + 
                                      session.token_usage.cache_creation_input_tokens + 
                                      session.token_usage.cache_read_input_tokens)
                cost = self.calculate_cost(session.token_usage, session.model or 'claude-sonnet-4-20250514')
                time_str = session.start_time.strftime('%Y-%m-%d %H:%M') if session.start_time else 'Unknown'
                print(f"  {time_str} - {os.path.basename(session.project_path)}: {total_session_tokens:,} tokens (${cost:.4f})")


def list_providers() -> int:
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


def list_available_models(provider: Optional[str] = None) -> int:
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


def calculate_alternative_model_cost(
    summaries: List[SessionSummary],
    alternative_model: str,
    claude_analyzer: Optional["ClaudeUsageAnalyzer"] = None,
) -> float:
    """Calculate what the cost would have been using an alternative model (Claude or non-Claude)."""
    if not claude_analyzer:
        return 0.0
    
    try:
        # Check if the specific model is available in pricing data
        pricing_data: Dict[str, Dict[str, Any]] = claude_analyzer.models_api.get_pricing()
        
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
        pricing: Dict[str, Any] = pricing_data[model_for_pricing]
        p_in = cast(float, pricing.get('input', 0.0))
        p_out = cast(float, pricing.get('output', 0.0))
        p_cache_w = cast(float, pricing.get('cache_write', p_in))
        p_cache_r = cast(float, pricing.get('cache_read', p_in))
        cost = 0.0
        
        # Basic input/output tokens (all models have these)
        cost += (total_usage.input_tokens / 1_000_000) * p_in
        cost += (total_usage.output_tokens / 1_000_000) * p_out
        
        # Cache tokens (only Claude models typically have cache pricing)
        if 'cache_write' in pricing and 'cache_read' in pricing:
            cost += (total_usage.cache_creation_input_tokens / 1_000_000) * p_cache_w
            cost += (total_usage.cache_read_input_tokens / 1_000_000) * p_cache_r
        else:
            # For non-Claude models, treat cache tokens as regular input tokens
            # since they don't have cache-specific pricing
            cost += (total_usage.cache_creation_input_tokens / 1_000_000) * p_in
            cost += (total_usage.cache_read_input_tokens / 1_000_000) * p_in
        
        return cost
    except Exception as e:
        return 0.0


def print_dedicated_comparison_summary(
    summaries: List[SessionSummary],
    compare_model: Optional[str] = None,
    claude_analyzer: Optional["ClaudeUsageAnalyzer"] = None,
) -> None:
    """Print a dedicated comparison summary focusing on cost differences."""
    if not compare_model or not claude_analyzer:
        return
        
    try:
        # Calculate current and alternative costs
        current_cost = sum(claude_analyzer.calculate_cost(summary.token_usage, summary.model or 'anthropic/claude-sonnet-4-20250514') 
                          for summary in summaries)
        alt_cost = calculate_alternative_model_cost(summaries, compare_model, claude_analyzer)
        
        if alt_cost <= 0:
            print(f"⚠️  Could not calculate cost for '{compare_model}' (model not found in pricing data)")
            return
            
        # Get time data for projections
        sessions_with_time = [s for s in summaries if s.start_time is not None]
        if len(sessions_with_time) >= 2:
            first_session = min(sessions_with_time, key=lambda s: cast(datetime, s.start_time))
            last_session = max(sessions_with_time, key=lambda s: cast(datetime, s.start_time))
            if first_session.start_time is not None and last_session.start_time is not None:
                total_days = (last_session.start_time - first_session.start_time).days + 1
            else:
                total_days = 1
        else:
            total_days = 1
            
        # Clean up model names
        current_models = set(s.model or 'claude-sonnet-4-20250514' for s in summaries)
        clean_compare_name = clean_model_name_for_display(compare_model)
        
        # Calculate token usage breakdown
        total_tokens = TokenUsage()
        for summary in summaries:
            usage = summary.token_usage
            total_tokens.input_tokens += usage.input_tokens
            total_tokens.output_tokens += usage.output_tokens
            total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
            total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens
            
        print("\n" + "=" * 80)
        print(f"📊 MODEL COMPARISON ANALYSIS")
        print("=" * 80)
        
        # Current vs Alternative summary
        cost_diff = current_cost - alt_cost
        percentage_diff = (cost_diff / current_cost) * 100 if current_cost > 0 else 0
        
        print(f"\n💰 TOTAL COST COMPARISON")
        print(f"   Current Models: ${current_cost:.4f}")
        print(f"   {clean_compare_name}: ${alt_cost:.4f}")
        if abs(cost_diff) > 0.001:
            if cost_diff > 0:
                print(f"   💚 Savings: ${cost_diff:.4f} ({percentage_diff:.1f}% less)")
            else:
                print(f"   🔴 Additional: ${-cost_diff:.4f} ({-percentage_diff:.1f}% more)")
        else:
            print(f"   ⚖️  Same cost")
            
        # Time-based projections
        print(f"\n⏰ COST PROJECTIONS")
        current_daily = current_cost / total_days
        alt_daily = alt_cost / total_days
        
        print(f"   Current Model(s):")
        print(f"   ├─ Daily: ${current_daily:.4f}")
        print(f"   ├─ Weekly: ${current_daily * 7:.4f}")
        print(f"   └─ Monthly: ${current_daily * 30.44:.4f}")
        
        print(f"   {clean_compare_name}:")
        print(f"   ├─ Daily: ${alt_daily:.4f}")
        print(f"   ├─ Weekly: ${alt_daily * 7:.4f}")
        print(f"   └─ Monthly: ${alt_daily * 30.44:.4f}")
        
        # Token breakdown with costs
        print(f"\n🔢 TOKEN USAGE & COST BREAKDOWN")
        
        # Get pricing for current and alternative models
        pricing_data = claude_analyzer.models_api.get_pricing()
        
        def clean_model_for_pricing(model_name: str) -> str:
            """Clean model name for pricing lookup."""
            # Remove common prefixes
            for prefix in ['anthropic/', 'openrouter/', 'openai/', 'groq/']:
                if model_name.startswith(prefix):
                    model_name = model_name[len(prefix):]
            # Handle nested prefixes like 'openrouter/openai/gpt-4o-mini'
            if '/' in model_name:
                parts = model_name.split('/')
                model_name = parts[-1]  # Take the last part
            return model_name
        
        current_model_raw = next(iter(current_models)) if current_models else 'claude-sonnet-4-20250514'
        current_model_for_pricing = clean_model_for_pricing(current_model_raw)
        alt_model_for_pricing = clean_model_for_pricing(compare_model)
        
        current_pricing = pricing_data.get(current_model_for_pricing, {})
        alt_pricing = pricing_data.get(alt_model_for_pricing, {})
        
        if current_pricing and alt_pricing:
            # Input tokens
            current_input_rate = current_pricing.get('input', current_pricing.get('prompt', 0))
            alt_input_rate = alt_pricing.get('input', alt_pricing.get('prompt', 0))
            current_input_cost = total_tokens.input_tokens * current_input_rate / 1_000_000
            alt_input_cost = total_tokens.input_tokens * alt_input_rate / 1_000_000
            
            # Output tokens
            current_output_rate = current_pricing.get('output', current_pricing.get('completion', 0))
            alt_output_rate = alt_pricing.get('output', alt_pricing.get('completion', 0))
            current_output_cost = total_tokens.output_tokens * current_output_rate / 1_000_000
            alt_output_cost = total_tokens.output_tokens * alt_output_rate / 1_000_000
            
            # Cache tokens
            current_cache_create_rate = current_pricing.get('cache_write', current_pricing.get('cache_creation_input', current_input_rate))
            alt_cache_create_rate = alt_pricing.get('cache_write', alt_pricing.get('cache_creation_input', alt_input_rate))
            current_cache_create_cost = total_tokens.cache_creation_input_tokens * current_cache_create_rate / 1_000_000
            alt_cache_create_cost = total_tokens.cache_creation_input_tokens * alt_cache_create_rate / 1_000_000
            
            current_cache_read_rate = current_pricing.get('cache_read', current_pricing.get('cache_read_input', current_input_rate))
            alt_cache_read_rate = alt_pricing.get('cache_read', alt_pricing.get('cache_read_input', alt_input_rate))
            current_cache_read_cost = total_tokens.cache_read_input_tokens * current_cache_read_rate / 1_000_000
            alt_cache_read_cost = total_tokens.cache_read_input_tokens * alt_cache_read_rate / 1_000_000
            
            print(f"   Input tokens ({total_tokens.input_tokens:,}):")
            print(f"   ├─ Current: ${current_input_cost:.6f}")
            print(f"   └─ {clean_compare_name}: ${alt_input_cost:.6f}")
            
            print(f"   Output tokens ({total_tokens.output_tokens:,}):")
            print(f"   ├─ Current: ${current_output_cost:.6f}")
            print(f"   └─ {clean_compare_name}: ${alt_output_cost:.6f}")
            
            if total_tokens.cache_creation_input_tokens > 0:
                print(f"   Cache creation ({total_tokens.cache_creation_input_tokens:,}):")
                print(f"   ├─ Current: ${current_cache_create_cost:.6f}")
                print(f"   └─ {clean_compare_name}: ${alt_cache_create_cost:.6f}")
                
            if total_tokens.cache_read_input_tokens > 0:
                print(f"   Cache read ({total_tokens.cache_read_input_tokens:,}):")
                print(f"   ├─ Current: ${current_cache_read_cost:.6f}")
                print(f"   └─ {clean_compare_name}: ${alt_cache_read_cost:.6f}")
        else:
            print(f"   ⚠️  Pricing data not available for detailed token breakdown")
            print(f"   Current model: {current_model_for_pricing} - Available: {'Yes' if current_pricing else 'No'}")
            print(f"   Compare model: {alt_model_for_pricing} - Available: {'Yes' if alt_pricing else 'No'}")
            
        print("=" * 80)
        
    except Exception as e:
        print(f"⚠️  Error generating comparison summary: {str(e)}")


def clean_model_name_for_display(model_name: str) -> str:
    """Clean up model name for display."""
    model_lower = model_name.lower()
    
    # Claude models
    if 'claude' in model_lower:
        if 'haiku' in model_lower:
            return 'Claude 3.5 Haiku'
        elif 'sonnet' in model_lower:
            if '4' in model_lower:
                return 'Claude Sonnet 4'
            else:
                return 'Claude 3.5 Sonnet'
        elif 'opus' in model_lower:
            return 'Claude 3 Opus'
    
    # OpenAI models
    elif 'gpt' in model_lower or 'o1' in model_lower:
        if 'gpt-4o' in model_lower:
            return 'GPT-4 Omni'
        elif 'gpt-4' in model_lower:
            return 'GPT-4'
        elif 'gpt-3.5' in model_lower:
            return 'GPT-3.5'
        elif 'o1-preview' in model_lower:
            return 'GPT-o1 Preview'
        elif 'o1-mini' in model_lower:
            return 'GPT-o1 Mini'
    
    # Groq models
    elif 'llama' in model_lower:
        if '70b' in model_lower:
            return 'Llama 3 70B (Groq)'
        elif '8b' in model_lower:
            return 'Llama 3 8B (Groq)'
        else:
            return 'Llama (Groq)'
    elif 'mixtral' in model_lower:
        return 'Mixtral 8x7B (Groq)'
    elif 'gemma' in model_lower:
        return 'Gemma 2 (Groq)'
    
    return model_name


def print_unified_summary(
    summaries: List[SessionSummary],
    recent_count: int = 10,
    claude_analyzer: Optional["ClaudeUsageAnalyzer"] = None,
    compare_model: Optional[str] = None,
) -> None:
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
    def clean_project_name(project_path: str) -> str:
        return os.path.basename(project_path)
    
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
        
        # Calculate date range and cost averages
        sessions_with_time = [s for s in summaries if s.start_time is not None]
        total_days = 1  # Default value
        if sessions_with_time:
            times = [cast(datetime, s.start_time) for s in sessions_with_time]
            earliest_date = min(times).date()
            latest_date = max(times).date()
            total_days = (latest_date - earliest_date).days + 1  # +1 to include both start and end dates
            
            # Calculate averages
            avg_daily_cost = total_cost / total_days if total_days > 0 else 0
            avg_weekly_cost = avg_daily_cost * 7
            avg_monthly_cost = avg_daily_cost * 30.44  # Average days per month (365.25/12)
            
            print(f"📅 {earliest_date} to {latest_date} ({total_days} days)")
            print(f"📊 Daily: ${avg_daily_cost:.2f} | Weekly: ${avg_weekly_cost:.2f} | Monthly: ${avg_monthly_cost:.2f}")
            
            # Add disclaimer for limited data
            if total_days < 7:
                print(f"⚠️  Projections based on {total_days} days - costs may vary significantly")
            elif total_days < 30:
                print(f"⚠️  Monthly projection based on {total_days} days - consider seasonal patterns")
        
        # Cost comparison with alternative model if requested
        if compare_model and claude_analyzer and claude_summaries:
            try:
                alt_cost = calculate_alternative_model_cost(summaries, compare_model, claude_analyzer)
                alt_total_cost = alt_cost  # alt_cost now includes all sessions, no need to add opencode_cost
                
                # Clean up model name for display
                clean_model_name = compare_model
                model_lower = compare_model.lower() if compare_model else ""
                
                if alt_cost > 0 and abs(alt_cost - total_cost) > 0.001:  # Avoid floating point comparison issues
                    cost_diff = total_cost - alt_cost
                    percentage_diff = (cost_diff / total_cost) * 100 if total_cost > 0 else 0
                    
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
                    
                    # Add alternative cost projections if we have time data
                    if sessions_with_time and total_days > 0:
                        alt_avg_daily = alt_total_cost / total_days
                        alt_avg_weekly = alt_avg_daily * 7
                        alt_avg_monthly = alt_avg_daily * 30.44
                        print(f"   └─ Alt. Daily: ${alt_avg_daily:.2f} | Weekly: ${alt_avg_weekly:.2f} | Monthly: ${alt_avg_monthly:.2f}")
                elif alt_cost > 0:
                    print(f"📊 Alternative: {clean_model_name} would cost ${alt_total_cost:.2f} (same cost)")
                    # Add alternative cost projections for same cost case too
                    if sessions_with_time and total_days > 0:
                        alt_avg_daily = alt_total_cost / total_days
                        alt_avg_weekly = alt_avg_daily * 7
                        alt_avg_monthly = alt_avg_daily * 30.44
                        print(f"   └─ Alt. Daily: ${alt_avg_daily:.2f} | Weekly: ${alt_avg_weekly:.2f} | Monthly: ${alt_avg_monthly:.2f}")
                else:
                    print(f"⚠️  Could not calculate cost for '{compare_model}' (model not found in pricing data)")
            except Exception as e:
                print(f"⚠️  Error comparing with {compare_model}: {str(e)}")
    
    # Token breakdown with costs (simplified)
    if total_tokens.cache_read_input_tokens > 0 or total_tokens.cache_creation_input_tokens > 0:
        print(f"\n📈 Token breakdown:")
        
        # Calculate costs per token type for the breakdown
        if claude_analyzer:
            try:
                pricing_data = claude_analyzer.models_api.get_pricing()
                default_model = 'claude-sonnet-4-20250514'
                pricing = pricing_data.get(default_model)
                if pricing is None:
                    # Find any available Claude model for pricing
                    available_models = [m for m in pricing_data.keys() if 'claude' in m.lower()]
                    if available_models:
                        default_model = available_models[0]
                        pricing = pricing_data[default_model]
                
                if pricing:
                    input_cost = (total_tokens.input_tokens / 1_000_000) * pricing['input']
                    output_cost = (total_tokens.output_tokens / 1_000_000) * pricing['output']
                    cache_creation_cost = (total_tokens.cache_creation_input_tokens / 1_000_000) * pricing['cache_write']
                    cache_read_cost = (total_tokens.cache_read_input_tokens / 1_000_000) * pricing['cache_read']
                    
                    print(f"   Input: {total_tokens.input_tokens:,} (${input_cost:.4f}) | Output: {total_tokens.output_tokens:,} (${output_cost:.4f})")
                    if total_tokens.cache_read_input_tokens > 0:
                        print(f"   Cache: {total_tokens.cache_read_input_tokens:,} read (${cache_read_cost:.4f}), {total_tokens.cache_creation_input_tokens:,} created (${cache_creation_cost:.4f})")
                else:
                    print(f"   Input: {total_tokens.input_tokens:,} | Output: {total_tokens.output_tokens:,}")
                    if total_tokens.cache_read_input_tokens > 0:
                        print(f"   Cache: {total_tokens.cache_read_input_tokens:,} read, {total_tokens.cache_creation_input_tokens:,} created")
            except:
                # Fallback to basic display if pricing calculation fails
                print(f"   Input: {total_tokens.input_tokens:,} | Output: {total_tokens.output_tokens:,}")
                if total_tokens.cache_read_input_tokens > 0:
                    print(f"   Cache: {total_tokens.cache_read_input_tokens:,} read, {total_tokens.cache_creation_input_tokens:,} created")
        else:
            print(f"   Input: {total_tokens.input_tokens:,} | Output: {total_tokens.output_tokens:,}")
            if total_tokens.cache_read_input_tokens > 0:
                print(f"   Cache: {total_tokens.cache_read_input_tokens:,} read, {total_tokens.cache_creation_input_tokens:,} created")
    
    # Top projects (simplified)
    project_usage: Dict[str, int] = defaultdict(int)
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
    model_usage: Dict[str, int] = defaultdict(int)
    for summary in summaries:
        model = summary.model or 'unknown'
        
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
    sessions_with_time = [s for s in summaries if s.start_time is not None]
    recent_sessions = sorted(sessions_with_time, 
                           key=lambda x: cast(datetime, x.start_time), reverse=True)[:recent_count]
    
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


def main() -> int:
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
        claude_summaries: List[SessionSummary] = []
        opencode_summaries: List[SessionSummary] = []
        claude_analyzer: Optional[ClaudeUsageAnalyzer] = None
        opencode_analyzer = None  # Runtime-only; type not declared to avoid missing symbol
        
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
                opencode_analyzer = OpenCodeUsageAnalyzer(opencode_dir=args.opencode_dir, quiet=opencode_quiet)  # type: ignore[name-defined]
                
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
        analyzer_for_cost: Optional[ClaudeUsageAnalyzer] = None
        if claude_summaries and not args.opencode_only:
            analyzer_for_cost = claude_analyzer
        
        if args.compare_model:
            # Show dedicated comparison summary for model comparison
            print_dedicated_comparison_summary(all_summaries, args.compare_model, analyzer_for_cost)
        else:
            # Show regular unified summary
            print_unified_summary(all_summaries, args.recent, analyzer_for_cost, args.compare_model)
        
        # Add verbose option for detailed breakdowns if requested
        if hasattr(args, 'verbose') and args.verbose:
            if claude_summaries and not args.opencode_only and claude_analyzer is not None:
                filtered_claude = [s for s in claude_summaries if s in all_summaries]
                if filtered_claude:
                    claude_analyzer._recent_count = args.recent
                    claude_analyzer.print_summary(filtered_claude)
            
            if opencode_summaries and not args.claude_only and opencode_analyzer is not None:
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
    return 0


if __name__ == '__main__':
    main()
