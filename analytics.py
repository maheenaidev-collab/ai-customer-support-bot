"""
Analytics module for AI Customer Support Bot.
Author: Maheen Riaz
"""

import json
import os
from datetime import datetime
from config import ANALYTICS_FILE


class Analytics:
    """Track and report chatbot performance metrics."""

    def __init__(self):
        self.data = self._load()

    def _load(self):
        """Load existing analytics data."""
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                return json.load(f)
        return {'conversations': [], 'daily_stats': {}}

    def _save(self):
        """Save analytics data."""
        os.makedirs(os.path.dirname(ANALYTICS_FILE), exist_ok=True)
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(self.data, f, indent=2)

    def log_interaction(self, session_id, intent, confidence, escalated):
        """Log a single interaction."""
        today = datetime.now().strftime('%Y-%m-%d')

        if today not in self.data['daily_stats']:
            self.data['daily_stats'][today] = {
                'total_messages': 0,
                'intents': {},
                'escalations': 0,
                'avg_confidence': 0,
                'confidence_sum': 0
            }

        stats = self.data['daily_stats'][today]
        stats['total_messages'] += 1
        stats['intents'][intent] = stats['intents'].get(intent, 0) + 1
        stats['confidence_sum'] += confidence
        stats['avg_confidence'] = round(stats['confidence_sum'] / stats['total_messages'], 4)

        if escalated:
            stats['escalations'] += 1

        self._save()

    def get_report(self):
        """Generate analytics report."""
        total_msgs = sum(d['total_messages'] for d in self.data['daily_stats'].values())
        total_escalations = sum(d['escalations'] for d in self.data['daily_stats'].values())

        all_intents = {}
        for day_stats in self.data['daily_stats'].values():
            for intent, count in day_stats['intents'].items():
                all_intents[intent] = all_intents.get(intent, 0) + count

        return {
            'total_messages': total_msgs,
            'total_escalations': total_escalations,
            'escalation_rate': round(total_escalations / max(total_msgs, 1) * 100, 1),
            'top_intents': dict(sorted(all_intents.items(), key=lambda x: x[1], reverse=True)[:5]),
            'daily_breakdown': self.data['daily_stats']
        }
