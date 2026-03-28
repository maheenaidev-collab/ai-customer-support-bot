"""
Core chatbot engine for AI Customer Support Bot.
Author: Maheen Riaz
"""

import json
import random
import joblib
from datetime import datetime
from preprocessor import TextPreprocessor
from config import CONFIDENCE_THRESHOLD, MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH


class CustomerSupportBot:
    """AI-powered customer support chatbot."""

    def __init__(self):
        """Initialize the chatbot."""
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.intents = None
        self.sessions = {}

    def load_model(self):
        """Load trained model and related files."""
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)

        with open('models/intents.json', 'r') as f:
            data = json.load(f)
            self.intents = {i['tag']: i['responses'] for i in data['intents']}

        print("✅ Model loaded successfully")

    def _get_or_create_session(self, session_id):
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'id': session_id,
                'history': [],
                'created_at': datetime.now().isoformat(),
                'escalated': False
            }
        return self.sessions[session_id]

    def predict_intent(self, text):
        """Predict the intent of user message."""
        cleaned = self.preprocessor.clean(text)
        vec = self.vectorizer.transform([cleaned])

        intent_idx = self.model.predict(vec)[0]
        probabilities = self.model.predict_proba(vec)[0]
        confidence = float(probabilities.max())

        intent = self.label_encoder.inverse_transform([intent_idx])[0]

        return intent, confidence

    def get_response(self, message, session_id='default'):
        """
        Get bot response for a customer message.

        Args:
            message (str): Customer's message
            session_id (str): Session identifier

        Returns:
            dict: Response with intent, text, confidence, and metadata
        """
        session = self._get_or_create_session(session_id)
        intent, confidence = self.predict_intent(message)

        escalated = False

        if confidence < CONFIDENCE_THRESHOLD:
            response_text = ("I'm not quite sure I understand. Could you rephrase that? "
                           "Or I can connect you with a human agent if you prefer.")
            intent = "low_confidence"
        elif intent == "escalation":
            response_text = random.choice(self.intents.get(intent, ["Connecting you to a human agent..."]))
            escalated = True
            session['escalated'] = True
        else:
            responses = self.intents.get(intent, ["I'm here to help! Could you tell me more?"])
            response_text = random.choice(responses)

        # Log to session history
        session['history'].append({
            'timestamp': datetime.now().isoformat(),
            'customer': message,
            'bot': response_text,
            'intent': intent,
            'confidence': confidence
        })

        return {
            'session_id': session_id,
            'intent': intent,
            'response': response_text,
            'confidence': round(confidence, 4),
            'escalated': escalated
        }

    def get_session_history(self, session_id):
        """Get conversation history for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return session['history']

    def get_analytics(self):
        """Get basic analytics across all sessions."""
        total_conversations = len(self.sessions)
        total_messages = sum(len(s['history']) for s in self.sessions.values())
        escalated = sum(1 for s in self.sessions.values() if s['escalated'])

        intent_counts = {}
        total_confidence = 0
        msg_count = 0

        for session in self.sessions.values():
            for msg in session['history']:
                intent = msg['intent']
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                total_confidence += msg['confidence']
                msg_count += 1

        return {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'escalated_sessions': escalated,
            'escalation_rate': round(escalated / max(total_conversations, 1) * 100, 1),
            'avg_confidence': round(total_confidence / max(msg_count, 1), 4),
            'intent_distribution': dict(sorted(intent_counts.items(), key=lambda x: x[1], reverse=True))
        }
