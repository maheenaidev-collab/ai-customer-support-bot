## 📄 config.py

```python
"""
Configuration settings for AI Customer Support Bot.
Author: Maheen Riaz
"""

# Model settings
CONFIDENCE_THRESHOLD = 0.65  # Below this, escalate to human
MODEL_PATH = "models/chatbot_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False

# Conversation settings
MAX_SESSION_HISTORY = 50
SESSION_TIMEOUT_MINUTES = 30

# Logging
LOG_DIR = "logs"
LOG_CONVERSATIONS = True

# Analytics
ANALYTICS_FILE = "logs/analytics.json"
