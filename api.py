"""
Flask REST API for AI Customer Support Bot.
Author: Maheen Riaz
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import CustomerSupportBot
from config import API_HOST, API_PORT, API_DEBUG

app = Flask(__name__)
CORS(app)

bot = CustomerSupportBot()
bot.load_model()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'AI Customer Support Bot'})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message and return bot response."""
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400

    message = data['message']
    session_id = data.get('session_id', 'default')

    result = bot.get_response(message, session_id)
    return jsonify(result)


@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new chat session."""
    data = request.get_json() or {}
    session_id = data.get('session_id', f'session_{len(bot.sessions) + 1}')

    bot._get_or_create_session(session_id)

    return jsonify({
        'session_id': session_id,
        'message': 'Session started. How can I help you today?'
    })


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session conversation history."""
    history = bot.get_session_history(session_id)

    if history is None:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({'session_id': session_id, 'history': history})


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Submit feedback on a bot response."""
    data = request.get_json()

    if not data or 'rating' not in data:
        return jsonify({'error': 'Rating is required'}), 400

    return jsonify({
        'status': 'received',
        'message': 'Thank you for your feedback!'
    })


@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Get bot performance analytics."""
    stats = bot.get_analytics()
    return jsonify(stats)


def start_api():
    """Start the Flask API server."""
    print(f"\n🚀 API server starting at http://{API_HOST}:{API_PORT}")
    print("📡 Endpoints:")
    print("   POST /api/chat          - Send a message")
    print("   POST /api/session/start - Start new session")
    print("   GET  /api/session/<id>  - Get session history")
    print("   POST /api/feedback      - Submit feedback")
    print("   GET  /api/analytics     - View analytics")
    print("   GET  /api/health        - Health check\n")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)


if __name__ == '__main__':
    start_api()
