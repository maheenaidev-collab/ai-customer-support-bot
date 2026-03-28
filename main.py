"""
AI Customer Support Bot - CLI Entry Point
Author: Maheen Riaz
"""

import argparse
from chatbot import CustomerSupportBot
from api import start_api


def interactive_chat(bot):
    """Run interactive chat mode."""
    print("\n" + "=" * 55)
    print("🤖 AI Customer Support Bot - Interactive Mode")
    print("=" * 55)
    print("Type your message. Type 'quit' to exit.")
    print("Type 'history' to see conversation history.")
    print("Type 'analytics' to see bot stats.\n")

    session_id = "interactive_user"

    while True:
        user_input = input("👤 You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\n🤖 Bot: Thank you for chatting! Goodbye! 👋")
            break

        if user_input.lower() == 'history':
            history = bot.get_session_history(session_id)
            if history:
                print("\n📜 Conversation History:")
                for msg in history:
                    print(f"  👤 {msg['customer']}")
                    print(f"  🤖 {msg['bot']} [{msg['intent']}:{msg['confidence']:.0%}]")
                    print()
            else:
                print("  No history yet.\n")
            continue

        if user_input.lower() == 'analytics':
            stats = bot.get_analytics()
            print(f"\n📊 Analytics:")
            print(f"  Sessions: {stats['total_conversations']}")
            print(f"  Messages: {stats['total_messages']}")
            print(f"  Escalated: {stats['escalated_sessions']}")
            print(f"  Avg Confidence: {stats['avg_confidence']:.0%}")
            print(f"  Top Intents: {stats['intent_distribution']}\n")
            continue

        result = bot.get_response(user_input, session_id)

        confidence_bar = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "🔴"
        print(f"🤖 Bot: {result['response']}")
        print(f"   {confidence_bar} [{result['intent']} | {result['confidence']:.0%}]\n")

        if result['escalated']:
            print("   ⚠️  Escalated to human agent\n")


def main():
    parser = argparse.ArgumentParser(description='AI Customer Support Bot')
    parser.add_argument('--chat', action='store_true', help='Start interactive chat')
    parser.add_argument('--api', action='store_true', help='Start REST API server')
    parser.add_argument('--analytics', action='store_true', help='Show analytics')

    args = parser.parse_args()

    bot = CustomerSupportBot()
    bot.load_model()

    if args.chat:
        interactive_chat(bot)
    elif args.api:
        start_api()
    elif args.analytics:
        stats = bot.get_analytics()
        print("\n📊 Bot Analytics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("🤖 AI Customer Support Bot")
        print("Run with --help to see options")
        print("Quick start: python main.py --chat")
        print("Start API: python main.py --api")


if __name__ == '__main__':
    main()
