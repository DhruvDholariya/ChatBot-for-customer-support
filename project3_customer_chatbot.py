"""
PROJECT 3: AI-BASED CUSTOMER SUPPORT CHATBOT
Author: ML Projects Collection
Date: 2025

AIM:
To create an intelligent conversational AI chatbot that can understand customer 
queries and provide automated responses for customer support services.

TECHNOLOGIES USED:
- Python 3.x
- scikit-learn (TF-IDF Vectorization, Cosine Similarity)
- NLTK / NLP techniques
- Regular Expressions
- NumPy (Array operations)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# Define chatbot knowledge base with intents and responses
INTENTS = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning", "good evening", "greetings",
            "what's up", "how are you", "howdy", "hola", "namaste"
        ],
        "responses": [
            "Hello! Welcome to our customer support. How can I assist you today?",
            "Hi there! I'm here to help. What can I do for you?",
            "Greetings! How may I help you today?",
            "Hello! Feel free to ask me anything about our services."
        ]
    },
    "product_inquiry": {
        "patterns": [
            "what products do you have", "show me products", "product list",
            "what do you sell", "available items", "product catalog",
            "what can i buy", "items for sale", "merchandise"
        ],
        "responses": [
            "We offer a wide range of products including Electronics, Clothing, Home Appliances, Books, and Sports Equipment. Which category interests you?",
            "Our product categories include: Electronics, Fashion, Home & Kitchen, Books, and Sports. What are you looking for?",
            "We have Electronics, Apparel, Home Goods, Books, and Sporting Goods available. How can I help you find something?"
        ]
    },
    "order_status": {
        "patterns": [
            "where is my order", "order status", "track order", "delivery status",
            "when will my order arrive", "shipping information", "order tracking",
            "check my order", "order location", "package tracking"
        ],
        "responses": [
            "To check your order status, please provide your order ID and I'll look it up for you.",
            "I can help you track your order. Please share your order number.",
            "Sure! To track your order, I'll need your order ID. Could you provide that?"
        ]
    },
    "payment": {
        "patterns": [
            "payment methods", "how to pay", "payment options", "accepted payments",
            "can i pay with card", "payment information", "what payment do you accept",
            "credit card payment", "debit card", "online payment"
        ],
        "responses": [
            "We accept Credit/Debit Cards, PayPal, Net Banking, UPI, and Cash on Delivery.",
            "You can pay using Credit Cards, Debit Cards, PayPal, UPI, or choose Cash on Delivery.",
            "Our payment options include: Cards (Visa, Mastercard, Amex), PayPal, UPI, Net Banking, and COD."
        ]
    },
    "return_policy": {
        "patterns": [
            "return policy", "can i return", "refund", "exchange product",
            "return item", "money back", "product return", "refund policy",
            "how to return", "return process"
        ],
        "responses": [
            "We offer a 30-day return policy. Items must be unused and in original packaging. Refunds are processed within 5-7 business days.",
            "You can return products within 30 days of delivery. The item should be in original condition. Refunds take 5-7 days.",
            "Our return policy allows returns within 30 days. Products must be unused. Refunds are issued within a week of receiving the returned item."
        ]
    },
    "customer_service": {
        "patterns": [
            "contact support", "customer service", "speak to agent", "human support",
            "talk to representative", "customer care", "support number", "help desk",
            "contact number", "email support"
        ],
        "responses": [
            "You can reach our customer service at: support@company.com or call +1-800-123-4567 (Mon-Fri, 9 AM - 6 PM)",
            "Our customer support is available at support@company.com and +1-800-123-4567 during business hours.",
            "Contact us at: Email - support@company.com, Phone - +1-800-123-4567 (Available 9 AM - 6 PM, Mon-Fri)"
        ]
    },
    "shipping": {
        "patterns": [
            "shipping cost", "delivery charges", "shipping time", "how long delivery",
            "shipping information", "delivery time", "shipping fee", "delivery charges"
        ],
        "responses": [
            "Standard shipping takes 5-7 business days and costs $5. Express shipping (2-3 days) costs $15. Free shipping on orders over $50!",
            "Shipping: Standard (5-7 days) - $5, Express (2-3 days) - $15. Orders above $50 ship free!",
            "We offer Standard Shipping ($5, 5-7 days) and Express Shipping ($15, 2-3 days). Free shipping for orders over $50!"
        ]
    },
    "thanks": {
        "patterns": [
            "thank you", "thanks", "appreciate it", "thanks a lot",
            "thank you very much", "thx", "appreciate your help", "grateful"
        ],
        "responses": [
            "You're welcome! Happy to help. Is there anything else I can assist you with?",
            "Glad I could help! Feel free to reach out if you need anything else.",
            "You're most welcome! Don't hesitate to ask if you have more questions."
        ]
    },
    "goodbye": {
        "patterns": [
            "bye", "goodbye", "see you", "talk to you later", "have a good day",
            "take care", "see you later", "catch you later", "farewell"
        ],
        "responses": [
            "Goodbye! Have a great day. Feel free to return if you need assistance!",
            "Take care! Thanks for contacting us. Have a wonderful day!",
            "See you! Don't hesitate to reach out if you need help. Have a great day!"
        ]
    }
}

class CustomerSupportChatbot:
    """AI-based Customer Support Chatbot using NLP"""

    def __init__(self, intents_data):
        """Initialize chatbot with intent data"""
        self.intents = intents_data
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))

        # Prepare training data
        self.patterns = []
        self.intent_labels = []

        for intent_name, intent_data in intents_data.items():
            for pattern in intent_data['patterns']:
                self.patterns.append(pattern.lower())
                self.intent_labels.append(intent_name)

        # Train vectorizer
        self.pattern_vectors = self.vectorizer.fit_transform(self.patterns)

        print("="*70)
        print("CHATBOT INITIALIZED SUCCESSFULLY")
        print("="*70)
        print(f"\nKnowledge Base:")
        print(f"- Total intents: {len(intents_data)}")
        print(f"- Total patterns: {len(self.patterns)}")
        print(f"- NLP Model: TF-IDF with Cosine Similarity")

    def preprocess(self, text):
        """Preprocess user input"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def get_response(self, user_input):
        """Get chatbot response for user input"""
        # Preprocess input
        processed_input = self.preprocess(user_input)

        # Vectorize input
        input_vector = self.vectorizer.transform([processed_input])

        # Calculate cosine similarity with all patterns
        similarities = cosine_similarity(input_vector, self.pattern_vectors)[0]

        # Find best match
        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]

        # If confidence too low, return default response
        if confidence < 0.3:
            return ("I'm not sure I understand. Could you rephrase that? "
                   "Or type 'help' for available options."), "unknown", confidence

        # Get matched intent and random response
        matched_intent = self.intent_labels[best_match_idx]
        response = np.random.choice(self.intents[matched_intent]['responses'])

        return response, matched_intent, confidence

def test_chatbot():
    """Test chatbot with sample queries"""
    print("\n" + "="*70)
    print("TESTING CHATBOT WITH SAMPLE QUERIES")
    print("="*70)

    # Initialize chatbot
    chatbot = CustomerSupportChatbot(INTENTS)

    # Test queries
    test_queries = [
        "Hello! I need some help",
        "What products are available?",
        "How can I track my order?",
        "What payment methods do you accept?",
        "What is your return policy?",
        "How much does shipping cost?",
        "Can I speak to customer service?",
        "Thank you for your help!",
        "Goodbye"
    ]

    conversation_log = []

    print("\n" + "-"*70)
    for query in test_queries:
        response, intent, confidence = chatbot.get_response(query)
        conversation_log.append({
            'user': query,
            'bot': response,
            'intent': intent,
            'confidence': confidence
        })

        print(f"\n👤 User: {query}")
        print(f"🤖 Bot: {response}")
        print(f"   [Intent: {intent} | Confidence: {confidence:.2f}]")
        print("-"*70)

    return chatbot, conversation_log

def visualize_conversation(conversation_log):
    """Generate chat interface visualization"""
    print("\nGenerating conversation visualization...")

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('AI Customer Support Chatbot - Conversation Log', 
                 fontsize=18, fontweight='bold', y=0.98)

    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(conversation_log) * 2 + 1)

    y_position = len(conversation_log) * 2

    colors_user = ['#e3f2fd', '#e1f5fe', '#e0f2f1']
    colors_bot = ['#f3e5f5', '#fce4ec', '#fff3e0']

    for idx, log in enumerate(conversation_log):
        # User message
        user_box = patches.FancyBboxPatch(
            (3.5, y_position - 0.7), 6, 0.6,
            boxstyle="round,pad=0.1",
            linewidth=2,
            edgecolor='#1976d2',
            facecolor=colors_user[idx % 3]
        )
        ax.add_patch(user_box)

        user_text = log['user']
        if len(user_text) > 60:
            user_text = user_text[:60] + '...'

        ax.text(6.5, y_position - 0.4, user_text,
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(9.7, y_position - 0.4, '👤', ha='center', va='center', fontsize=16)

        y_position -= 1

        # Bot message
        bot_box = patches.FancyBboxPatch(
            (0.5, y_position - 0.7), 6, 0.6,
            boxstyle="round,pad=0.1",
            linewidth=2,
            edgecolor='#7b1fa2',
            facecolor=colors_bot[idx % 3]
        )
        ax.add_patch(bot_box)

        bot_text = log['bot']
        if len(bot_text) > 60:
            bot_text = bot_text[:57] + '...'

        ax.text(3.5, y_position - 0.4, bot_text,
                ha='center', va='center', fontsize=10)
        ax.text(0.3, y_position - 0.4, '🤖', ha='center', va='center', fontsize=16)

        intent_text = f"[{log['intent']}]"
        ax.text(6.8, y_position - 0.4, intent_text,
                ha='left', va='center', fontsize=7, style='italic', color='#666')

        y_position -= 1.2

    # Header
    header_box = patches.FancyBboxPatch(
        (0.5, y_position + 0.5), 9, 0.8,
        boxstyle="round,pad=0.1",
        linewidth=3,
        edgecolor='#2196f3',
        facecolor='#bbdefb'
    )
    ax.add_patch(header_box)
    ax.text(5, y_position + 0.9, '💬 Customer Support Chat Interface',
            ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('chatbot_conversation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: chatbot_conversation.png")
    plt.show()

def interactive_mode(chatbot):
    """Run chatbot in interactive mode"""
    print("\n" + "="*70)
    print("INTERACTIVE CHATBOT MODE")
    print("="*70)
    print("Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        user_input = input("👤 You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'stop']:
            print("🤖 Bot: Goodbye! Have a great day!")
            break

        if not user_input:
            continue

        response, intent, confidence = chatbot.get_response(user_input)
        print(f"🤖 Bot: {response}\n")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("AI-BASED CUSTOMER SUPPORT CHATBOT")
    print("="*70)

    # Test chatbot
    chatbot, conversation_log = test_chatbot()

    # Visualize conversation
    visualize_conversation(conversation_log)

    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)

    # Optional: Run interactive mode
    # Uncomment below to chat with the bot
    # interactive_mode(chatbot)

if __name__ == "__main__":
    main()
