from main import ChatbotHandler
import json
import sys
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "Model")
    print(f"Current directory: {current_dir}", file=sys.stderr)
    print(f"Model directory: {model_dir}", file=sys.stderr)
    chatbot = ChatbotHandler("intents.json")
    chatbot.load_model(
        os.path.join(model_dir, "model.pth"),
        os.path.join(model_dir, "model_data.json")
    )
    message = sys.stdin.read().strip()
    response = chatbot.process_message(message)
    print(response)

if __name__ == "__main__":
    main()