from main import ChatbotHandler
import json

def main():
    chatbot = ChatbotHandler("../intents.json")
    chatbot.parse_intents()
    chatbot.prep_data()
    print("Training the model...")
    chatbot.train(batch_size=8, lr=0.001, epochs=100)
    print("Saving the model...")
    chatbot.save_model("../Model/model.pth", "../Model/model_data.json")
    print("Training and saving completed successfully!")

if __name__ == "__main__":
    main() 