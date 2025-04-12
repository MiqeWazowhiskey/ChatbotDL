import os
import sys
import json
import random

import nltk
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#Download the punkt tokenizer for the first time
#nltk.download('punkt_tab')
#nltk.download('wordnet')

class ChatbotModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        #relu
        x=self.relu(self.fc1(x))
        x=self.dropout(x)
        #relu
        x=self.relu(self.fc2(x))
        x=self.dropout(x)
        #going to apply softmax in a loss function
        x=self.fc3(x)

        return x

class ChatbotHandler:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings if function_mappings else {}

        self.X = None
        self.y = None

    @staticmethod
    def tokenize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        #lemmatize
        return [lemmatizer.lemmatize(word.lower()) for word in words]


    @staticmethod
    def BoW(words,vocabulary):
        return[1 if word in words else 0 for word in vocabulary]

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as file:
                intents = json.load(file)

            #Iterate through the intents to parse
            for intent in intents['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    ptrn_words = self.tokenize(pattern)
                    self.vocabulary.extend(ptrn_words)
                    self.documents.append((ptrn_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prep_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.BoW(words, self.vocabulary)

            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train(self,batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1],len(self.intents))

        #loss function
        criterion = nn.CrossEntropyLoss()
        #optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        #train
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
        print("Training complete.")


    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as file:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents),
                'vocabulary': self.vocabulary,
                'intents': self.intents,
                'intents_responses': self.intents_responses
            }, file)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as file:
            dimensions = json.load(file)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        
        self.vocabulary = dimensions['vocabulary']
        self.intents = dimensions['intents']
        self.intents_responses = dimensions['intents_responses']

    def process_message(self, message):
        #tokenize
        words = self.tokenize(message)
        #bag of words
        bag = self.BoW(words, self.vocabulary)
        if not any(bag):
            print("Warning: No words matched in vocabulary", file=sys.stderr)
            return "I'm sorry, I don't understand that."
           
        #convert to tensor
        bag_tensor = torch.tensor(bag, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor).unsqueeze(0)

        predicted_class_index = torch.argmax(predictions,dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None
