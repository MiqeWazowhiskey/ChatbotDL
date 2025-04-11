import os
import json
import random

import nltk
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

#Download the punkt tokenizer for the first time
#nltk.download('punkt_tab')

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

class ChatbotHandler:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = []

        self.function_mappings = function_mappings if function_mappings else {}

        self.X = None
        self.y = None

    @staticmethod
    def tokenize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        #lemmatize
        return [lemmatizer.lemmatize(word.lower()) for word in words]






