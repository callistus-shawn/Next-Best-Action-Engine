#!/usr/bin/env python3

import json
import os
import random
import re
import string
import numpy as np
from collections import Counter, defaultdict


import tensorflow as tf
from transformers import AutoTokenizer
from keras.layers import TFSMLayer



class SimpleMBTIClassifier:
    """MBTI classifier using BERT model"""
    
    def __init__(self, model_path: str):
        """Initialize the classifier."""
        
        self.mbti_types = [
            'ENTJ', 'ENTP', 'ENFJ', 'ENFP', 'ESTJ', 'ESTP', 'ESFJ', 'ESFP',
            'INTJ', 'INTP', 'INFJ', 'INFP', 'ISTJ', 'ISTP', 'ISFJ', 'ISFP'
        ]
        
        self.model = None
        self.tokenizer = None
        
        
        
        if os.path.exists(model_path):
            try:
                print(f"Loading BERT model from {model_path}...")
                call_endpoint = "serving_default"
                self.model = TFSMLayer(model_path, call_endpoint=call_endpoint)
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                print(f"BERT model loaded successfully!")
            except Exception as e:
                print(f"Failed to load BERT model: {e}")
  
        else:
            print(f"Model file not found: {model_path}")
    
        
        print(f"Initialized MBTI classifier")
        print(f"Model path: {model_path}")
        print(f"Using BERT model")
    
    def preprocess_text(self, text: str) -> str:
        regex = re.compile('[%s]' % re.escape('|'))
        text = regex.sub(" ", text)
        words = str(text).split()
        words = [i.lower() + " " for i in words]
        words = [i for i in words if not "http" in i]
        words = " ".join(words)
        words = words.translate(str.maketrans('', '', string.punctuation))
        return words
    
    def predict_with_model(self, text: str) -> str:
        """Predict MBTI type using BERT model."""
        try:
        
            cleaned = self.preprocess_text(text)
            
            maxlen = 256
            input_ids = self.tokenizer.encode(
                cleaned, 
                max_length=maxlen, 
                padding='max_length', 
                truncation=True
            )
            input_array = np.array([input_ids], dtype=np.int32)
            
            # Get prediction
            output = self.model(input_array)
            print(output)
            output_tensor = output['dense_2']
            predicted_class = np.argmax(output_tensor.numpy(), axis=1)[0]
            
            mbti_type = self.mbti_types[predicted_class]
            return mbti_type
            
        except Exception as e:
            print(f"Error in BERT prediction: {e}")
            return 'Unknown'

    def predict_with_random(self, text: str) -> str:
        """Predict MBTI type using random selection (fallback)."""
        try:
            mbti_type = random.choice(self.mbti_types)
            return mbti_type
        except Exception as e:
            print(f"Error in random prediction: {e}")
            return 'Unknown'
    
    def predict(self, text: str) -> str:
        """Predict MBTI type for text."""
        result = self.predict_with_model(text)
        return result

def extract_customer_text(conversation):
    """Extract customer text from conversation."""
    customer_messages = []
    
   
    chat_history = conversation.get('chat_history', [])
    for message in chat_history:
        if message.get('response_type') == 'Customer':
            text = message.get('response', {}).get('text', '')
            if text:
                customer_messages.append(text)
    
    return ' '.join(customer_messages)

def mbti_tag():
    """Entry function."""
 
    input_file = "data/twitter_conversations_tagged.json"
    output_file = "data/twitter_conversations_with_mbti.json"
    model_path = "bertcls"  
    
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    try:
        # Load conversations
        print(f"Loading conversations from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        conversations = conversations[:20]
        print(f"Loaded {len(conversations)} conversations")
        
       
        customer_conversations = defaultdict(list)
        
        for conversation in conversations:
            customer_id = conversation.get('customer_id', 'unknown')
            customer_conversations[customer_id].append(conversation)
        
        print(f"Found {len(customer_conversations)} unique customers")
        
     
        classifier = SimpleMBTIClassifier(model_path)
        
        
       
        customer_mbti_data = {}
        
        for i, (customer_id, customer_convos) in enumerate(customer_conversations.items()):
            try:
                # Extract all customer text from all conversations
                all_customer_text = []
                total_conversations = len(customer_convos)
                
                for conversation in customer_convos:
                    customer_text = extract_customer_text(conversation)
                    if customer_text.strip():
                        all_customer_text.append(customer_text)
                
                # Combine all customer text
                combined_text = ' '.join(all_customer_text)
                
                   
                mbti_type = classifier.predict(combined_text)
                    
                mbti_data = {
                        'mbti_type': mbti_type,
                        'text_length': len(combined_text.split()),
                        'total_conversations': total_conversations
                    }
                
            
                customer_mbti_data[customer_id] = mbti_data
                
              
                    
            except Exception as e:
                print(f"Error processing customer {customer_id}: {e}")
                customer_mbti_data[customer_id] = {
                    'mbti_type': 'Error',
                    'text_length': 0,
                    'total_conversations': len(customer_convos)
                }
        
        # Add MBTI data to all conversations for each customer
        print(f"Adding MBTI data to conversations...")
        updated_conversations = []
        
        for conversation in conversations:
            customer_id = conversation.get('customer_id', 'unknown')
            mbti_data = customer_mbti_data.get(customer_id, {
                'mbti_type': 'Unknown',
                'text_length': 0,
                'total_conversations': 0
            })
            
           
            conversation['mbti_type'] = mbti_data['mbti_type']
            updated_conversations.append(conversation)
        
        # Save results
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_conversations, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(updated_conversations)} conversations with MBTI tags")
        
    
        
        print(f"Total conversations: {len(updated_conversations)}")
        print(f"Unique customers: {len(customer_conversations)}")
    
        
    except Exception as e:
        print(f"Error: {e}")
