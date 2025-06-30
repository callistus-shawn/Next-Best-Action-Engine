import json
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import re
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

class NBAEvaluation:
    def __init__(self, api_key: str):
        """
        Initialize the NBA Customer Service Evaluator with API key for LLM features.
        
        Args:
            api_key: Google API key for LLM features
        """
        self.llm = None
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
                print("LLM initialized successfully")
            except Exception as e:
                print(f"Failed to initialize LLM: {e}")
        
        else:
            print("No API key provided. LLM features will be disabled.")
    
    def load_nba_data(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Load NBA customer service data from JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing NBA data
            
        Returns:
            List of customer service interactions
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Loaded {len(data)} customer service interactions")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def load_tagged_conversations(self, json_file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load tagged conversations and create a lookup dictionary by customer_id.
        
        Args:
            json_file_path: Path to the tagged conversations JSON file
            
        Returns:
            Dictionary mapping customer_id to conversation data
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                conversations = json.load(file)
            
           
            conversation_lookup = {}
            for conv in conversations:
                customer_id = conv.get('customer_id')
                if customer_id:
                    conversation_lookup[customer_id] = conv
            
            print(f"Loaded {len(conversations)} tagged conversations")
            print(f"Found {len(conversation_lookup)} unique customer conversations")
            return conversation_lookup
        except Exception as e:
            print(f"Error loading tagged conversations: {e}")
            return {}
    
    def format_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Format chat history for LLM context.
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            Formatted string representation of chat history
        """
        if not chat_history:
            return "No chat history available."
        
        formatted_history = []
        for i, message in enumerate(chat_history):
            response_type = message.get('response_type', 'Unknown')
            response = message.get('response', {})
            text = response.get('text', 'No text')
            created_at = response.get('created_at', 'Unknown time')
            
            formatted_history.append(f"{i+1}. [{response_type}] {created_at}: {text}")
        
        return "\n".join(formatted_history)
    
    def evaluate_response_usefulness(self, interaction: Dict[str, Any], conversation_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to evaluate if the company's last reply was useful from a customer perspective.
        
        Args:
            interaction: Dictionary containing customer service interaction data
            conversation_lookup: Dictionary mapping customer_id to conversation data
            
        Returns:
            Dictionary with evaluation results
        """
     
        
        customer_id = interaction.get('customer_id', '')
        conversation_data = conversation_lookup.get(customer_id, {})
        chat_history = conversation_data.get('chat_history', [])
        
        # Prepare context for evaluation
        context = f"""
Customer Service Interaction Analysis:

Customer ID: {customer_id}
Channel: {interaction.get('channel', 'N/A')}
Issue Status: {interaction.get('issue_status', 'N/A')}
Nature of Support: {conversation_data.get('nature_of_support', 'N/A')}
Customer Sentiment: {conversation_data.get('customer_sentiment', 'N/A')}
Primary Tweet: {conversation_data.get('primary_tweet', 'N/A')}

Full Conversation History:
{self.format_chat_history(chat_history)}

Company's Final Response:
{interaction.get('message', 'N/A')}

Company's Reasoning for Response:
{interaction.get('reasoning', 'N/A')}

Response Timestamp: {interaction.get('send_time', 'N/A')}

Task: Evaluate the usefulness of the company's response from a customer perspective, considering the full conversation context.
"""
        
        prompt = f"""
{context}

As a customer service evaluation expert, analyze the company's response and determine if it was useful from a customer's perspective, considering the full conversation history.

Evaluation Criteria:
1. **Relevance**: Does the response directly address the customer's issue based on the conversation context?
2. **Actionability**: Does it provide clear next steps or solutions?
3. **Empathy**: Does it show understanding of the customer's situation and frustration level?
4. **Timeliness**: Is the response appropriate for the urgency and length of the conversation?
5. **Completeness**: Does it provide sufficient information to resolve the issue or move it forward?
6. **Context Awareness**: Does the response acknowledge the conversation history and previous interactions?

Scoring System:
- 5: Excellent - Fully addresses the issue with clear next steps, shows empathy for the conversation history
- 4: Good - Addresses most aspects of the issue, acknowledges the conversation context
- 3: Adequate - Partially addresses the issue, some awareness of context
- 2: Poor - Minimal or unclear response, doesn't consider conversation history
- 1: Very Poor - Does not address the issue or makes it worse, ignores conversation context

Respond in this exact JSON format:
{{
    "customer_id": "{customer_id}",
    "usefulness_score": 1-5,
    "evaluation": "brief summary of evaluation"
}}
"""
        
        try:
            response = self.llm.invoke(prompt)
            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"LLM evaluation failed for customer {customer_id}: {e}")
           
    
    def evaluate_all_responses(self, nba_file_path: str, tagged_file_path: str) -> List[Dict[str, Any]]:
        """
        Evaluate all customer service responses in the NBA dataset with full conversation context.
        
        Args:
            nba_file_path: Path to the NBA JSON file containing company responses
            tagged_file_path: Path to the tagged conversations JSON file
            
        Returns:
            List of evaluation results for all interactions
        """
        interactions = self.load_nba_data(nba_file_path)
        if not interactions:
            return []
        
        conversation_lookup = self.load_tagged_conversations(tagged_file_path)
        if not conversation_lookup:
            print("Warning: No conversation data loaded. Evaluations will be limited.")
        
        evaluations = []
        print(f"Evaluating {len(interactions)} customer service interactions...")
        
        for i, interaction in enumerate(interactions):
            customer_id = interaction.get('customer_id', 'N/A')
            print(f"Evaluating interaction {i+1}/{len(interactions)} - Customer ID: {customer_id}")
            
           
            if customer_id not in conversation_lookup:
                print(f"  Warning: No conversation data found for customer {customer_id}")
            
            evaluation = self.evaluate_response_usefulness(interaction, conversation_lookup)
            evaluations.append(evaluation)
        
        return evaluations
    
    def save_evaluations(self, evaluations: List[Dict[str, Any]], output_file: str = "nba_evaluations.json"):
        """
        Save evaluation results to a JSON file.
        
        Args:
            evaluations: List of evaluation results
            output_file: Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(evaluations, file, indent=2, ensure_ascii=False)
            print(f"Evaluations saved to {output_file}")
        except Exception as e:
            print(f"Error saving evaluations: {e}")

