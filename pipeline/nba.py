import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import re
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

class NBA:
    def __init__(self, api_key: str):
        """Initialize the analyzer with optional API key for LLM features."""
        self.llm = None
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            except Exception as e:
                print(f"Failed to initialize LLM: {e}")
        else:
            print("No API key provided. LLM features will be disabled.")
    
    def extract_features(self, conversation: Dict) -> Dict[str, Any]:
        """
        Extract required features from a conversation.
        
        Args:
            conversation: Dictionary containing conversation data
            
        Returns:
            Dictionary with extracted features
        """
       
        primary_tweet_id = conversation.get('primary_tweet_id', '')
        primary_tweet = conversation.get('primary_tweet', '')
        chat_history = conversation.get('chat_history', [])
        resolved = conversation.get('resolved', False)
        resolution_status = conversation.get('resolution_status', 'waiting_for_company')
        nature_of_support = conversation.get('nature_of_support', '')
        customer_sentiment = conversation.get('customer_sentiment', '')
        customer_id = conversation.get('customer_id', '')
        # Extract customer pattern tags
        most_frequent_sentiment = conversation.get('most_frequent_sentiment', '')
        most_frequent_support_type = conversation.get('most_frequent_support_type', '')
        # Calculate conversation length
        conversation_length = len(chat_history)
        
       
        company_responses = [msg for msg in chat_history if msg['response_type'] == 'Company']
        customer_has_reply = len(company_responses) > 0
        
        return {
            'customer_id': customer_id,
            'primary_tweet': primary_tweet,
            'nature_of_support': nature_of_support,
            'sentiment': customer_sentiment,
            'most_frequent_sentiment': most_frequent_sentiment,
            'most_frequent_support_type': most_frequent_support_type,
            'conversation_length': conversation_length,
            'chat_history': chat_history,
            'resolved': resolved,
            'resolution_status': resolution_status,
            'customer_has_reply': customer_has_reply
        }
    
    def determine_next_best_action_llm(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to determine the next best action for customers waiting for replies.
        
        Args:
            features: Extracted features for a conversation
            
        Returns:
            Dictionary with next best action recommendation
        """
        
        return {
            "customer_id": features['customer_id'],
            "channel": "twitter_dm_reply",
            "send_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "message": "Thank you for reaching out. We're here to help resolve your issue.",
            "reasoning": "Default recommendation - LLM not available",
            "issue_status": "pending_customer_reply"
        }
        
        # Prepare context for LLM
        context = f"""
Customer Support Analysis:
- Customer ID: {features['customer_id']}
- Primary Tweet: {features['primary_tweet']}
- Nature of Support: {features['nature_of_support']}
- Sentiment: {features['sentiment']}
- Most Frequent Sentiment: {features['most_frequent_sentiment']}
- Most Frequent Support Type: {features['most_frequent_support_type']}
- Conversation Length: {features['conversation_length']} messages

Chat History:
"""
        
        # Add chat history to context
        for i, message in enumerate(features['chat_history']):
            response_type = message['response_type']
            content = message['response']['text']
            created_at = message['response']['created_at']
            
            
            if response_type == 'Customer' and 'customer_id' in message:
                context += f"{i+1}. [{response_type} - {message['customer_id']}] {created_at}: {content}\n"
            else:
                context += f"{i+1}. [{response_type}] {created_at}: {content}\n"

        context += f"""
Available Channels:
1. twitter_dm_reply - Direct message on Twitter
2. scheduling_phone_call - Schedule a phone call
3. email_reply - Send an email response

Task: Recommend the best next action for this customer who is waiting for a company reply.
"""
        
        prompt = f"""
{context}

Based on the customer's issue, sentiment, conversation context, and historical behavior patterns, recommend the best next action with the objective-function of
maximizing issue resolution.

Consider:
- Customer sentiment (Lead with empathy, acknowledge frustration)
- Nature of support (Technical issues often need detailed explanations)
- Conversation length (Apologize for extended interaction, escalate proactively)
- Customer behavior patterns:
  * Most frequent sentiment: {features['most_frequent_sentiment']} - indicates their typical emotional state
  * Most frequent support type: {features['most_frequent_support_type']} - shows their typical issue patterns
  * If customer has consistently negative sentiment, prioritize empathy and escalation
  * If customer frequently has the same support type, consider proactive solutions. Example: If they are frequently raising Technical Issues, they might not be a techinically sound person. Deal with them accordingly with patience.

Channel selection rules:
- email_reply: conversation ≤ 6 messages for account issues, billing issues, security issues, for detailed explanations, when customer prefers written communication, private issues
- twitter_dm_reply: conversation ≤ 6 messages for quick issues, non-technical problems, product feedback, and customer grievances needing empathy
- scheduling_phone_call: For complex issues, urgent disruptions, escalated complaints, if conversation ≥ 7 messages and there is a need to call to resolve, or customers with consistently negative sentiment patterns

send_time: 
1-2 hours: Urgent issues, escalated complaints, conversations ≥ 7 messages, customers with consistently negative sentiment
4-6 hours: Standard negative sentiment cases, technical issues
6-8 hours: Simple issues, neutral sentiment

- message: Should be empathetic, specific to their issue, and provide clear next steps

After the message is sent, the issue status will be updated to the following options:
Issue Status Options:
- "resolved": if the issue is completely resolved after this action
- "pending_customer_reply": if the customer needs to provide more information or take action
- "escalated": if the issue requires escalation to higher level support or specialist team
- "scheduled_followup": If there is a scheduled follow-up call or meeting
- "waiting_for_third_party": if the issue depends on external party (vendor, partner, etc.)


Respond in this exact JSON format:
{{
    "customer_id": "{features['customer_id']}",
    "channel": "twitter_dm_reply|scheduling_phone_call|email_reply",
    "send_time": "2025-01-XXTXX:XX:00Z",
    "message": "string",
    "reasoning": "string explaining why this channel/time/message is best considering customer patterns",
    "issue_status": "resolved|pending_customer_reply|escalated|in_progress|scheduled_followup|waiting_for_third_party|pending_verification"
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
            print(f"LLM analysis failed: {e}")
         
    
    def process_conversations(self, json_file_path: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process conversations and generate next best actions for customers waiting for replies.
        Only processes conversations where processed == False.
        """
       
        with open(json_file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Only process unprocessed conversations
        conversations = [c for c in conversations if not c.get('processed', False)]
        print(f"Loaded {len(conversations)} unprocessed conversations from {json_file_path}")
        
        all_features = []
        for conv in conversations:
            features = self.extract_features(conv)
            all_features.append(features)
        
        customers_waiting_for_replies = [
            features for features in all_features 
            if features['resolution_status'] == 'waiting_for_company'
        ]
        
        print(f"Found {len(customers_waiting_for_replies)} customers waiting for company replies out of {len(conversations)} total conversations")
        
        # Track all loaded conversations' primary_tweet_id
        processed_ids = {c.get('primary_tweet_id') for c in conversations}

        # Generate next best actions
        recommendations = []
        for features in customers_waiting_for_replies:
            recommendation = self.determine_next_best_action_llm(features)
            recommendations.append(recommendation)

        # Mark processed in the raw input JSON
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                all_conversations = json.load(f)
            for conv in all_conversations:
                if not conv.get('processed', False) and conv.get('primary_tweet_id') in processed_ids:
                    conv['processed'] = True
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_conversations, f, indent=2, ensure_ascii=False)
            print(f"Marked {len(processed_ids)} conversations as processed in the input JSON.")
        except Exception as e:
            print(f"Error updating processed status in input JSON: {e}")

        return recommendations

