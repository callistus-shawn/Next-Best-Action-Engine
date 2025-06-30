import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import re
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

class NBAWithMBTI:
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
        Extract required features from a conversation including MBTI personality type.
        
        Args:
            conversation: Dictionary containing conversation data
            
        Returns:
            Dictionary with extracted features including MBTI
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
        # Extract MBTI personality type
        mbti_type = conversation.get('mbti_type', 'Unknown')
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
            'mbti_type': mbti_type,
            'conversation_length': conversation_length,
            'chat_history': chat_history,
            'resolved': resolved,
            'resolution_status': resolution_status,
            'customer_has_reply': customer_has_reply
        }
    
    def get_mbti_communication_guidelines(self, mbti_type: str) -> str:
        """
        Get communication guidelines based on MBTI personality type.
        
        Args:
            mbti_type: MBTI personality type (e.g., 'INTP', 'ENFP', etc.)
            
        Returns:
            String with communication guidelines for the personality type
        """
        mbti_guidelines = {
            'INTJ': 'Prefer direct, logical communication. Value efficiency and competence. Avoid emotional appeals.',
            'INTP': 'Appreciate detailed technical explanations. Prefer written communication. Value logical reasoning.',
            'ENTJ': 'Prefer direct, results-oriented communication. Value efficiency and clear action plans.',
            'ENTP': 'Enjoy exploring multiple options. Prefer engaging, dynamic communication. Value innovation.',
            'INFJ': 'Appreciate empathetic, personal communication. Value authenticity and meaningful connections.',
            'INFP': 'Prefer gentle, supportive communication. Value personal values and emotional authenticity.',
            'ENFJ': 'Appreciate warm, encouraging communication. Value harmony and helping others.',
            'ENFP': 'Prefer enthusiastic, creative communication. Value possibilities and personal growth.',
            'ISTJ': 'Prefer clear, structured communication. Value reliability and practical solutions.',
            'ISFJ': 'Appreciate patient, supportive communication. Value tradition and helping others.',
            'ESTJ': 'Prefer direct, organized communication. Value efficiency and clear procedures.',
            'ESFJ': 'Appreciate warm, cooperative communication. Value harmony and practical help.',
            'ISTP': 'Prefer practical, hands-on communication. Value flexibility and immediate solutions.',
            'ISFP': 'Appreciate gentle, artistic communication. Value personal space and authentic experiences.',
            'ESTP': 'Prefer dynamic, action-oriented communication. Value immediate results and flexibility.',
            'ESFP': 'Appreciate enthusiastic, social communication. Value fun and helping others.',
            'Unknown': 'Use standard professional communication guidelines.'
        }
        
        return mbti_guidelines.get(mbti_type, mbti_guidelines['Unknown'])
    
    def determine_next_best_action_llm(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to determine the next best action for customers waiting for replies, incorporating MBTI personality insights.
        
        Args:
            features: Extracted features for a conversation including MBTI
            
        Returns:
            Dictionary with next best action recommendation
        """
        
        if not self.llm:
            return {
                "customer_id": features['customer_id'],
                "channel": "twitter_dm_reply",
                "send_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "message": "Thank you for reaching out. We're here to help resolve your issue.",
                "reasoning": "Default recommendation - LLM not available",
                "issue_status": "pending_customer_reply"
            }
        
        # Get MBTI communication guidelines
        mbti_guidelines = self.get_mbti_communication_guidelines(features['mbti_type'])
        
        # Prepare context for LLM
        context = f"""
Customer Support Analysis with MBTI Personality Insights:
- Customer ID: {features['customer_id']}
- MBTI Personality Type: {features['mbti_type']}
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

Task: Recommend the best next action for this customer who is waiting for a company reply, considering their MBTI personality type for optimal communication.
"""
        
        prompt = f"""
{context}

Based on the customer's issue, sentiment, conversation context, historical behavior patterns, and MBTI personality type, recommend the best next action with the objective-function of maximizing issue resolution while respecting their communication preferences.

Consider:
- Customer sentiment (Lead with empathy, acknowledge frustration)
- Nature of support (Technical issues often need detailed explanations)
- Conversation length (Apologize for extended interaction, escalate proactively)

- MBTI Personality Type: {features['mbti_type']}
  * Communication Guidelines: {mbti_guidelines}
  * Adapt your message tone and approach based on their personality preferences
  * Consider their preferred communication style when choosing channels

- Customer behavior patterns:
  * Most frequent sentiment: {features['most_frequent_sentiment']} - indicates their typical emotional state
  * Most frequent support type: {features['most_frequent_support_type']} - shows their typical issue patterns
  * If customer has consistently negative sentiment, prioritize empathy and escalation
  * If customer frequently has the same support type, consider proactive solutions

Channel selection rules:
- email_reply: conversation ≤ 6 messages for account issues, billing issues, security issues, for detailed explanations, when customer prefers written communication, private issues, or for Introverted/Thinking types
- twitter_dm_reply: conversation ≤ 6 messages for quick issues, non-technical problems, product feedback, customer grievances needing empathy, or for Extraverted/Feeling types
- scheduling_phone_call: For complex issues, urgent disruptions, escalated complaints, if conversation ≥ 7 messages and there is a need to call to resolve, customers with consistently negative sentiment patterns, or for Extraverted types who prefer personal interaction

send_time: 
1-2 hours: Urgent issues, escalated complaints, conversations ≥ 7 messages, customers with consistently negative sentiment
4-6 hours: Standard negative sentiment cases, technical issues
6-8 hours: Simple issues, neutral sentiment

message: Should be empathetic, specific to their issue, provide clear next steps, and adapt to their MBTI communication preferences. It should not have references to the mbti type in the message itself


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
    "reasoning": "string explaining why this channel/time/message is best considering customer patterns and MBTI personality type",
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
            return {
                "customer_id": features['customer_id'],
                "channel": "twitter_dm_reply",
                "send_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "message": "Thank you for reaching out. We're here to help resolve your issue.",
                "reasoning": f"Default recommendation - LLM failed. MBTI type: {features['mbti_type']}",
                "issue_status": "pending_customer_reply"
            }
    
    def process_conversations(self, json_file_path: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process conversations and generate next best actions for customers waiting for replies, incorporating MBTI insights.
        
        Args:
            json_file_path: Path to the JSON file with conversations (must include mbti_type field)
            api_key: API key for LLM features
            
        Returns:
            List of next best action recommendations
        """
       
        with open(json_file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        
        print(f"Loaded {len(conversations)} conversations from {json_file_path}")
        
        # Check if conversations have MBTI data
        conversations_with_mbti = [conv for conv in conversations if conv.get('mbti_type') and conv.get('mbti_type') != 'Unknown']
        print(f"Found {len(conversations_with_mbti)} conversations with MBTI data")
        
        if len(conversations_with_mbti) == 0:
            print("Warning: No conversations found with MBTI data. Please run simple_mbti.py first to add MBTI types.")
            return []
        
        # Extract features including MBTI
        all_features = []
        for conv in conversations_with_mbti:
            features = self.extract_features(conv)
            all_features.append(features)
        
        # Filter customers waiting for replies
        customers_waiting_for_replies = [
            features for features in all_features 
            if features['resolution_status'] == 'waiting_for_company'
        ]
        
        print(f"Found {len(customers_waiting_for_replies)} customers waiting for company replies out of {len(conversations_with_mbti)} conversations with MBTI data")
        
        # Generate next best actions
        recommendations = []
        for features in customers_waiting_for_replies:
            print(f"Processing customer {features['customer_id']} (MBTI: {features['mbti_type']})")
            recommendation = self.determine_next_best_action_llm(features)
            recommendations.append(recommendation)
        
        return recommendations


def nba_mbti():
    """Main function to run NBA with MBTI analysis."""
    input_file = "data/twitter_conversations_with_mbti.json"
    output_file = "data/nba_with_mbti.json"
    api_key = os.getenv("GOOGLE_API_KEY")

    nba = NBAWithMBTI(api_key)  
    
    # Process conversations
    recommendations = nba.process_conversations(input_file, api_key)
    
    # Save recommendations
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(recommendations)} recommendations to {output_file}")
    