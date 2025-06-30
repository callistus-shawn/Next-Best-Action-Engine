import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import re
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

class NBAMBTIEvaluation:
    def __init__(self, api_key: str):
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
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Loaded {len(data)} NBA recommendations from {json_file_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def load_tagged_conversations(self, json_file_path: str) -> Dict[str, Dict[str, Any]]:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                conversations = json.load(file)
            conversation_lookup = {}
            for conv in conversations:
                customer_id = conv.get('customer_id')
                if customer_id:
                    conversation_lookup[customer_id] = conv
            print(f"Loaded {len(conversations)} tagged conversations")
            return conversation_lookup
        except Exception as e:
            print(f"Error loading tagged conversations: {e}")
            return {}

    def format_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
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
        customer_id = interaction.get('customer_id', '')
        conversation_data = conversation_lookup.get(customer_id, {})
        chat_history = conversation_data.get('chat_history', [])
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

As a customer service evaluation expert, analyze the company's response and reasoning and determine if it was useful from a customer's perspective, considering the full conversation history.

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
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"LLM evaluation failed for customer {customer_id}: {e}")
            return {
                "customer_id": customer_id,
                "usefulness_score": 0,
                "evaluation": f"LLM evaluation failed: {e}"
            }

    def compare_nba_and_mbti(self, nba_file: str, mbti_file: str, tagged_file: str, output_file: str = "nba_mbti_comparison.json"):
        nba_data = self.load_nba_data(nba_file)
        mbti_data = self.load_nba_data(mbti_file)
        conversation_lookup = self.load_tagged_conversations(tagged_file)
        
        nba_by_id = {item['customer_id']: item for item in nba_data}
        mbti_by_id = {item['customer_id']: item for item in mbti_data}
        all_customers = set(nba_by_id.keys()) | set(mbti_by_id.keys())
        results = []
       
        for customer_id in all_customers:
            nba_rec = nba_by_id.get(customer_id)
            mbti_rec = mbti_by_id.get(customer_id)
            if not nba_rec or not mbti_rec:
                continue
            nba_eval = self.evaluate_response_usefulness(nba_rec, conversation_lookup)
            mbti_eval = self.evaluate_response_usefulness(mbti_rec, conversation_lookup)
            result = {
                "customer_id": customer_id,
                "nba_score": nba_eval.get("usefulness_score", 0),
                "nba_eval": nba_eval.get("evaluation", ""),
                "mbti_score": mbti_eval.get("usefulness_score", 0),
                "mbti_eval": mbti_eval.get("evaluation", "")
            }
           
            results.append(result)
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        

def mbti_eval():
    api_key = os.getenv("GOOGLE_API_KEY")
    nba_file = "data/nba.json"
    mbti_file = "data/nba_with_mbti.json"
    tagged_file = "data/twitter_conversations_with_mbti.json"
    output_file = "data/nba_mbti_comparison.json"
    evaluator = NBAMBTIEvaluation(api_key)
    evaluator.compare_nba_and_mbti(nba_file, mbti_file, tagged_file, output_file)
