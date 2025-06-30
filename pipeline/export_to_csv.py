import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

def export_nba_to_csv(json_file_path: str, output_csv_path: str, conversations_file_path: str = ""):
    """
    Export Next Best Action JSON data to CSV format.
    
    Args:
        json_file_path: Path to the NBA JSON file
        output_csv_path: Path for the output CSV file
        conversations_file_path: Path to the conversations JSON file (optional, for chat history)
    """
    
    # Load NBA data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            nba_data = json.load(f)
        print(f"Loaded {len(nba_data)} NBA recommendations from {json_file_path}")
    except Exception as e:
        print(f"Error loading NBA JSON file: {e}")
        return
    
   
    conversations_data = {}
    if conversations_file_path:
        try:
            with open(conversations_file_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
           
            for conv in conversations:
                customer_id = conv.get('customer_id')
                if customer_id:
                    conversations_data[customer_id] = conv
            print(f"Loaded {len(conversations)} conversations for chat history")
            print(f"Mapped {len(conversations_data)} conversations by customer_id")
        except Exception as e:
            print(f"Warning: Could not load conversations file: {e}")
    
   
    csv_data = []
    
    for nba_item in nba_data:
        customer_id = nba_item.get('customer_id', '')
        channel = nba_item.get('channel', '')
        send_time = nba_item.get('send_time', '')
        message = nba_item.get('message', '')
        reasoning = nba_item.get('reasoning', '')
        issue_status = nba_item.get('issue_status', '')
        
        # Get chat history for this customer
        chat_history_with_new_message = ""
        if customer_id in conversations_data:
            conversation = conversations_data[customer_id]
            chat_history = conversation.get('chat_history', [])
            
           
            chat_lines = []
            for i, msg in enumerate(chat_history):
                response_type = msg['response_type']
                content = msg['response']['text']
                created_at = msg['response']['created_at']
                
                chat_lines.append(f"[{response_type}]: {content}")
            
           
            if chat_lines:
                chat_lines.append(f"[Company - RECOMMENDED]: {message}")
            
            chat_history_with_new_message = "\n".join(chat_lines)
        else:
            print(f"Warning: No conversation found for customer_id: {customer_id}")
        
        # Add row to CSV data
        csv_data.append({
            'customer_id': customer_id,
            'channel': channel,
            'send_time': send_time,
            'message': message,
            'reasoning': reasoning,
            'issue_status': issue_status,
            'chat_log': chat_history_with_new_message
        })
   
   
    df = pd.DataFrame(csv_data)
    
    try:
        df.to_csv(output_csv_path, mode='w', index=False, encoding='utf-8')
        print(f"Successfully exported {len(csv_data)} records to {output_csv_path}")
      
    except Exception as e:
        print(f"Error saving CSV file: {e}")

