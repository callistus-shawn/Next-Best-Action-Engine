import pandas as pd
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

def convert_twitter_csv_to_json(csv_file_path: str, output_file_path: Optional[str] = None) -> List[Dict]:
    """
    Convert Twitter customer service CSV data to structured JSON format with deduplication.
    Can append to existing JSON file and handle thread connections.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to JSON file (both for reading existing and writing output)
    
    Returns:
        List of conversation dictionaries
    """
    
    # Load existing conversations if provided
    existing_convo = {}
    existing_tweet_ids = set()
    if output_file_path:
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for conv in existing_data:
                    existing_convo[conv['primary_tweet_id']] = conv
                    for msg in conv.get('chat_history', []):
                        existing_tweet_ids.add(msg['response']['tweet_id'])

                print(f"Loaded {len(existing_data)} existing conversations")

        except FileNotFoundError:
            print(f"No existing JSON file found at {output_file_path}, starting fresh")
        except Exception as e:
            print(f"Error loading existing JSON file: {e}, starting fresh")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path, dtype={'tweet_id': str, 'in_response_to_tweet_id': str, 'response_tweet_id': str})
        df=df[:4000]
        print(f"Loaded {len(df)} tweets from CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    # Clean column names and handle missing values

    df.columns = df.columns.str.strip()
    df['tweet_id'] = df['tweet_id'].astype(str).str.strip()
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna('').astype(str).str.strip()
    df['response_tweet_id'] = df['response_tweet_id'].fillna('').astype(str).str.strip()
    
    # Remove duplicate tweets based on tweet_id

    initial_count = len(df)
    df = df.drop_duplicates(subset=['tweet_id'], keep='first')
    final_count = len(df)
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate tweets")
    
    # Remove rows with empty tweet_id

    df = df[df['tweet_id'] != '']
    print(f"After cleaning: {len(df)} unique tweets")

    def find_conversation_root(tweet_id: str, df_tweets: pd.DataFrame) -> Optional[str]:
        """Find the root tweet of a conversation thread."""
        if tweet_id == '' or pd.isna(tweet_id):
            return None
            
        tweet_row = df_tweets[df_tweets['tweet_id'] == str(tweet_id)]
        if tweet_row.empty:
            return None
            
        in_response_to = tweet_row.iloc[0]['in_response_to_tweet_id']
        
        if in_response_to == '' or pd.isna(in_response_to):
            return str(tweet_id)
        else:
            return find_conversation_root(in_response_to, df_tweets)
    
    new_convo = {}
    
    # Identify all root tweets
    for _, tweet in df.iterrows():
        tweet_id_str = str(tweet['tweet_id'])
        
        
        if tweet_id_str in existing_tweet_ids:
            continue
            
        root_id = find_conversation_root(tweet_id_str, df)
        
        if root_id:
            # Check if this root matches any existing tail_id
            matching_conversation = None
            for conv_id, conversation in existing_convo.items():
                if conversation['tail_id'] == root_id:
                    matching_conversation = conv_id
                    break
            
            if matching_conversation:
                # This is a continuation of an existing conversation
                # We'll mark it as unprocessed later if new messages are appended
                pass

            elif root_id not in new_convo:
                # This is a new conversation
                root_tweet = df[df['tweet_id'] == root_id]
                if not root_tweet.empty:
                    new_convo[root_id] = {
                        'primary_tweet_id': root_id,
                        'primary_tweet': root_tweet.iloc[0]['text'],
                        'tail_id': root_id,
                        'customer_id': '',
                        'company_id': '',
                        'chat_history': [],
                        'processed': False  # Mark new conversations as unprocessed
                    }
    
    # Build conversation threads
    for _, tweet in df.iterrows():
        tweet_id_str = str(tweet['tweet_id'])
        
       
        if tweet_id_str in existing_tweet_ids:
            continue
            
        root_id = find_conversation_root(tweet_id_str, df)
        
        # Check if this belongs to an existing conversation
        matching_conversation = None
        for conv_id, conversation in existing_convo.items():
            if conversation['tail_id'] == root_id:
                matching_conversation = conv_id
                break
        
        if matching_conversation:
         
            conversation = existing_convo[matching_conversation]
            
            # Determine if this is a customer or company response
            is_customer = tweet['inbound'] == True or str(tweet['inbound']).lower() == 'true'
            response_type = 'Customer' if is_customer else 'Company'
            
        
            message_entry = {
                'response_type': response_type,
                'response': {
                    'tweet_id': tweet_id_str,
                    'text': tweet['text'],
                    'created_at': tweet['created_at']
                }
            }
            
           
            if response_type == 'Customer':
                message_entry['customer_id'] = tweet.get('author_id', '')
            
          
            existing_tweet_ids_in_conv = [msg['response']['tweet_id'] for msg in conversation['chat_history']]
            if tweet_id_str not in existing_tweet_ids_in_conv:
                conversation['chat_history'].append(message_entry)
                existing_tweet_ids.add(tweet_id_str)
                # Mark as unprocessed if new message is appended
                conversation['processed'] = False
                
        elif root_id and root_id in new_convo:
            # Add to new conversation
            conversation = new_convo[root_id]
            
            # Determine if this is a customer or company response
            is_customer = tweet['inbound'] == True or str(tweet['inbound']).lower() == 'true'
            response_type = 'Customer' if is_customer else 'Company'
           
            message_entry = {
                'response_type': response_type,
                'response': {
                    'tweet_id': tweet_id_str,
                    'text': tweet['text'],
                    'created_at': tweet['created_at']
                }
            }
            
          
            if response_type == 'Customer':
                message_entry['customer_id'] = tweet.get('author_id', '')
            
           
            existing_tweet_ids_in_conv = [msg['response']['tweet_id'] for msg in conversation['chat_history']]
            if tweet_id_str not in existing_tweet_ids_in_conv:
                conversation['chat_history'].append(message_entry)
                # processed is already set to False for new conversations
    
    # Process new conversations
    print(f"\nProcessing {len(new_convo)} new conversations...")
    for i, (conv_id, conversation) in enumerate(new_convo.items()):
        try:
            conversation['chat_history'].sort(
                key=lambda x: datetime.strptime(x['response']['created_at'], '%a %b %d %H:%M:%S %z %Y')
            )
        except:
            pass
        
        # Add tail_id
        if conversation['chat_history']:
            conversation['tail_id'] = conversation['chat_history'][-1]['response']['tweet_id']
        else:
            conversation['tail_id'] = conversation['primary_tweet_id']
        
        customer_ids = set()
        company_ids = set()
        
        for msg in conversation['chat_history']:
            if msg['response_type'] == 'Customer' and 'customer_id' in msg:
                customer_ids.add(msg['customer_id'])
            elif msg['response_type'] == 'Company':
                # For company messages, get author_id from the original tweet
                tweet_id = msg['response']['tweet_id']
                tweet_row = df[df['tweet_id'] == tweet_id]
                if not tweet_row.empty:
                    company_id = tweet_row.iloc[0].get('author_id', '')
                    if company_id:
                        company_ids.add(company_id)
        
        conversation['customer_id'] = list(customer_ids)[0] if customer_ids else ''
        conversation['company_id'] = list(company_ids)[0] if company_ids else ''
        
        for msg in conversation['chat_history']:
            if 'customer_id' in msg:
                del msg['customer_id']
    
 
    updated_existing_count = 0
    for conv_id, conversation in existing_convo.items():
        if len(conversation['chat_history']) > 0:
            try:
                conversation['chat_history'].sort(
                    key=lambda x: datetime.strptime(x['response']['created_at'], '%a %b %d %H:%M:%S %z %Y')
                )
                updated_existing_count += 1
            except:
              
                pass
        
        # Update tail_id
        if conversation['chat_history']:
            conversation['tail_id'] = conversation['chat_history'][-1]['response']['tweet_id']
        
   
        customer_ids = set()
        company_ids = set()
        
        for msg in conversation['chat_history']:
            if msg['response_type'] == 'Customer' and 'customer_id' in msg:
                customer_ids.add(msg['customer_id'])
            elif msg['response_type'] == 'Company':
           
                tweet_id = msg['response']['tweet_id']
                tweet_row = df[df['tweet_id'] == tweet_id]
                if not tweet_row.empty:
                    company_id = tweet_row.iloc[0].get('author_id', '')
                    if company_id:
                        company_ids.add(company_id)
        
        # Update customer_id and company_id to conversation
        if customer_ids:
            conversation['customer_id'] = list(customer_ids)[0]
        if company_ids:
            conversation['company_id'] = list(company_ids)[0]
        
      
        for msg in conversation['chat_history']:
            if 'customer_id' in msg:
                del msg['customer_id']
    
    # Combine existing and new conversations
    result = list(existing_convo.values()) + list(new_convo.values())
    
    # Print summary
    print(f"\nConversion Summary:")
    print(f"Existing conversations: {len(existing_convo)}")
    print(f"New conversations: {len(new_convo)}")
    print(f"Updated existing conversations: {updated_existing_count}")
    print(f"Total conversations: {len(result)}")
    

    
 
    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nJSON data saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
    
    return result

