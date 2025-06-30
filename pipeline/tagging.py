import json
import re
from typing import List, Dict, Any
import os
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

def classify_support_nature_llm(primary_tweet: str, chat_history: List[Dict]) -> str:
    """
    Use LLM to classify the nature of support request.
    """

   
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        print("Warning: GOOGLE_API_KEY not found in environment variables. Using default classification.")
        return "Technical Issue (Simple / Minor)"
    
   
    context_messages = [primary_tweet]
    customer_messages = [msg for msg in chat_history if msg['response_type'] == 'Customer']
    context_messages.extend([msg['response']['text'] for msg in customer_messages])
    conversation_context = "\n".join(context_messages)
    
    prompt = f"""
Analyze this customer service conversation and classify the nature of support request into ONE of these categories:
1. Technical Issue (Simple / Minor)
Minor bugs, product/app glitches, or usage questions that don't block overall functionality.

Examples:
    "@AppleSupport causing the reply to be disregarded and the tapped notification under the keyboard is opened😡😡😡"
    "Spotify Premium skipping through songs constantly on android tablet & bluetooth speaker. Tried everything!"
    "So the new update does not let me listen to music and go on whatsapp at the same time?!?"

2. Account or Login Issues
Problems related to account access, login failures, verification, or password resets.
Examples:
    "@AppleSupport I need a new code for my I-store. I haven't recd any but msg is too many sent. Help!"
    "Not receiving the 2FA code on my new device." (hypothetical)

3. Billing or Refund Request
Payment problems, unauthorized charges, refund demands, or subscription issues.
Examples:
    "@76803 I service last 40 min sprint can I get a refund"
    "Why was I billed again after cancellation?" (hypothetical)

4. Escalated Complaint
Highly dissatisfied tone, demand for escalation or manager, repeated failure, or public outrage.
Examples:
    "I'm so angry with you guys. I've been waiting for 2 hours and you're not helping me. I'm going to cancel my subscription."
    "This is the 5th time I've contacted you about the same issue. Give me someone competent."
    "I want to talk to a manager."

5. Product Feedback
General praise, feature requests, or improvement suggestions — not necessarily a support issue.
Examples:
    "I wish Amazon had an option of where I can just get it shipped to the UPS store so I can avoid a lot of the struggle."
    "Would be great if you could allow split-screen multitasking."

6. Urgent Service Disruption
Critical service failure or outage (e.g. no internet, website down, flight canceled).
Examples:
    "My internet is down and xfinity talkin about 24–72 hours... y'all have the game messed up."
    "Hey @Tesco, your website's broken. It's telling me there are no delivery slots for the next 3 weeks."

7. Customer Grievance
Expression of dissatisfaction, sarcasm, or emotional frustration — without a clear technical or billing issue.
Examples:
    "@Tesco Maybe hire colleagues who can see?"
    "You've paralysed my phone with your update @76099 grrrrrrrrrr"
    "This update sux I hate it fix it bye"

8. Order or Delivery Problem 
Support requests where the customer is complaining about a wrong, missing, or incomplete order, or delivery issues
Examples:
    @ChipotleTweets messed up today and didn't give me my $3 burrito although I was dressed up 😭
    @marksandspencer y do they charge you for a meat bag... If you buy meat you r required to supply customers a free bag.
    no Diet Coke and a literal bone this Boorito was extra spooky!

9. Other
Anything unrelated to the above — jokes, off-topic, marketing comments, or ambiguous sarcasm.
Examples:

    "I guess this means free cable for the neighborhood."
    "Just curious….will I get the companion pass the exact moment I get enough points?"

Customer conversation:
{conversation_context}

Respond with ONLY the category name (e.g., "Technical Issue (Simple / Minor)").
"""

    try:
        google_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0
        )
        response = google_llm.invoke(prompt)
        
        classification = response.content.strip()

        # Clean the classification by removing numbers and dots
        cleaned_classification = re.sub(r'^\d+\.\s*', '', classification).strip()

        valid_categories = [
            "Technical Issue (Simple / Minor)",
            "Account or Login Issues", 
            "Billing or Refund Request",
            "Escalated Complaint",
            "Product Feedback",
            "Urgent Service Disruption",
            "Customer Grievance",
            "Order or Delivery Problem",
            "Other"
        ]
        
        if cleaned_classification not in valid_categories:
            print(f"Unexpected classification from Gemini: {classification}")
        
        return cleaned_classification
    
    except Exception as e:
        print(f"Gemini classification failed: {e}")
        return "Technical Issue (Simple / Minor)" 


def analyze_sentiment_llm(primary_tweet: str, chat_history: List[Dict]) -> str:
    """
    Use LLM to analyze the overall sentiment of the customer conversation.
    """

   
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        print("Warning: GOOGLE_API_KEY not found in environment variables. Using default sentiment analysis.")
        return "Neutral"
    
   
    context_messages = [primary_tweet]
    customer_messages = [msg for msg in chat_history if msg['response_type'] == 'Customer']
    context_messages.extend([msg['response']['text'] for msg in customer_messages])
    conversation_context = "\n".join(context_messages)
    
    prompt = f"""
Analyze the overall sentiment of this customer in their conversation with customer service.

Consider the customer's tone, language, satisfaction level, and emotional state throughout the conversation.

Customer messages:
{conversation_context}

Classify the overall customer sentiment as ONE of these categories:
- Positive: Customer is satisfied, pleased, grateful, or positive
- Negative: Customer is frustrated, angry, disappointed, upset, or negative
- Neutral: Customer is matter-of-fact, professional, or shows mixed/unclear emotions

Respond with ONLY one word: "Positive", "Negative", or "Neutral".
"""

    try:
        google_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0
        )
        response = google_llm.invoke(prompt)
        
        sentiment = response.content.strip()
                
        return sentiment
    
    except Exception as e:
        print(f"Gemini sentiment analysis failed: {e}")
        return "Neutral"  # Default fallback


def determine_resolution_status_llm(chat_history: List[Dict]) -> str:
    """
    Use LLM to determine the resolution status of a conversation.
    """
    if not chat_history:
        return "waiting_for_company"

   
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        print("Warning: GOOGLE_API_KEY not found in environment variables. Using default resolution status analysis.")
    
    
    conversation_text = []
    for msg in chat_history:
        speaker = "Customer" if msg['response_type'] == 'Customer' else "Company"
        text = msg['response']['text']
        conversation_text.append(f"{speaker}: {text}")
    
    conversation_context = "\n".join(conversation_text)
    
    prompt = f"""
Analyze this customer service conversation and determine its current status.

Conversation:
{conversation_context}

Classify the conversation status into ONE of these categories:

1. "resolved" - The issue has been successfully resolved. Look for:
   - Customer expressing satisfaction, thanks, or confirmation that the problem is fixed
   - Company providing final confirmation or closure
   - Both parties agreeing the issue is resolved
   - Positive indicators like "thanks", "solved", "working", "great", "perfect"

2. "waiting_for_customer" - The company sent the LAST message and is waiting for customer response. Look for:
   - Company asking questions or requesting information
   - Company providing instructions and waiting for customer to follow up
   - Company asking if the solution worked
   - Company offering help but customer hasn't responded yet

3. "waiting_for_company" - The customer sent the LAST message and is waiting for company response. Look for:
   - Customer asking questions or requesting help
   - Customer providing information and waiting for company to act
   - Customer expressing frustration without resolution
   - Customer making requests that haven't been addressed

Respond with ONLY one word: "resolved", "waiting_for_customer", or "waiting_for_company"
"""

    try:
        google_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0
        )
        response = google_llm.invoke(prompt)
        
        status = response.content.strip()
        
        return status
    
    except Exception as e:
        print(f"LLM resolution status analysis failed: {e}")
 


def calculate_customer_patterns(conversations: List[Dict]) -> Dict[str, Dict[str, str]]:
    """
    Calculate most frequent sentiment and support type for each customer.
    
    Args:
        conversations: List of conversation dictionaries
        
    Returns:
        Dictionary mapping customer_id to their most frequent patterns
    """
    # Group conversations by customer_id
    customer_conversations = {}
    for conv in conversations:
        customer_id = conv.get('customer_id')
        if customer_id:
            if customer_id not in customer_conversations:
                customer_conversations[customer_id] = []
            customer_conversations[customer_id].append(conv)
    
    # Calculate patterns for each customer
    customer_patterns = {}
    
    for customer_id, convs in customer_conversations.items():
        # Count sentiments
        sentiments = [conv.get('sentiment', 'Neutral') for conv in convs]
        sentiment_counts = Counter(sentiments)
        most_frequent_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "Neutral"
        
        # Count support types
        support_types = [conv.get('nature_of_support', 'Technical Issue (Simple / Minor)') for conv in convs]
        support_counts = Counter(support_types)
        most_frequent_support_type = support_counts.most_common(1)[0][0] if support_counts else "Technical Issue (Simple / Minor)"
        
        customer_patterns[customer_id] = {
            'most_frequent_sentiment': most_frequent_sentiment,
            'most_frequent_support_type': most_frequent_support_type
        }
    
    print(f"Calculated patterns for {len(customer_patterns)} unique customers")
    return customer_patterns


def tag_conversations(json_file_path: str, output_file_path: str = None) -> List[Dict]:
    """
    Tag conversations with support nature, sentiment, and resolution status using LLM.
    Only processes conversations with 'processed': False and appends them to the output file.
    """
    # Load all conversations
    with open(json_file_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations for tagging")

    # Filter for unprocessed conversations
    unprocessed = [c for c in conversations if not c.get('processed', False)]
    print(f"Found {len(unprocessed)} unprocessed conversations to tag")

    if not unprocessed:
        print("No unprocessed conversations to tag.")
        return []

    # Tag only unprocessed conversations
    print(f"\nTagging conversations with LLM analysis...")
    for i, conversation in enumerate(unprocessed):
        # Classify support nature using LLM
        conversation['nature_of_support'] = classify_support_nature_llm(
            conversation['primary_tweet'], 
            conversation['chat_history']
        )
        # Analyze customer sentiment using LLM
        conversation['sentiment'] = analyze_sentiment_llm(
            conversation['primary_tweet'], 
            conversation['chat_history']
        )
        # Determine resolution status using LLM
        resolved = determine_resolution_status_llm(conversation['chat_history'])
        conversation['resolved'] = (resolved == "resolved")
        conversation['resolution_status'] = resolved
        # Do NOT mark as processed here

    # Calculate customer patterns for just the new tagged conversations
    print(f"\nCalculating customer patterns...")
    customer_patterns = calculate_customer_patterns(unprocessed)
    # Add customer patterns to each conversation
    print(f"Adding customer patterns to conversations...")
    for conversation in unprocessed:
        customer_id = conversation.get('customer_id')
        if customer_id and customer_id in customer_patterns:
            patterns = customer_patterns[customer_id]
            conversation['most_frequent_sentiment'] = patterns['most_frequent_sentiment']
            conversation['most_frequent_support_type'] = patterns['most_frequent_support_type']
        else:
            conversation['most_frequent_sentiment'] = conversation.get('sentiment', 'Neutral')
            conversation['most_frequent_support_type'] = conversation.get('nature_of_support', 'Technical Issue (Simple / Minor)')

    # Print summary
    print(f"\nTagging Summary:")
    print(f"Total newly tagged conversations: {len(unprocessed)}")
    print(f"Resolved conversations: {sum(1 for c in unprocessed if c['resolved'])}")
    print(f"Resolution status breakdown:")
    status_categories = ["resolved", "waiting_for_customer", "waiting_for_company"]
    for status in status_categories:
        count = sum(1 for c in unprocessed if c['resolution_status'] == status)
        print(f"  {status}: {count}")
    print(f"Customer sentiment breakdown:")
    sentiment_categories = ["Positive", "Negative", "Neutral"]
    for sentiment in sentiment_categories:
        count = sum(1 for c in unprocessed if c['sentiment'] == sentiment)
        print(f"  {sentiment}: {count}")

    # Append/overwrite to output file
    if output_file_path:
        try:
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            else:
                existing = []
           
            existing_by_id = {c['primary_tweet_id']: c for c in existing}
            for c in unprocessed:
                c['processed'] = False  
                existing_by_id[c['primary_tweet_id']] = c  

            combined = list(existing_by_id.values())
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving JSON file: {e}")

    # Mark processed in the raw input JSON
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_conversations = json.load(f)
        

        tagged_ids = {c['primary_tweet_id'] for c in unprocessed}
        for conv in all_conversations:
            if conv.get('primary_tweet_id') in tagged_ids:
                conv['processed'] = True
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2, ensure_ascii=False)

        print(f"Marked {len(tagged_ids)} conversations as processed in the raw input JSON.")
    except Exception as e:
        print(f"Error updating processed status in input JSON: {e}")

    return unprocessed

