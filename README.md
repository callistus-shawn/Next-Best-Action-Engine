# Riverline-ML-Assignment
#### This is Next Best Action (NBA) engine that decides what to do next when a customer’s issue remains open. 
* It ingests raw customer support conversations, deduplicates and structures them, and then uses large language models (LLMs) to tag each conversation with support type, sentiment, resolution status. 
The pipeline then generates next-best-action recommendations for unresolved cases taking in conversation context. The next best action decides on the channel, send-time, message and reasoning with the objective-function of maximizing issue resolution. The 3 channels are replying back on twitter, replying over email, or scheduling a phone call to a customer. 
It includes an LLM based evaluation functionality to measure the effectiveness of the company's response
