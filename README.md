# Riverline-ML-Assignment
### This is Next Best Action (NBA) engine that decides what to do next when a customer’s issue remains open. 
* It ingests raw customer support conversations, deduplicates and structures them, and then uses large language models (LLMs) to tag each conversation with support type, sentiment, resolution status. 
* The pipeline then generates next-best-action recommendations for unresolved cases taking in conversation context. The next best action decides on the channel, send-time, message and reasoning with the objective-function of maximizing issue resolution. The 3 channels are replying back on twitter, replying over email, or scheduling a phone call to a customer. 
* It includes an LLM based evaluation functionality to measure the effectiveness of the company's response

### Project Structure
| File    | Description |
|------------------|-------------|
| `main.py`        | Entry point for the whole pipeline |
| `output.csv`        | Result CSV for 1000 customers |

---

###  `data/` – Datasets and Intermediate Files

| File                                | Description |
|-------------------------------------|-------------|
| `twcs.csv`                          | Raw Twitter Customer Support dataset. |
| `twitter_conversations_raw.json`   | Parsed and ingested Twitter conversations. |
| `twitter_conversations_tagged.json`   | Conversations tagged with nature of support, sentiment, resolution_stautus... |
| `nba.json`                          | NBA (Next Best Action) recommendations generated. |
| `nba_evaluations.json`             | LLM-generated qualitative evaluations of NBA recommendations. |
| `twitter_conversations_with_mbti.json` | Conversations tagged with MBTI personality types. |
| `nba_with_mbti.json`               | NBA recommendations personalized using MBTI types. |
| `nba_mbti_comparison.json`         | Comparative analysis of NBA vs NBA+MBTI responses. |


---

###  `pipeline/` – Core Data & NBA Pipeline

| File                | Description |
|---------------------|-------------|
| `data_pipeline.py`  | Parses raw data, deduplicates tweets, and constructs structured conversation threads. |
| `tagging.py`        | Adds LLM-based tags for support type, sentiment and more |
| `nba.py`            | Recommends Next Best Action (NBA) |
| `nba_evaluation.py` | Uses LLM to judge the quality of the NBA recommendations. |
| `export_to_csv.py`  | Converts structured JSON data into flat CSV files for output. |

---

###  `mbti/` – MBTI Tagging & Personality-Aware NBA

| File              | Description |
|-------------------|-------------|
| `mbti_tagging.py` | Classifies MBTI personality types of users using a fine-tuned BERT model. |
| `nba_mbti.py`     | Generates NBA recommendations enhanced with MBTI personality insights. |
| `nba_mbti_eval.py`| Evaluates whether MBTI-aware NBAs offer improvements over standard ones. |

---


### `notebooks/` – EDA, Training BERT Model

| File                    | Description |
|--------------------------|-------------|
| `mbti_classifier.ipynb`  | Notebook for training the MBTI classifier. |
| `Visualization.ipynb`    | Charts and graphs to explore conversation structures and tagging distributions. |

---
