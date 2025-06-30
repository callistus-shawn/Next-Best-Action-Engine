
from pipeline.data_pipeline import convert_twitter_csv_to_json
from pipeline.tagging import tag_conversations
from pipeline.nba import NBA
from pipeline.nba_evaluation import NBAEvaluation
from pipeline.export_to_csv import export_nba_to_csv
from mbti.mbti_tagging import mbti_tag
from mbti.nba_mbti_eval import mbti_eval
from mbti.nba_mbti import nba_mbti 


import json
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

mbti=False
## Injestion  

injestion_input_file = "data/twcs.csv"
injestion_output_file = "data/twitter_conversations_raw.json"

conversations = convert_twitter_csv_to_json(injestion_input_file, injestion_output_file)
    
if conversations:
    print(f"\nFiles created:")
    print(f"- {injestion_output_file} (raw conversation data)")
else:
    print("Conversion failed!") 

if mbti==False:
    ## Tagging

    tagging_input_file = "data/twitter_conversations_raw.json"
    tagging_output_file = "data/twitter_conversations_tagged.json"
        
    tagged_conversations = tag_conversations(tagging_input_file, tagging_output_file)

    if tagged_conversations:
        print(f"\nFiles created:")
        print(f"- {tagging_output_file} (tagged conversation data)")
    else:
        print("Tagging failed!") 

    # NBA

    analyzer = NBA(api_key=api_key)

    nba_input_file = "data/twitter_conversations_tagged.json"
    nba_output_file = "data/nba.json"
    recommendations = analyzer.process_conversations(nba_input_file)


    if os.path.exists(nba_output_file):
        with open(nba_output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.extend(recommendations)
    with open(nba_output_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(recommendations)} next best action recommendations")


    # NBA Evaluation

    evaluator = NBAEvaluation(api_key=api_key)


    evaluations_input_file = "data/nba.json"
    evaluations_output_file = "data/nba_evaluations.json"
    evaluations = evaluator.evaluate_all_responses(evaluations_input_file, tagging_output_file)

    if evaluations:
        
        evaluator.save_evaluations(evaluations, evaluations_output_file)
        print(f"\nEvaluation completed successfully!")
        print(f"Total evaluations: {len(evaluations)}")
        print(f"Results saved to: {evaluations_output_file}")
    else:
        print("No evaluations were performed. Please check your data and API key.")


    export_nba_json_file = "data/nba.json"
    export_conversations_json_file = "data/twitter_conversations_tagged.json"
    export_output_csv_file = "output.csv"

    export_nba_to_csv(export_nba_json_file, export_output_csv_file, export_conversations_json_file)

    print(f"\nFiles created:")
    print(f"- {export_output_csv_file} (NBA recommendations in CSV format)")

else:
    mbti_tag()
    nba_mbti()
    mbti_eval()
