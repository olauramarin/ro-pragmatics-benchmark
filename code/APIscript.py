
import csv
import openai
import os
import json
import time
from collections import defaultdict

# It's recommended to store API keys in environment variables for security.

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Model Selection and Configuration ---.
# The 'temperature' parameter controls the randomness of the output.
# A value of 0.0 ensures the output is the same every time for the same input.
GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,  # top_p = 1.0 ensures no tokens are filtered based on probability mass
    "n": 1,        # Request only one response
}


# This script is set up for GPT-4.
GPT_MODEL = "gpt-4-1106-preview"  

def call_openai_api(prompt, model_name=GPT_MODEL):
    """
    Calls the OpenAI API to get a forced-choice response.
    
    Args:
        prompt (str): The constructed prompt for the model.
        model_name (str): The name of the OpenAI model to use.
        
    Returns:
        str: The model's response, or None if an error occurs.
    """
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes pragmatic cues in Romanian. Your only task is to choose the most appropriate option between A and B based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            **GENERATION_CONFIG,
        )
        # The model's response is in the 'content' of the first choice.
        # We also strip any leading/trailing whitespace.
        prediction = response.choices[0].message.content.strip().upper()
        # Ensure the prediction is only 'A' or 'B'
        if prediction not in ["A", "B"]:
            # If the model gives an extraneous response, we treat it as incorrect.
            return "UNKNOWN"
        return prediction
    except openai.RateLimitError:
        print("Rate limit exceeded. Waiting before retrying...")
        time.sleep(60)  # Wait for 60 seconds
        return call_openai_api(prompt, model_name)
    except Exception as e:
        print(f"An API error occurred: {e}")
        return None

def main():
    """
    Main function to run the evaluation experiment.
    """
  
    input_csv_path = "ro-pragmatics-benchmark/data/items_all.csv"
    prompts_data = []
    
    try:
        with open(input_csv_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                prompts_data.append(row)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return
    
    print(f"Loaded {len(prompts_data)} prompts. Starting evaluation for model: {GPT_MODEL}")

    # --- Run Experiment and Get Predictions ---
    results_data = []
    
    for index, row in enumerate(prompts_data):
        # Construct the single-turn prompt for the model.
        prompt = (
            f"Context: {row['Context']}\n\n"
            f"A: {row['Option_A']}\n"
            f"B: {row['Option_B']}\n\n"
            "Select the most pragmatically appropriate option (A or B). Please respond with only 'A' or 'B'."
        )
        
        # Make the API call to get the model's prediction.
        prediction = call_openai_api(prompt)
        
        # Create a new dictionary for the result, including the prediction.
        result_row = row.copy()
        result_row['Model_Prediction'] = prediction
        result_row['Correct'] = 'True' if prediction == row['Gold_Label'] else 'False'
        results_data.append(result_row)
        
        print(f"Prompt {index+1}/{len(prompts_data)} | Gold: {row['Gold_Label']} | Prediction: {prediction} | Correct: {result_row['Correct']}")

    # --- Analysis and Reporting ---
    print("\n--- Quantitative Analysis ---")
    
    # Calculate Correctness for Analysis
    correct_counts = sum(1 for row in results_data if row['Correct'] == 'True')
    total_items = len(results_data)
    
    # Overall Accuracy
    overall_accuracy = (correct_counts / total_items) * 100 if total_items > 0 else 0
    print(f"\nTotal Accuracy: {overall_accuracy:.2f}% ({correct_counts}/{total_items} correct)")
    
    # Accuracy per Phenomenon, Domain, and Role-Direction
    metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for row in results_data:
        phenomenon = row['Phenomenon']
        domain = row['Domain']
        role_direction = row['Role_Direction']
        is_correct = row['Correct'] == 'True'

        metrics['phenomenon'][phenomenon]['total'] += 1
        metrics['domain'][domain]['total'] += 1
        metrics['role_direction'][role_direction]['total'] += 1
        
        if is_correct:
            metrics['phenomenon'][phenomenon]['correct'] += 1
            metrics['domain'][domain]['correct'] += 1
            metrics['role_direction'][role_direction]['correct'] += 1

    print("\nAccuracy by Phenomenon:")
    for key, val in metrics['phenomenon'].items():
        accuracy = (val['correct'] / val['total']) * 100 if val['total'] > 0 else 0
        print(f"{key:<15}: {accuracy:.2f}%")
    
    print("\nAccuracy by Domain:")
    for key, val in metrics['domain'].items():
        accuracy = (val['correct'] / val['total']) * 100 if val['total'] > 0 else 0
        print(f"{key:<15}: {accuracy:.2f}%")
        
    print("\nAccuracy by Role-Direction:")
    for key, val in metrics['role_direction'].items():
        accuracy = (val['correct'] / val['total']) * 100 if val['total'] > 0 else 0
        print(f"{key:<15}: {accuracy:.2f}%")

    # Macro-Recall
    gold_A_counts = {'correct': 0, 'total': 0}
    gold_B_counts = {'correct': 0, 'total': 0}
    for row in results_data:
        if row['Gold_Label'] == 'A':
            gold_A_counts['total'] += 1
            if row['Correct'] == 'True':
                gold_A_counts['correct'] += 1
        else: # row['Gold_Label'] == 'B'
            gold_B_counts['total'] += 1
            if row['Correct'] == 'True':
                gold_B_counts['correct'] += 1
                
    recall_A = (gold_A_counts['correct'] / gold_A_counts['total']) if gold_A_counts['total'] > 0 else 0
    recall_B = (gold_B_counts['correct'] / gold_B_counts['total']) if gold_B_counts['total'] > 0 else 0
    macro_recall = ((recall_A + recall_B) / 2) * 100
    print(f"\nMacro-Recall: {macro_recall:.2f}%")

    # --- Save Results ---
    output_csv_path = f"results_{GPT_MODEL.replace('-', '_')}.csv"
    
    fieldnames = list(prompts_data[0].keys()) + ['Model_Prediction', 'Correct']
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)
        
    print(f"\nExperiment complete. Results saved to '{output_csv_path}'")

if __name__ == "__main__":
    main()
