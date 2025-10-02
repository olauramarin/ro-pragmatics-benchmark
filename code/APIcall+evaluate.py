import csv
import openai 
from openai import OpenAI
import os
import json
import time
from collections import defaultdict
from dataclasses import dataclass


import os 

#OpenAI-GPT5 as an example. API calls can be found in the documentation for each model.

client = OpenAI(api_key="insert key here if you want it hardcoded or insert it using the terminal")

GENERATION_CONFIG = {
    "n": 1   # request one response
}

GPT_MODEL = "gpt-5"

def call_openai_api(prompt, model_name=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Ești un asistent util care "
                    "analizează indiciile pragmatice în limba română. "
                    "Alege opțiunea cea mai potrivită între A și B."},
                {"role": "user", "content": prompt}
            ],
            **GENERATION_CONFIG,
        )
        prediction = response.choices[0].message.content.strip().upper()
        if prediction not in ["A", "B"]:
            return "UNKNOWN"
        return prediction
    except Exception as e:
        print(f"API error: {e}")
        return None

def main():
    """
    Main function to run the evaluation experiment.
    """
  
    input_csv_path = "./data/items_all.csv"
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
            "Select the most pragmatically appropriate option (A or B). "
            "Please respond with only 'A' or 'B'."
        )
        
        # Make the API call to get the model's prediction.
        prediction = call_openai_api(prompt)
        
        # Create a new dictionary for the result, including the prediction.
        result_row = row.copy()
        result_row['Model_Prediction'] = prediction
        result_row['Correct'] = 'True' if prediction == row['Gold_Label'] else 'False'
        results_data.append(result_row)
        
        print(f"Prompt {index+1}/{len(prompts_data)} | Gold: {row['Gold_Label']}"
              f" | Prediction: {prediction} | Correct: {result_row['Correct']}")

    # --- Analysis and Reporting ---
    print("\n--- Quantitative Analysis ---")

    # A dataclass for holding correctness/totals
    @dataclass
    class Metric:
        correct: int
        total: int
    
    # Compute correct percentage from a metric
    def calc_percentage(m):
        return (m.correct / m.total) * 100 if m.total > 0 else 0

    # Calculate Correctness for Analysis
    overall_metric = Metric(
            correct=sum(1 for row in results_data if row['Correct'] == 'True'),
            total=len(results_data))
    
    # Overall Accuracy
    print(f"\nTotal Accuracy: {calc_percentage(overall_metric):.2f}%"
          f"({correct_counts}/{total_items} correct)")


    # Accuracy per Phenomenon, Domain, and Role-Direction
    phenomenon_metrics = defaultdict(lambda: Metric(0, 0))
    domain_metrics = defaultdict(lambda: Metric(0, 0))
    role_direction_metrics = defaultdict(lambda: Metric(0, 0))

    for row in results_data:
        phenomenon = row['Phenomenon']
        domain = row['Domain']
        role_direction = row['Role_Direction']
        is_correct = row['Correct'] == 'True'

        phenomenon_metrics[phenomenon].total += 1
        domain_metrics[domain].total += 1
        role_direction_metrics[role_direction].total += 1
        
        if is_correct:
            phenomenon_metrics[phenomenon].correct += 1
            domain_metrics[domain].correct += 1
            role_direction_metrics[role_direction].correct += 1

    def print_accuracies(metrics):
        for key, val in metrics.items():
            print(f"{key:<15}: {calc_percentage(val):.2f}%")

    print("\nAccuracy by Phenomenon:")
    print_accuracies(phenomenon_metrics)
    
    print("\nAccuracy by Domain:")
    print_accuracies(domain_metrics)
        
    print("\nAccuracy by Role-Direction:")
    print_accuracies(role_direction_metrics)

    # Macro-Recall
    gold_A_counts = Metric(0, 0)
    gold_B_counts = Metric(0, 0)
    for row in results_data:
        if row['Gold_Label'] == 'A':
            gold_A_counts.total += 1
            if row['Correct'] == 'True':
                gold_A_counts.correct += 1
        else: # row['Gold_Label'] == 'B'
            gold_B_counts.total += 1
            if row['Correct'] == 'True':
                gold_B_counts.correct += 1
                
    recall_A = calc_percentage(gold_A_counts)
    recall_B = calc_percentage(gold_B_counts)
    macro_recall = ((recall_A + recall_B) / 2)
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
