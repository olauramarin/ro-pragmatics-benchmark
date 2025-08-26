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

