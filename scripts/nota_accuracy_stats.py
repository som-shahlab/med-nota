import pandas as pd
import numpy as np
from scipy import stats

def perform_bootstrap(original_correct, nota_correct, n_questions, n_iterations=1000):
    """
    Perform bootstrap resampling to calculate confidence intervals for accuracy drop.
    
    Args:
        original_correct (list): Binary list of correctness for original questions (1=correct, 0=incorrect)
        nota_correct (list): Binary list of correctness for NOTA questions (1=correct, 0=incorrect)
        n_questions (int): Number of questions
        n_iterations (int): Number of bootstrap iterations
    
    Returns:
        dict: Dictionary with bootstrap results including mean and confidence intervals
    """
    # Ensure both arrays are the same length
    assert len(original_correct) == len(nota_correct) == n_questions
    
    # Store bootstrap results
    bootstrap_drops = []
    
    for _ in range(n_iterations):
        # Sample with replacement
        indices = np.random.choice(n_questions, size=n_questions, replace=True)
        
        # Get bootstrapped samples
        bootstrap_original = [original_correct[i] for i in indices]
        bootstrap_nota = [nota_correct[i] for i in indices]
        
        # Calculate accuracies
        bootstrap_original_acc = sum(bootstrap_original) / n_questions
        bootstrap_nota_acc = sum(bootstrap_nota) / n_questions
        
        # Calculate and store accuracy drop
        bootstrap_drops.append(bootstrap_original_acc - bootstrap_nota_acc)
    
    # Sort results for percentile calculation
    bootstrap_drops.sort()
    
    # Calculate 95% confidence interval
    lower_bound = bootstrap_drops[int(n_iterations * 0.025)]
    upper_bound = bootstrap_drops[int(n_iterations * 0.975)]
    mean_drop = sum(bootstrap_drops) / n_iterations
    
    return {
        'mean_drop': mean_drop,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'bootstrap_samples': bootstrap_drops
    }

def perform_mcnemars_test(original_correct, nota_correct):
    """
    Perform McNemar's test to assess statistical significance of accuracy difference.
    
    Args:
        original_correct (list): Binary list of correctness for original questions
        nota_correct (list): Binary list of correctness for NOTA questions
    
    Returns:
        dict: Dictionary with test results
    """
    # Count discordant pairs
    b = 0  # Original correct, NOTA incorrect
    c = 0  # Original incorrect, NOTA correct
    
    for i in range(len(original_correct)):
        if original_correct[i] == 1 and nota_correct[i] == 0:
            b += 1
        elif original_correct[i] == 0 and nota_correct[i] == 1:
            c += 1
    
    # McNemar's test statistic with continuity correction
    if b + c == 0:
        statistic = 0
        p_value = 1.0
    else:
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        # Calculate p-value using chi-square distribution with 1 df
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'b': b,  # Original correct, NOTA incorrect
        'c': c   # Original incorrect, NOTA correct
    }

def analyze_accuracy_drop(model_data, clinician_data):
    """
    Analyze the accuracy drop in the NOTA experiment.
    
    Args:
        model_data (pd.DataFrame): DataFrame containing model responses and correctness
        clinician_data (pd.DataFrame): DataFrame containing clinician annotations
    
    Returns:
        dict: Dictionary with accuracy metrics
    """
    clinician_data = clinician_data.drop_duplicates(subset=['question_id'])

    # Make sure question_id is the same type in both dataframes
    model_data['question_id'] = model_data.index + 1  # Add question_id based on row index
    model_data['question_id'] = model_data['question_id'].astype(str)
    clinician_data['question_id'] = clinician_data['question_id'].astype(str)
    
    # Merge the dataframes on question_id
    merged_data = pd.merge(model_data, clinician_data, on='question_id', how='inner')
    
    # Filter for cases where NOTA is the correct answer according to clinician
    #nota_correct_cases = merged_data[merged_data['is_none_still_best'] == 'yes']
    excluded_ids = ['10', '32', '45', '75', '44']
    nota_correct_cases = merged_data[
        (merged_data['is_none_still_best'] == 'yes') & 
        (~merged_data['question_id'].isin(excluded_ids))
    ]
    
    # Convert string values to integers where needed
    merged_data['original_cot_correct'] = pd.to_numeric(merged_data['original_cot_correct'])
    merged_data['noto_correct'] = pd.to_numeric(merged_data['noto_correct'])
    merged_data['no_cot_correct'] = pd.to_numeric(merged_data['no_cot_correct'])
    
    # Calculate relevant metrics
    total_questions = len(merged_data)
    original_correct = merged_data['original_cot_correct'].sum()
    nota_correct = merged_data['noto_correct'].sum()
    no_cot_correct = merged_data['no_cot_correct'].sum()
    
    # For NOTA correct cases specifically
    nota_cases_count = len(nota_correct_cases)
    if nota_cases_count > 0:
        nota_correct_cases['original_cot_correct'] = pd.to_numeric(nota_correct_cases['original_cot_correct'])
        nota_correct_cases['noto_correct'] = pd.to_numeric(nota_correct_cases['noto_correct'])
        original_correct_nota_cases = nota_correct_cases['original_cot_correct'].sum()
        nota_correct_nota_cases = nota_correct_cases['noto_correct'].sum()
        nota_correct_cases['no_cot_correct'] = pd.to_numeric(nota_correct_cases['no_cot_correct'])
        no_cot_correct_nota_cases = nota_correct_cases['no_cot_correct'].sum()
    else:
        original_correct_nota_cases = 0
        nota_correct_nota_cases = 0
        no_cot_correct_nota_cases = 0 
    
    # Calculate the accuracy drop - specifically for cases where NOTA is the correct answer
    accuracy_drop = (original_correct_nota_cases - nota_correct_nota_cases) / nota_cases_count if nota_cases_count > 0 else 0
    
    # Find questions that were correct in original CoT but incorrect in NOTA experiment
    # and where NOTA is the correct answer according to clinician
    dropped_questions = nota_correct_cases[
        (nota_correct_cases['original_cot_correct'] == 1) & 
        (nota_correct_cases['noto_correct'] == 0)
    ]
    
    return {
        'total_questions': total_questions,
        'original_accuracy': original_correct / total_questions,
        'nota_accuracy': nota_correct / total_questions,
        'overall_accuracy_drop': (original_correct - nota_correct) / total_questions,
        'nota_cases_count': nota_cases_count,
        'accuracy_drop_nota_cases': accuracy_drop,
        'original_accuracy_nota_cases': original_correct_nota_cases / nota_cases_count if nota_cases_count > 0 else 0,
        'nota_accuracy_nota_cases': nota_correct_nota_cases / nota_cases_count if nota_cases_count > 0 else 0,
        'no_cot_accuracy': no_cot_correct / total_questions,
        'no_cot_accuracy_nota_cases': no_cot_correct_nota_cases / nota_cases_count if nota_cases_count > 0 else 0,
        'dropped_questions_count': len(dropped_questions),
        'dropped_questions': dropped_questions
    }

def analyze_accuracy_drop_with_stats(model_data, clinician_data):
    """
    Analyze the accuracy drop in the NOTA experiment with statistical analysis.
    
    Args:
        model_data (pd.DataFrame): DataFrame containing model responses and correctness
        clinician_data (pd.DataFrame): DataFrame containing clinician annotations
    
    Returns:
        dict: Dictionary with accuracy metrics and statistical analysis
    """
    # First get the base results using your existing function
    base_results = analyze_accuracy_drop(model_data, clinician_data)
    
    # Make sure to convert question_id types just like in the analyze_accuracy_drop function
    model_data_copy = model_data.copy()
    clinician_data_copy = clinician_data.copy()
    
    # Add question_id if not present in model_data
    if 'question_id' not in model_data_copy.columns:
        model_data_copy['question_id'] = model_data_copy.index + 1
    
    # Convert to string in both dataframes to ensure consistent types
    model_data_copy['question_id'] = model_data_copy['question_id'].astype(str)
    clinician_data_copy['question_id'] = clinician_data_copy['question_id'].astype(str)
    
    # Prepare binary correctness arrays for NOTA cases
    nota_correct_cases = pd.merge(model_data_copy, 
                                  clinician_data_copy[clinician_data_copy['is_none_still_best'] == 'yes'], 
                                  on='question_id', how='inner')
    
    # Convert to numeric if needed
    nota_correct_cases['original_cot_correct'] = pd.to_numeric(nota_correct_cases['original_cot_correct'])
    nota_correct_cases['noto_correct'] = pd.to_numeric(nota_correct_cases['noto_correct'])
    
    # Create binary arrays
    original_correct_binary = nota_correct_cases['original_cot_correct'].tolist()
    nota_correct_binary = nota_correct_cases['noto_correct'].tolist()
    
    # Perform bootstrap
    bootstrap_results = perform_bootstrap(
        original_correct_binary, 
        nota_correct_binary, 
        len(nota_correct_cases),
        n_iterations=1000
    )
    
    # Perform McNemar's test
    mcnemars_results = perform_mcnemars_test(original_correct_binary, nota_correct_binary)
    
    # Add statistical results to the base results
    base_results.update({
        'bootstrap_mean_drop': bootstrap_results['mean_drop'],
        'bootstrap_ci_lower': bootstrap_results['lower_bound'],
        'bootstrap_ci_upper': bootstrap_results['upper_bound'],
        'mcnemars_statistic': mcnemars_results['statistic'],
        'mcnemars_p_value': mcnemars_results['p_value'],
        'mcnemars_b': mcnemars_results['b'],
        'mcnemars_c': mcnemars_results['c']
    })
    
    return base_results

# Modify the main function to include the statistical analysis
def main_with_stats(model_csv_path, clinician_csv_path):
    """
    Main function to run the analysis with statistical testing.
    
    Args:
        model_csv_path (str): Path to the model CSV file
        clinician_csv_path (str): Path to the clinician annotation CSV file
    """
    # Load the data
    model_data = pd.read_csv(model_csv_path)
    clinician_data = pd.read_csv(clinician_csv_path)
    
    # Run the analysis with statistical testing
    results = analyze_accuracy_drop_with_stats(model_data, clinician_data)
    
    # Print the results
    print("=== Accuracy Analysis Results ===")
    print(f"Total questions analyzed: {results['total_questions']}")
    print(f"Original CoT accuracy: {results['original_accuracy']:.2%}")
    print(f"NOTA experiment accuracy: {results['nota_accuracy']:.2%}")
    print(f"Overall accuracy drop: {results['overall_accuracy_drop']:.2%}")
    
    print("\n=== NOTA-specific Analysis with Statistical Testing ===")
    print(f"Number of questions where NOTA is correct: {results['nota_cases_count']}")
    print(f"Original CoT accuracy on NOTA cases: {results['original_accuracy_nota_cases']:.2%}")
    print(f"NOTA experiment accuracy on NOTA cases: {results['nota_accuracy_nota_cases']:.2%}")
    print(f"Accuracy drop for NOTA cases: {results['accuracy_drop_nota_cases']:.2%}")
    print(f"Bootstrap 95% CI: [{results['bootstrap_ci_lower']:.2%}, {results['bootstrap_ci_upper']:.2%}]")
    print(f"McNemar's test statistic: {results['mcnemars_statistic']:.4f}")
    print(f"McNemar's test p-value: {results['mcnemars_p_value']:.6f}")
    print(f"Questions correct in original but incorrect in NOTA: {results['mcnemars_b']}")
    print(f"Questions incorrect in original but correct in NOTA: {results['mcnemars_c']}")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Check if file paths are provided as command-line arguments
    if len(sys.argv) == 3:
        model_csv_path = sys.argv[1]
        clinician_csv_path = sys.argv[2]
    else:
        # Default file paths if not provided
        model_csv_path = "../../data/medqa_nato_results_gpt.csv"
        clinician_csv_path = "../../data/clinical_annotations.csv"
        print(f"Using default file paths: {model_csv_path} and {clinician_csv_path}")
    
    try:
        # Run the enhanced analysis with statistical testing
        results = main_with_stats(model_csv_path, clinician_csv_path)
        
        # Save results to file
        output_file = "nota_accuracy_results_with_stats.txt"
        with open(output_file, "w") as f:
            f.write("=== Accuracy Analysis Results ===\n")
            f.write(f"Total questions analyzed: {results['total_questions']}\n")
            f.write(f"Original CoT accuracy: {results['original_accuracy']:.2%}\n")
            f.write(f"NOTA experiment accuracy: {results['nota_accuracy']:.2%}\n")
            f.write(f"Overall accuracy drop: {results['overall_accuracy_drop']:.2%}\n")
            
            f.write("\n=== NOTA-specific Analysis with Statistical Testing ===\n")
            f.write(f"Number of questions where NOTA is correct: {results['nota_cases_count']}\n")
            f.write(f"Original CoT accuracy on NOTA cases: {results['original_accuracy_nota_cases']:.2%}\n")
            f.write(f"NOTA experiment accuracy on NOTA cases: {results['nota_accuracy_nota_cases']:.2%}\n")
            f.write(f"Accuracy drop for NOTA cases: {results['accuracy_drop_nota_cases']:.2%}\n")
            f.write(f"Bootstrap 95% CI: [{results['bootstrap_ci_lower']:.2%}, {results['bootstrap_ci_upper']:.2%}]\n")
            f.write(f"McNemar's test statistic: {results['mcnemars_statistic']:.4f}\n")
            f.write(f"McNemar's test p-value: {results['mcnemars_p_value']:.6f}\n")
            f.write(f"Questions correct in original but incorrect in NOTA: {results['mcnemars_b']}\n")
            f.write(f"Questions incorrect in original but correct in NOTA: {results['mcnemars_c']}\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()