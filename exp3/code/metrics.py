import os
import ast
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Function to extract 'question_type' from metadata string
def extract_question_type(metadata_str):
    if pd.isna(metadata_str):
        return np.nan
    try:
        metadata_dict = ast.literal_eval(metadata_str)
        return metadata_dict.get('question_type')
    except (ValueError, SyntaxError):
        return np.nan

def compute_metrics(file_path, display_overall_only=False):
    # Determine the output file path
    output_file_path = os.path.splitext(file_path)[0] + '_final_results.csv'
    
    # Check if the output file already exists
    if os.path.exists(output_file_path):
        print(f"File already exists: {output_file_path}. Loading it directly.")
        return pd.read_csv(output_file_path)

    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract question type
    df['question_type'] = df['metadata'].apply(extract_question_type)

    # Ensure 'correctness' column is boolean
    df['correctness'] = df['correctness'].astype(bool)

    # Check for 'correctness_method2' column and map its values if present
    if 'correctness_method2' in df.columns:
        df['correctness_method2_mapped'] = df['correctness_method2'].apply(lambda x: 1 if x >= 4 else 0)

    # Remove rows with NaN values
    df = df.dropna(subset=['question_type', 'reference_answer', 'agent_answer'])

    # Load the pre-trained sentence transformer model
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # Generate embeddings for reference and agent answers
    reference_embeddings = model.encode(df['reference_answer'].tolist(), convert_to_tensor=True)
    agent_embeddings = model.encode(df['agent_answer'].tolist(), convert_to_tensor=True)

    # Calculate cosine similarity between embeddings
    cosine_similarities = util.pytorch_cos_sim(reference_embeddings, agent_embeddings).diagonal().tolist()
    df['answer similarity'] = cosine_similarities

    # Define the metrics
    available_metrics = [metric for metric in ['RAGAS Context Recall', 'RAGAS Faithfulness', 'RAGAS Answer Relevancy', 'RAGAS Context Precision'] if metric in df.columns]
    metrics = available_metrics + ['answer similarity']

    if not display_overall_only:
        # Compute average metrics per question type
        average_metrics_per_question_type = df.groupby('question_type')[metrics].mean().reset_index()

        # Calculate Context F1 score if both Recall and Precision are available
        if 'RAGAS Context Recall' in available_metrics and 'RAGAS Context Precision' in available_metrics:
            average_metrics_per_question_type['RAGAS Context F1 Score'] = 2 * (
                (average_metrics_per_question_type['RAGAS Context Recall'] * average_metrics_per_question_type['RAGAS Context Precision']) /
                (average_metrics_per_question_type['RAGAS Context Recall'] + average_metrics_per_question_type['RAGAS Context Precision'])
            )
        else:
            average_metrics_per_question_type['RAGAS Context F1 Score'] = np.nan

        # Calculate correctness for each question type
        correctness_per_question_type = df.groupby('question_type')['correctness'].mean().reset_index()
        correctness_per_question_type['correctness'] = correctness_per_question_type['correctness'] * 100  # Convert to percentage

        # If 'correctness_method2' is present, calculate its raw and mapped averages for each question type
        if 'correctness_method2' in df.columns:
            correctness_method2_raw_per_question_type = df.groupby('question_type')['correctness_method2'].mean().reset_index()
            correctness_method2_mapped_per_question_type = df.groupby('question_type')['correctness_method2_mapped'].mean().reset_index()
            correctness_method2_mapped_per_question_type['correctness_method2_mapped'] = correctness_method2_mapped_per_question_type['correctness_method2_mapped'] * 100  # Convert to percentage
            average_metrics_per_question_type = pd.merge(average_metrics_per_question_type, correctness_method2_raw_per_question_type, on='question_type', how='left')
            average_metrics_per_question_type = pd.merge(average_metrics_per_question_type, correctness_method2_mapped_per_question_type, on='question_type', how='left')

        # Merge correctness with average metrics
        average_metrics_per_question_type = pd.merge(average_metrics_per_question_type, correctness_per_question_type, on='question_type')

        # Set metrics to 'N/A' for 'out of scope' questions
        out_of_scope_idx = average_metrics_per_question_type['question_type'] == 'out of scope'
        average_metrics_per_question_type.loc[out_of_scope_idx, metrics[:-1]] = np.nan  # Exclude 'answer similarity'
        average_metrics_per_question_type.loc[out_of_scope_idx, 'RAGAS Context F1 Score'] = np.nan

    # Filter out 'out of scope' questions for overall metrics except for correctness and answer similarity
    in_scope_df = df[df['question_type'] != 'out of scope']
    overall_metrics_excluding_out_of_scope = in_scope_df[available_metrics].mean()

    # Calculate overall Context F1 score excluding 'out of scope' if applicable
    if 'RAGAS Context Recall' in available_metrics and 'RAGAS Context Precision' in available_metrics:
        overall_context_f1_score_excluding_out_of_scope = 2 * (
            (overall_metrics_excluding_out_of_scope['RAGAS Context Recall'] * overall_metrics_excluding_out_of_scope['RAGAS Context Precision']) /
            (overall_metrics_excluding_out_of_scope['RAGAS Context Recall'] + overall_metrics_excluding_out_of_scope['RAGAS Context Precision'])
        )
    else:
        overall_context_f1_score_excluding_out_of_scope = np.nan

    # Calculate overall correctness and answer similarity including all questions
    overall_correctness = df['correctness'].mean() * 100  # Convert to percentage
    overall_answer_similarity = df['answer similarity'].mean()

    # If 'correctness_method2' is present, calculate its overall raw and mapped averages
    if 'correctness_method2' in df.columns:
        overall_correctness_method2_raw = df['correctness_method2'].mean()  # Raw average
        overall_correctness_method2_mapped = df['correctness_method2_mapped'].mean() * 100  # Convert to percentage
    else:
        overall_correctness_method2_raw = np.nan
        overall_correctness_method2_mapped = np.nan

    # Combine overall metrics into a DataFrame
    overall_metrics_df = pd.DataFrame(overall_metrics_excluding_out_of_scope).transpose()
    overall_metrics_df['question_type'] = 'Overall'
    overall_metrics_df['RAGAS Context F1 Score'] = overall_context_f1_score_excluding_out_of_scope
    overall_metrics_df['correctness'] = overall_correctness
    overall_metrics_df['answer similarity'] = overall_answer_similarity
    if 'correctness_method2' in df.columns:
        overall_metrics_df['correctness_method2'] = overall_correctness_method2_raw
        overall_metrics_df['correctness_method2_mapped'] = overall_correctness_method2_mapped

    if display_overall_only:
        final_df = overall_metrics_df
    else:
        # Combine the per-question type metrics with the overall metrics
        final_df = pd.concat([average_metrics_per_question_type, overall_metrics_df], ignore_index=True)

    # Save the final DataFrame to a CSV file
    final_df.to_csv(output_file_path, index=False)
    print(f"Metrics saved to: {output_file_path}")

    return final_df
