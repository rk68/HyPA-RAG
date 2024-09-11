import pandas as pd
from typing import Dict
from tqdm import tqdm
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
import argparse
from utils_code import initialize_openai_creds, create_llm
from dotenv import load_dotenv, find_dotenv


CORRECTNESS_SYS_TMPL = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query,
- a reference answer, and
- a generated answer.

Your job is to judge the correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.
The reasoning MUST NOT UNDER ANY CIRCUMSTANCES BE LONGER THAN 1 SENTENCE.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- Use the following criteria for scoring correctness:

1. Score of 1:
    - The generated answer is completely incorrect.
    - Contains major factual errors or misconceptions.
    - Does not address any components of the user query correctly.
    - Example:
      - Query: "What is the capital of France?"
      - Generated Answer: "The capital of France is Berlin."

2. Score of 2:
    - Significant mistakes are present.
    - Addresses at least one component of the user query correctly but has major errors in other parts.
    - Example:
      - Query: "What is the capital of France and its population?"
      - Generated Answer: "The capital of France is Paris, and its population is 100 million."

3. Score of 3:
    - Partially correct with some incorrect information.
    - Addresses multiple components of the user query correctly.
    - Minor factual errors are present.
    - Example:
      - Query: "What is the capital of France and its population?"
      - Generated Answer: "The capital of France is Paris, and its population is around 3 million."

4. Score of 4:
    - Mostly correct with minimal errors.
    - Correctly addresses all components of the user query.
    - Errors do not substantially affect the overall correctness.
    - Example:
      - Query: "What is the capital of France and its population?"
      - Generated Answer: "The capital of France is Paris, and its population is approximately 2.1 million."

5. Score of 5:
    - Completely correct.
    - Addresses all components of the user query correctly without any errors.
    - Providing more information than necessary should not be penalized as long as all provided information is correct.
    - Example:
      - Query: "What is the capital of France and its population?"
      - Generated Answer: "The capital of France is Paris, and its population is approximately 2.1 million. Paris is known for its rich history and iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral."

Checklist for Evaluation:
  - Component Coverage: Does the answer cover all parts of the query?
  - Factual Accuracy: Are the facts presented in the answer correct?
  - Error Severity: How severe are any errors present in the answer?
  - Comparison to Reference: How closely does the answer align with the reference answer?

Edge Cases:
  - If the answer includes both correct and completely irrelevant information, focus only on the relevant portions for scoring.
  - If the answer is correct but incomplete, score based on the completeness criteria within the relevant score range.
  - If the answer provides more information than necessary, it should not be penalized as long as all information is correct.
"""

CORRECTNESS_USER_TMPL = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

# Create a chat template for evaluation
eval_chat_template = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=CORRECTNESS_SYS_TMPL),
        ChatMessage(role=MessageRole.USER, content=CORRECTNESS_USER_TMPL),
    ]
)

def run_correctness_eval(
    query_str: str,
    reference_answer: str,
    generated_answer: str,
    llm: AzureOpenAI,
    threshold: float = 4.0,
) -> Dict:
    """Run correctness evaluation for a single query."""
    # Format the chat messages using the prompt template
    fmt_messages = eval_chat_template.format_messages(
        llm=llm,
        query=query_str,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )

    # Get the chat response from the LLM
    chat_response = llm.chat(fmt_messages)
    raw_output = chat_response.message.content

    # Extract the score and reasoning from the response
    score_str, reasoning_str = raw_output.split("\n", 1)
    score = float(score_str.strip())
    reasoning = reasoning_str.strip()

    return {"passing": score >= threshold, "score": score, "reason": reasoning}

def process_correctness_scores(file_path: str, llm: AzureOpenAI, threshold: float = 4.0, num_rows: int = None) -> float:
    """Process correctness scores from a CSV file, calculate average score."""
    # Load the data
    df = pd.read_csv(file_path)

    # Extract necessary columns into lists
    questions = df['question'].tolist()
    reference_answers = df['reference_answer'].tolist()
    agent_answers = df['agent_answer'].tolist()

    # Set num_rows to total rows if not provided
    if num_rows is None:
        num_rows = len(df)

    # Initialize the results list
    results = []

    # Use tqdm for progress bar while processing rows
    for question, ref_answer, agent_answer in tqdm(zip(questions[:num_rows], reference_answers[:num_rows], agent_answers[:num_rows]),
                                                   total=num_rows, desc="Processing Correctness Scores"):
        result = run_correctness_eval(question, ref_answer, agent_answer, llm=llm, threshold=threshold)
        results.append(result)

    # Add scores and reasons to DataFrame
    df.loc[:num_rows - 1, 'correctness_method2'] = [result['score'] for result in results]
    df.loc[:num_rows - 1, 'reason_method2'] = [result['reason'] for result in results]

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

    # Calculate and print the average score
    total_score = sum(result['score'] for result in results)
    average_score = total_score / num_rows
    print(f"\nAverage Correctness Score: {average_score:.2f}")

    return average_score

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process correctness scores from a CSV file.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file.")
    parser.add_argument("--threshold", type=float, default=4.0, help="Threshold for passing the evaluation (default: 4.0).")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to process (default: all rows).")
    args = parser.parse_args()

    # LLM setup
    gpt35_creds, gpt4o_mini_creds, gpt4o_creds = initialize_openai_creds()
    llm = create_llm(model='gpt-4o-mini',gpt35_creds=gpt35_creds, gpt4o_creds=gpt4o_creds, gpt4o_mini_creds=gpt4o_mini_creds)

    # Call the process_correctness_scores function with parsed arguments
    process_correctness_scores(
        file_path=args.file_path,
        llm=llm,
        threshold=args.threshold,
        num_rows=args.num_rows
    )
