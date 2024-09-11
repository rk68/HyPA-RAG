from typing import List

def get_classification_prompt(categories_list: List[str]) -> str:
    """Generate classification prompt based on the categories list."""
    categories_str = ', '.join([f"'{category}'" for category in categories_list])
    return (
        f"Classify the following query into one of the following categories: {categories_str}. "
        f"If it doesn't fit into any category, respond with 'None'. "
        f"Return the classification, do not output absolutely anything else."
    )


def get_query_generation_prompt(query_str: str, num_queries: int) -> str:
    """Generate query generation prompt based on query string and number of sub-queries."""
    return (
        f"You are an expert at distilling a user question into sub-questions that can be used to fully answer the original query. "
        f"First, identify the key words from the original question below: \n"
        f"{query_str}"
        f"Generate {num_queries} sub-queries that cover the different aspects needed to fully address the user's query.\n\n"
        f"Here is an example: \n"
        f"Original Question: What does test data mean and what do I need to know about it?\n"
        f"Output:\n"
        f"definition of 'test data'\n"
        f"test data requirements and conditions for a bias audit\n"
        f"examples of the use of test data in a bias audit\n\n"
        f"Output the rewritten sub-queries, one on each line, do not output absolutely anything else."
    )
