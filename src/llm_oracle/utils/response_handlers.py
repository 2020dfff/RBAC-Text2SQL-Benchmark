"""
Response handlers for different types of queries
"""

def response_failure(prompt_user, model, e):
    """
    Handle failure response for chat queries
    """
    print(f"err: The following error occurred when querying {prompt_user} through {model}:")
    print(e)
    return {"query": prompt_user, "answer": "QUERY_FAILED"}

def response_failure_embed(list_of_text, model, e):
    """
    Handle failure response for embedding queries
    """
    print(f"err: The following error occurred when querying the following list of text through {model}:")
    print(list_of_text)
    print(e)
    return {"query": list_of_text, "answer": ["QUERY_FAILED"] * len(list_of_text)}
