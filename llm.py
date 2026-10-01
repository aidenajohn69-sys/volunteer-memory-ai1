import os
import time
from google import genai
from google.genai import errors

# Safely fetch the API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: API key not found in environment variables.")
    print("Please run the $env:GOOGLE_API_KEY command in PowerShell again.")
    exit()

# Initialize the SDK client
client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """
You are the operational assistant for a student volunteer network. 
Your job is to answer the coordinator's question using ONLY the provided context.

CRITICAL RULES:
1. CONFLICT RESOLUTION: The context may contain contradictory instructions from different semesters. ALWAYS prioritize the information with the most recent timestamp. 
2. EXPLAIN THE CHANGE: If you resolve a conflict, briefly mention what the old rule was and that you are providing the updated rule.
3. TRANSLATION: If a relevant note is in a regional language, translate it to English in your answer.
4. NO HALLUCINATION: If the context does not contain the answer, reply exactly with: "Information not found in archives." Do not invent an answer.
"""

def format_context(retrieved_docs):
    """Converts Person 2's output into a readable string for the LLM."""
    formatted_text = ""
    for doc in retrieved_docs:
        formatted_text += f"[Source: {doc['source']} | Date: {doc['timestamp']}]\n{doc['text']}\n\n"
    return formatted_text

def get_answer(query: str, retrieved_docs: list) -> tuple[str, list]:
    """
    The main function Person 4 will call from the Streamlit UI.
    Includes a retry mechanism for busy servers.
    """
    formatted_context = format_context(retrieved_docs)
    unique_sources = list(set([doc['source'] for doc in retrieved_docs]))
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{formatted_context}\n\nUSER QUESTION: {query}"
    
    # Retry loop: Try up to 3 times if the server is busy
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=full_prompt
            )
            return response.text, unique_sources
            
        except errors.ServerError:
            if attempt < max_retries - 1:
                print(f"API busy. Retrying in 5 seconds... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(5)
            else:
                return "The AI servers are currently overloaded. Please try again later.", unique_sources

# --- LOCAL TESTING ---
if __name__ == "__main__":
    test_docs = [
        {
            "text": "बस बुकिंग के लिए कम से कम तीन दिन पहले कॉल करें और ड्राइवर का फोन नंबर हमेशा वेरीफाई करें।", # Hindi for booking 3 days in advance & verifying driver phone
            "source": "regional_notes_2023.txt",
            "timestamp": "2023-08-15"
        }
    ]

    print("\n--- TEST: Regional Translation ---")
    ans, src = get_answer("What are the driver instructions?", test_docs)
    print(ans)

    print("\n--- TEST: Missing Information Guardrail ---")
    ans_empty, src_empty = get_answer("What is the refund policy for train tickets?", test_docs)
    print(ans_empty)