import json

def parse_invoice_from_response(response_text):
    """Parses the LLM response and extracts invoice data."""
    try:
        invoice_data = json.loads(response_text)  # Assuming response is JSON
        return invoice_data  # Modify based on your response format
    except json.JSONDecodeError:
        print("❌ Error: Response is not in JSON format")
        return None
