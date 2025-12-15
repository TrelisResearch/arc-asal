"""
VLM integration for checking Lenia patterns against descriptions
"""

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / '.env')


def check_pattern_match(image_path: str, description: str) -> tuple[bool, str]:
    """
    Check if a Lenia pattern image matches a given description.

    Args:
        image_path: Path to the pattern image (PNG)
        description: The description to match against

    Returns:
        tuple of (matches: bool, explanation: str)
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # Read and encode the image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    model = "gemini-flash-lite-latest"

    prompt = f"""Look at this cellular automaton pattern image.

I want you to determine if this pattern matches the following description:
"{description}"

Respond with ONLY a JSON object in this exact format:
{{"matches": true/false, "explanation": "brief reason"}}
"""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=base64.b64decode(image_data),
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        temperature=0.2,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    response_text = response.text.strip()

    # Parse the JSON response
    import json
    try:
        # Handle markdown code blocks
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]

        result = json.loads(response_text)
        return result.get('matches', False), result.get('explanation', 'No explanation provided')
    except json.JSONDecodeError:
        # Try to extract boolean from response
        lower_response = response_text.lower()
        if 'true' in lower_response or 'yes' in lower_response or 'matches' in lower_response:
            return True, response_text
        return False, response_text


if __name__ == '__main__':
    # Test the VLM checker
    import sys

    if len(sys.argv) < 3:
        print("Usage: python vlm_checker.py <image_path> <description>")
        sys.exit(1)

    image_path = sys.argv[1]
    description = ' '.join(sys.argv[2:])

    print(f"Checking if image matches: '{description}'")
    matches, explanation = check_pattern_match(image_path, description)
    print(f"Matches: {matches}")
    print(f"Explanation: {explanation}")
