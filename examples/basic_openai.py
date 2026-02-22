"""
Basic OpenAI tracking example

Shows the simplest possible setup - just one line to enable tracking.
"""

import os
import tokenr
import openai

# One line to enable tracking
tokenr.init(token=os.getenv("TOKENR_TOKEN"), debug=True)

# Use OpenAI exactly as you normally would
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")
print("Cost automatically tracked to Tokenr!")
