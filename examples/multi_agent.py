"""
Multi-agent tracking example

Shows how to track costs for different AI agents separately.
"""

import os
import tokenr
import openai
from anthropic import Anthropic

# Initialize tracking
tokenr.init(token=os.getenv("TOKENR_TOKEN"), debug=True)


def customer_support_bot(question):
    """Customer support agent using GPT-4"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a customer support agent."},
            {"role": "user", "content": question}
        ],
        tokenr_agent_id="customer-support",
        tokenr_feature="support-chat"
    )
    return response.choices[0].message.content


def content_writer_bot(topic):
    """Content writer using Claude"""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"Write a blog post about {topic}"}
        ],
        tokenr_agent_id="content-writer",
        tokenr_feature="blog-generation"
    )
    return response.content[0].text


def code_reviewer_bot(code):
    """Code reviewer using GPT-3.5"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a code reviewer."},
            {"role": "user", "content": f"Review this code:\n\n{code}"}
        ],
        tokenr_agent_id="code-reviewer",
        tokenr_feature="code-review"
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Each agent's costs are tracked separately in Tokenr
    print("Support Bot:", customer_support_bot("How do I reset my password?"))
    print("\nContent Bot:", content_writer_bot("AI safety"))
    print("\nCode Reviewer:", code_reviewer_bot("def hello(): print('hi')"))

    print("\nAll costs tracked by agent! Check your Tokenr dashboard.")
