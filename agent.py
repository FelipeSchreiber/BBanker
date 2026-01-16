import os
import gradio as gr
from openai import AzureOpenAI

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", model_name)

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

def chat_with_agent(message, history):
    """
    Chat function for Gradio interface
    """
    # Build messages list from history
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Get response from Azure OpenAI
    response = client.chat.completions.create(
        messages=messages,
        max_completion_tokens=16384,
        model=deployment
    )
    
    return response.choices[0].message.content

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="BBanker AI Assistant",
    description="Chat with the AI assistant powered by Azure OpenAI",
    examples=[
        "I am going to Paris, what should I see?",
        "What are the best restaurants in Tokyo?",
        "Tell me about the history of Rome"
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)