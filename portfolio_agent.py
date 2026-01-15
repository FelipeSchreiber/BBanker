import os
import gradio as gr
from openai import AzureOpenAI
import json
import base64
from PIL import Image
from io import BytesIO
from portfolio_tools import (
    load_returns_from_file,
    calculate_optimal_portfolio,
    generate_efficient_frontier,
    format_portfolio_results
)

endpoint = "https://bbanker-foundry.cognitiveservices.azure.com/"
model_name = "gpt-5-nano"
deployment = "gpt-5-nano"

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Store uploaded file data
portfolio_data = {
    'returns': None
}


def upload_file(file):
    """Handle CSV/ODS file upload."""
    if file is None:
        return "No file uploaded."
    
    try:
        returns = load_returns_from_file(file.name)
        portfolio_data['returns'] = returns
        
        assets = list(returns.columns)
        num_observations = len(returns)
        
        file_type = "ODS" if file.name.endswith('.ods') else "CSV"
        return f"✅ {file_type} loaded successfully!\n\n**Assets found:** {len(assets)}\n**Observations:** {num_observations}\n\n**Assets:**\n" + "\n".join([f"- {asset}" for asset in assets[:10]]) + ("\n- ..." if len(assets) > 10 else "")
    except Exception as e:
        return f"❌ Error loading file: {str(e)}"


def optimize_portfolio_tool(risk_free_rate):
    """
    Tool to calculate optimal portfolio allocation.
    
    Args:
        risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%)
    """
    if portfolio_data['returns'] is None:
        return {"error": "No data loaded. Please upload a file first."}
    
    try:
        rfr = float(risk_free_rate)
        optimal = calculate_optimal_portfolio(portfolio_data['returns'], rfr)
        return optimal
    except Exception as e:
        return {"error": f"Error calculating optimal portfolio: {str(e)}"}


def efficient_frontier_tool(risk_free_rate=None):
    """
    Tool to generate efficient frontier visualization.
    
    Args:
        risk_free_rate: Optional annual risk-free rate to mark optimal portfolio
    """
    if portfolio_data['returns'] is None:
        return {"error": "No data loaded. Please upload a file first."}
    
    try:
        rfr = float(risk_free_rate) if risk_free_rate else None
        image_base64 = generate_efficient_frontier(portfolio_data['returns'], rfr)
        return {"image": image_base64}
    except Exception as e:
        return {"error": f"Error generating efficient frontier: {str(e)}"}


# Define tools for the AI agent
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_optimal_portfolio",
            "description": "Calculate the optimal portfolio allocation that maximizes the Sharpe ratio given a risk-free rate. Returns the optimal weights for each asset, annual return, annual volatility, and Sharpe ratio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_free_rate": {
                        "type": "number",
                        "description": "The annual risk-free rate as a decimal (e.g., 0.05 for 5%). This is required to calculate the Sharpe ratio and find the optimal portfolio."
                    }
                },
                "required": ["risk_free_rate"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_efficient_frontier",
            "description": "Generate a visualization of the efficient frontier, showing the optimal risk-return tradeoffs. If a risk-free rate is provided, it will also mark the optimal portfolio on the chart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_free_rate": {
                        "type": "number",
                        "description": "Optional annual risk-free rate as a decimal (e.g., 0.05 for 5%). If provided, the optimal portfolio will be marked on the efficient frontier."
                    }
                },
                "required": []
            }
        }
    }
]


def chat_with_agent(message, history):
    """
    Chat function for Gradio interface with tool calling support.
    Returns a tuple of (text_response, image) for display.
    """
    # Check if file is loaded
    if portfolio_data['returns'] is None and any(keyword in message.lower() for keyword in ['portfolio', 'optimal', 'efficient', 'allocation', 'sharpe']):
        return "⚠️ Please upload a CSV or ODS file with stock returns first using the file upload section above.", None
    
    # Build messages list from history
    messages = [
        {
            "role": "system",
            "content": """You are a portfolio optimization assistant. You help users analyze their stock portfolios and find optimal asset allocations.

When a user uploads a CSV or ODS file with stock returns, you can:
1. Calculate the optimal portfolio allocation that maximizes the Sharpe ratio (requires risk-free rate)
2. Generate efficient frontier visualizations

IMPORTANT: 
- Always ask for the risk-free rate when users want portfolio optimization
- The risk-free rate should be annual and in decimal form (e.g., 0.05 for 5%)
- Explain results in clear, professional terms
- When presenting optimal portfolios, highlight the key metrics (return, volatility, Sharpe ratio) and the asset allocation

If users ask about portfolio optimization but haven't provided a risk-free rate, ask them for it."""
        }
    ]
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Get response from Azure OpenAI with tools
    response = client.chat.completions.create(
        messages=messages,
        max_completion_tokens=16384,
        model=deployment,
        tools=tools,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    image_to_display = None
    
    # Handle tool calls
    if assistant_message.tool_calls:
        # Add assistant message with tool calls to history
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in assistant_message.tool_calls
            ]
        })
        
        # Execute tool calls
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "calculate_optimal_portfolio":
                result = optimize_portfolio_tool(function_args.get("risk_free_rate"))
                
                # Add tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
                
            elif function_name == "generate_efficient_frontier":
                result = efficient_frontier_tool(function_args.get("risk_free_rate"))
                
                # Store image for display
                if "image" in result:
                    image_to_display = result["image"]
                
                # Add tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"status": "Image generated successfully"} if "image" in result else result)
                })
        
        # Get final response after tool execution
        final_response = client.chat.completions.create(
            messages=messages,
            max_completion_tokens=16384,
            model=deployment
        )
        
        # Return response with image if available
        response_text = final_response.choices[0].message.content
        if image_to_display:
            # Convert base64 to PIL Image
            image_data = base64.b64decode(image_to_display)
            image = Image.open(BytesIO(image_data))
            return response_text, image
        return response_text, None
    
    return assistant_message.content, None


# Create Gradio interface
with gr.Blocks(title="BBanker Portfolio Optimization") as demo:
    gr.Markdown("# 📊 BBanker Portfolio Optimization Assistant")
    gr.Markdown("Upload a CSV or ODS file with stock returns and get optimal portfolio allocations powered by AI.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload Returns Data")
            file_upload = gr.File(label="Upload CSV or ODS file (first column: dates, rest: stock returns)", file_types=[".csv", ".ods"])
            upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=10)
            
            file_upload.change(
                fn=upload_file,
                inputs=[file_upload],
                outputs=[upload_status]
            )
            
            gr.Markdown("""
            ### 📝 File Format
            - **Supported formats:** CSV, ODS
            - First column: Dates
            - Remaining columns: Stock returns
            - Delimiter (CSV): semicolon (;) or comma (,)
            - Decimal separator: comma (,) or period (.)
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with Portfolio Assistant")
            
            chatbot_messages = gr.Chatbot(label="Chat History", height=400)
            chart_output = gr.Image(label="Efficient Frontier")
            
            msg = gr.Textbox(
                label="Message",
                placeholder="Ask about portfolio optimization...",
                lines=1
            )
            
            gr.Examples(
                examples=[
                    "What is the optimal portfolio allocation with a 5% risk-free rate?",
                    "Show me the efficient frontier",
                    "Calculate the optimal portfolio with a 3% risk-free rate and show the efficient frontier",
                    "What are the key metrics of the optimal portfolio?"
                ],
                inputs=msg
            )
            
            def user_submit(message, history):
                if not message:
                    return history, ""
                # Add user message
                history.append({"role": "user", "content": message})
                return history, ""
            
            def bot_respond(history):
                if not history or history[-1]["role"] != "user":
                    return history, None
                
                user_message = history[-1]["content"]
                # Get conversation history for agent (excluding last user message)
                conversation_history = []
                for msg in history[:-1]:
                    if msg["role"] == "user":
                        conversation_history.append([msg["content"], None])
                    elif msg["role"] == "assistant" and conversation_history:
                        conversation_history[-1][1] = msg["content"]
                
                bot_response, image = chat_with_agent(user_message, conversation_history)
                
                # Add bot response
                history.append({"role": "assistant", "content": bot_response})
                return history, image
            
            msg.submit(user_submit, [msg, chatbot_messages], [chatbot_messages, msg]).then(
                bot_respond, [chatbot_messages], [chatbot_messages, chart_output]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
