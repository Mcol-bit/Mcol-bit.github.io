from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
import numexpr as ne
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()

# --- 1. Setup and Tool Definitions ---

# ⚠️ SECURITY ALERT: Do NOT hardcode API keys in production code. 
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)

# Initialize the search tool
ddg_search = DuckDuckGoSearchResults()

# Create a simple math evaluation tool using numexpr
@tool
def calculate(expression: str) -> str:
    """A highly accurate calculator tool for mathematical expressions. 
    Input should be a numerical expression string (e.g., '10 + 5 * 2')."""
    try:
        # Evaluate the expression safely
        result = ne.evaluate(expression).item()
        return str(result)
    except Exception as e:
        # Return a clear error message if calculation fails
        return f"Error calculating the expression '{expression}': {e}"

# Pass the tools to the agent  
tools = [ddg_search, calculate]

# --- 2. Agent Initialization ---

# Create the agent using langgraph
agent_executor = create_react_agent(llm, tools)
# --- 3. Execution ---

prompt_text = """
What are the personality traits of a shiba inu? 
What is the difference of 90-100?
"""

print(f"The prompt is: {prompt_text}\n")
print("="*50)

# Execute
for chunk in agent_executor.stream(
    {"messages": [("user", prompt_text)]},
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()

print("="*50)