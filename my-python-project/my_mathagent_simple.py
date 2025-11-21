from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
import numexpr as ne
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup API key
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# Initialize the LLM (using gemini-2.0-flash-exp for the latest model)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-exp", temperature=0)

# Initialize the search tool
ddg_search = DuckDuckGoSearchResults()

# Create a calculator tool
@tool
def calculate(expression: str) -> str:
    """Calculator for math expressions. Input should be like '10 + 5 * 2'"""
    try:
        result = ne.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Create the agent using langgraph
from langgraph.prebuilt import create_react_agent

tools = [ddg_search, calculate]
agent_executor = create_react_agent(llm, tools)

# Run the agent
prompt = """
Which government department is Elon Musk heading currently? 
What is the result of 58 + 92?
"""

print(f"Prompt: {prompt}\n")
print("="*50)

# Execute
for chunk in agent_executor.stream(
    {"messages": [("user", prompt)]},
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()

print("="*50)
