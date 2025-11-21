from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import numexpr as ne

# --- 1. Setup and Tool Definitions ---

# ⚠️ SECURITY ALERT: Do NOT hardcode API keys in production code. 
# Use os.environ['GOOGLE_API_KEY'] to load it from your environment.
# Set your API key in the environment: export GOOGLE_API_KEY='your_key_here'
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

# Initialize the LLM (Gemini 2.5 Flash is a good choice for tool use)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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

# 1. Create the Agent (defines the logic and prompt)
# This uses the HumanMessage, list of tools, and an LLM to generate the agent's logic.
agent_runnable = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    # The default prompt handles the agent's instructions and history
)

# 2. Create the Agent Executor (manages the execution loop)
# This takes the agent logic and the tools, handling the tool calls and final response generation.
agent_executor = AgentExecutor(
    agent=agent_runnable,
    tools=tools,
    verbose=True # Set to True to see the thought process!
)


# --- 3. Execution ---

prompt_text = """
Which government department is Elon Musk heading currently? 
What is the result of 58 + 92?
"""

print(f"The prompt is: {prompt_text}\n")
print("--- Agent Execution Log ---")

# Invoke the Agent Executor
response = agent_executor.invoke({"input": prompt_text})

print("\n--- Final Answer ---")
# The final answer is in the 'output' key
print(response['output'])