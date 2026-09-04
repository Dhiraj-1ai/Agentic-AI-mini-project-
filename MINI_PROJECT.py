"""
Simple LangChain Agent — Calculator + Web Search
--------------------------------------------------
Uses AgentExecutor with two tools:
  1. A calculator (add, subtract, multiply, divide) built with @tool
  2. DuckDuckGo web search (no API key needed)

Install requirements:
    pip install langchain langchain-ollama langchain-community duckduckgo-search

Make sure Ollama is running locally with the model pulled:
    ollama pull llama3.1:latest
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_tool_calling_agent, AgentExecutor


# -----------------------------
# 1. Calculator Tools
# -----------------------------
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second. Returns an error message if dividing by zero."""
    if b == 0:
        return "Error: cannot divide by zero."
    return a / b


# -----------------------------
# 2. Web Search Tool
# -----------------------------
search = DuckDuckGoSearchRun()


# -----------------------------
# 3. Model
# -----------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(
    llm = endpoint
)

# To use OpenAI instead, comment the above and uncomment below:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -----------------------------
# 4. Custom Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a simple, helpful AI assistant. "
               "You have access to a calculator and a web search tool. "
               "Use these tools whenever they would help answer the question accurately — "
               "for example, use the calculator for any math, and web search for "
               "current events or facts you're unsure about. "
               "If a tool isn't needed, just answer directly."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


# -----------------------------
# 5. Build the Agent
# -----------------------------
tools = [add, subtract, multiply, divide, search]

agent = create_tool_calling_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# -----------------------------
# 6. Try it out
# -----------------------------
if __name__ == "__main__":
    print("Simple AI Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        result = agent_executor.invoke({"input": user_input})
        print("\nAssistant:", result["output"], "\n")