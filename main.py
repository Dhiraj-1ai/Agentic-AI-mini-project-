"""
Simple LangChain Agent — Calculator + Web Search
Uses:
1. Calculator tools (add, subtract, multiply, divide)
2. DuckDuckGo web search
3. Hugging Face Inference API
"""

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

# Load .env file
load_dotenv()

# -----------------------------
# Calculator Tools
# -----------------------------
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float):
    """Divide a by b."""
    if b == 0:
        return "Error: Cannot divide by zero."
    return a / b


# -----------------------------
# Web Search Tool
# -----------------------------
search = DuckDuckGoSearchRun()

# -----------------------------
# Hugging Face Model
# -----------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="model="microsoft/Phi-3-mini-4k-instruct"",
    provider="auto",          # don't force hf-inference
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=endpoint)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Use the calculator for math and DuckDuckGo for web searches whenever needed."
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# -----------------------------
# Build Agent
# -----------------------------
# Build Agent
agent = create_agent(
    model=model,
    tools=[add, subtract, multiply, divide, search],
    system_prompt="You are a helpful AI assistant."
)

# Chat loop
print("Simple AI Assistant (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ("exit", "quit"):
        break

    result = agent.invoke({
    "messages": [{"role": "user", "content": user_input}]
})

    print("\nAssistant:", result["messages"][-1].content, "\n")

# -----------------------------
# Run Chat
# -----------------------------
if __name__ == "__main__":
    print("Simple AI Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            result = agent_executor.invoke({"input": user_input})
            print("\nAssistant:", result["output"], "\n")
        except Exception as e:
            print("\nError:", e, "\n")