import os
import sys

import httpx
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

def create_ollama_agent(base_url="http://localhost:11434", model_name="llama2"):
    """
    Create a LangChain agent using a remote Ollama instance

    Args:
        base_url (str): URL of the Ollama instance
        model_name (str): Name of the model to use

    Returns:
        AgentExecutor: Configured agent ready to use
    """

    # Initialize the Ollama chat model
    llm = ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=0.5,
    )

    # Define tools the agent can use
    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for searching the internet to find answers to questions.  Do not exceed a question of more than 2 sentences.  Input should be a search query."
        )
    ]

    # Create prompt template
    prompt = PromptTemplate.from_template("""You are a helpful AI assistant. Answer to Human only in English.
    Use the following tools to answer user questions:
    {tools}

    Available tools: {tool_names}

    To use a tool, please use the following format:
    Thought: I need to do something
    Action: tool_name
    Action Input: input for the tool
    Observation: tool output
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I know what to respond
    Final Answer: Final response to human.  Please provide the response entirely in English without mixing other languages. 

    Current conversation:
    Human: {input}
    Assistant: Let's approach this step by step:
    {agent_scratchpad}
    """)

    # Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=1,
        handle_parsing_errors=True
    )

    return agent_executor


def check_ollama_connection(base_url):
    """
    Check if Ollama is accessible at the given URL
    """
    try:
        resp = httpx.get(f"{base_url}/api/tags")
        if resp.status_code == 200:
            return True
        return False
    except httpx.ConnectError:
        return False
    except Exception as e:
        print(f"Error checking Ollama connection: {e}")
        return False

# Example usage
if __name__ == "__main__":
    def load_env(file_path=".env"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The environment file {file_path} does not exist.")
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    load_env()
    ollama_intranet_url: str = os.environ.get('OLLAMA_INTRANET_URL')
    if check_ollama_connection(ollama_intranet_url):
        print("Successfully connected to Ollama")
    else:
        print("Could not connect to Ollama")
        sys.exit(1)
    # Create agent connected to remote Ollama instance
    agent = create_ollama_agent(
        base_url=ollama_intranet_url,  # Replace with your remote server
        model_name="qwq"
    )

    # Test the agent
    response = agent.invoke({"input": "What's the latest news about AI?"})
    print(response["output"])
