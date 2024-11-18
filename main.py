import os
from enum import Enum
from typing import TypedDict, Annotated, List
from uuid import uuid4

from langchain.agents import Tool, AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.gitlab.tool import GitLabAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore

class QueryType(Enum):
    CODE = "code"
    CHAT_MISTRAL = "chat_mistral"
    CHAT_HAIKU = "chat_haiku"
    TOOLS = "tools"
    GITLAB = "gitlab"
    END = "end"


class State(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str


def create_agent_executor(llm: BaseLanguageModel, toollist: List[Tool], system_prompt: str) -> AgentExecutor:
    """Create an agent executor with the given LLM and tools."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=toollist,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=True
    )


def detect_query_type(state: State) -> str:
    """
    Route messages based on content type.
    """
    if isinstance(state, list):
        messages = state
    else:
        messages = state.get("messages", [])

    if not messages:
        return QueryType.CHAT_MISTRAL.value

    last_message = messages[-1]
    if isinstance(last_message, tuple):
        content = last_message[1]
    elif isinstance(last_message, HumanMessage):
        content = last_message.content
    else:
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return QueryType.TOOLS.value
        return QueryType.END.value

    content = content.lower()

    # Add GitLab-specific keywords
    gitlab_keywords = [
        "gitlab", "merge request", "mr", "pipeline", "ci/cd",
        "issue", "repository", "repo", "commit", "branch"
    ]

    if any(keyword in content for keyword in gitlab_keywords):
        return QueryType.GITLAB.value

    code_keywords = [
        "code", "function", "programming", "debug", "error",
        "python", "javascript", "java", "cpp", "c++",
        "algorithm", "compile", "syntax", "git", "github"
    ]

    if any(keyword in content for keyword in code_keywords):
        return QueryType.CODE.value

    message_count = len([m for m in messages if isinstance(m, (tuple, HumanMessage))])
    return QueryType.CHAT_HAIKU.value if message_count % 2 == 0 else QueryType.CHAT_MISTRAL.value


def create_gitlab_agent(llm: BaseLanguageModel) -> AgentExecutor:
    """Create a GitLab-specific agent with proper tool configuration."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a GitLab operations specialist. Use the available tools to interact with GitLab.
        Provide clear and concise responses about the operations performed.

        When using tools:
        1. For listing commits: Use list_commits
        2. For merge requests: Use list_merge_requests
        3. For branches: Use list_branches
        4. For issues: Use list_issues

        Always provide the operation result in a clear format."""),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}")
    ])

    agent = create_openai_tools_agent(llm, gitlab_tools, prompt)

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=gitlab_tools,
        handle_parsing_errors=True,
        max_iterations=2,
        verbose=True
    )


# Update GitLabNode
class GitLabNode:
    """Node for handling GitLab-related operations."""

    def __init__(self, llm: BaseLanguageModel):
        self.agent = create_gitlab_agent(llm)
        print("GitLabNode initialized with new agent configuration")

    def process_state(self, state: State) -> dict:
        """Process GitLab-related queries using the agent executor."""
        try:
            last_message = state["messages"][-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)

            # Map common queries to specific tools
            content_lower = content.lower()
            if "commit" in content_lower:
                tool_name = "list_commits"
            elif "merge" in content_lower:
                tool_name = "list_merge_requests"
            elif "branch" in content_lower:
                tool_name = "list_branches"
            elif "issue" in content_lower:
                tool_name = "list_issues"
            else:
                return {"messages": [AIMessage(
                    content="Please specify what GitLab information you want to retrieve (commits, merge requests, branches, or issues)")]}

            # Execute the appropriate tool
            for tool in gitlab_tools:
                if tool.name == tool_name:
                    result = tool.func("")
                    return {"messages": [AIMessage(content=f"GitLab {tool_name} result:\n{result}")]}

            return {"messages": [AIMessage(content="No matching GitLab operation found.")]}

        except Exception as ex:
            error_msg = f"Error in GitLabNode processing: {type(ex).__name__}: {str(ex)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {"messages": [AIMessage(content=f"Error processing GitLab request:\n{error_msg}")]}

class CodeNode:
    """Node specifically for handling code-related queries with Deepseek."""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def process_state(self, state: State) -> dict:
        """Process code-related queries directly with Deepseek."""
        try:
            last_message = state["messages"][-1]
            if isinstance(last_message, tuple):
                content = last_message[1]
            else:
                content = last_message.content

            messages = [
                SystemMessage(
                    content="You are a code-focused AI assistant. Provide clean, well-documented code and clear explanations."),
                HumanMessage(content=content)
            ]

            response = self.llm.invoke(messages)
            return {"messages": [response]}
        except Exception as ex:
            print(f"Error in CodeNode processing: {str(ex)}")
            return {"messages": [AIMessage(
                content="I encountered an error processing your code request. Could you please rephrase or try again?")]}


class ChatNode:
    """Node for handling chat queries with tools."""

    def __init__(self, agent: AgentExecutor):
        self.agent = agent

    def process_state(self, state: State) -> dict:
        """Process chat queries using the agent executor."""
        try:
            last_message = state["messages"][-1]
            if isinstance(last_message, tuple):
                content = last_message[1]
            else:
                content = last_message.content

            response = self.agent.invoke({"input": content})
            return {"messages": [AIMessage(content=response["output"])]}
        except Exception as ex:
            print(f"Error in ChatNode processing: {str(ex)}")
            return {"messages": [AIMessage(
                content="I encountered an error processing your request. Could you please rephrase or try again?")]}


def stream_graph_updates(u_input: str):
    """Stream graph updates with proper checkpointing configuration."""
    try:
        print(f"stream_graph_updates {u_input}")
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=u_input)],  # Changed from tuple to HumanMessage
            "query_type": "",
        }
        print(f"initial_state {initial_state}")
        config = {
            "configurable": {
                "thread_id": str(uuid4()),
                "checkpoint_ns": "chat_interaction"
            }
        }
        print(f"config {config}")
        for event in graph.stream(initial_state, config=config):
            for value in event.values():
                if "messages" in value:
                    if isinstance(value["messages"][-1], (HumanMessage, SystemMessage, AIMessage)):
                        print("Assistant:", value["messages"][-1].content)
                    elif isinstance(value["messages"][-1], tuple):
                        print("Assistant:", value["messages"][-1][1])
                    else:
                        print("Assistant:", value["messages"][-1])
    except Exception as ex:
        print(f"Error in stream_graph_updates: {str(ex)}")
        print(f"Full error details: ", ex.__dict__)  # Add more error details for debugging

def load_env(file_path=".env"):
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()


def create_gitlab_tool(name: str, description: str, mode: str) -> Tool:
    """Create a GitLab tool with proper run method."""
    gitlab_action = GitLabAction(
        name=name,
        description=description,
        mode=mode
    )

    def tool_function(instructions: str) -> str:
        try:
            return gitlab_action.run(instructions)
        except Exception as e:
            return f"Error executing GitLab operation: {str(e)}"

    return Tool(
        name=name,
        func=tool_function,
        description=description
    )

def create_node_with_logging(node_name, node):
    def logged_process(state):
        print(f"Processing in {node_name} node")
        try:
            result = node.process_state(state)
            print(f"{node_name} processing result: {result}")
            return result
        except Exception as e:
            print(f"Error in {node_name} node: {str(e)}")
            raise

    return logged_process

if __name__ == "__main__":
    load_env()

    # Initialize models
    api_key: str = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError("The ANTHROPIC_API_KEY environment variable is not set.")

    # Initialize LLMs
    haiku = ChatAnthropic(model="claude-3-5-haiku-20241022")
    mistral = ChatOllama(model="mistral")
    deepseek = ChatOllama(model="deepseek-coder-v2")

    # Initialize tools
    ddg = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="search",
        func=ddg.run,
        description="Useful for searching the internet for current information."
    )
    # Initialize GitLab tools
    gitlab_tools = [
        Tool(
            name="list_commits",
            description="List recent commits in the repository",
            func=lambda x: GitLabAction(mode="commits", name="list_commits", description="List commits").run("list")
        ),
        Tool(
            name="list_merge_requests",
            description="List open merge requests",
            func=lambda x: GitLabAction(mode="merge_requests", name="list_merge_requests", description="List MRs").run(
                "list")
        ),
        Tool(
            name="list_branches",
            description="List repository branches",
            func=lambda x: GitLabAction(mode="branches", name="list_branches", description="List branches").run("list")
        ),
        Tool(
            name="list_issues",
            description="List open issues",
            func=lambda x: GitLabAction(mode="issues", name="list_issues", description="List issues").run("list")
        )
    ]

    tools = [search_tool] + gitlab_tools

    # Create agent executors with specific prompts
    haiku_prompt = """You are a helpful AI assistant using Claude Haiku. 
    Use the available tools when needed to provide accurate and up-to-date information.
    Always respond in a clear and concise manner."""

    mistral_prompt = """You are a helpful AI assistant using Mistral. 
    Use the available tools when needed to provide accurate and up-to-date information.
    Always respond in a clear and concise manner."""

    deepseek_prompt = """You are a code-focused AI assistant using Deepseek Coder.
    Analyze code, fix bugs, and provide programming assistance. Use tools when needed.
    Always respond with clean, well-documented code and clear explanations."""

    # Create agents with simplified prompts
    haiku_agent = create_agent_executor(haiku, tools, haiku_prompt)
    mistral_agent = create_agent_executor(mistral, tools, mistral_prompt)
    gitlab_agent = create_agent_executor(
        llm=haiku,  # Using Claude Haiku for GitLab operations
        toollist=gitlab_tools,
        system_prompt="""You are a GitLab operations specialist. 
        When handling GitLab requests:
        1. Understand the user's intent
        2. Select the appropriate GitLab tool
        3. Execute the operation with precise parameters
        4. Format the response clearly

        Available tools:
        - gitlab_issues: Manage GitLab issues
        - gitlab_merge_requests: Handle merge requests
        - gitlab_commits: View and manage commits
        - gitlab_branches: Handle branch operations

        Always provide clear feedback about what operation was performed and its result."""
    )
    # Create nodes
    haiku_node = ChatNode(haiku_agent)
    mistral_node = ChatNode(mistral_agent)
    deepseek_node = CodeNode(deepseek)
    tool_node = ToolNode(tools=tools)
    gitlab_node = GitLabNode(gitlab_agent)

    # Build graph
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("haiku", create_node_with_logging("haiku", haiku_node))
    graph_builder.add_node("mistral", create_node_with_logging("mistral", mistral_node))
    graph_builder.add_node("deepseek", create_node_with_logging("deepseek", deepseek_node))
    graph_builder.add_node("tools", create_node_with_logging("tools", tool_node))
    graph_builder.add_node("gitlab", create_node_with_logging("gitlab", gitlab_node))

    def detect_query_type_with_logging(state: State) -> str:
        print(f"Detecting query type for state: {state}")
        query_type = detect_query_type(state)
        print(f"Detected query type: {query_type}")
        return query_type

    # Add conditional edges from START
    graph_builder.add_conditional_edges(
        START,
        detect_query_type,
        {
            QueryType.CODE.value: "deepseek",
            QueryType.CHAT_MISTRAL.value: "mistral",
            QueryType.CHAT_HAIKU.value: "haiku",
            QueryType.TOOLS.value: "tools",
            QueryType.END.value: END
        }
    )
    # Update edges
    edges = {
        QueryType.CODE.value: "deepseek",
        QueryType.CHAT_MISTRAL.value: "mistral",
        QueryType.CHAT_HAIKU.value: "haiku",
        QueryType.TOOLS.value: "tools",
        QueryType.GITLAB.value: "gitlab",
        QueryType.END.value: END
    }

    # Add edges from START
    graph_builder.add_conditional_edges(
        START,
        detect_query_type_with_logging,
        edges
    )

    # Add edges from each node
    for node in ["haiku", "mistral", "deepseek", "tools", "gitlab"]:
        graph_builder.add_conditional_edges(
            node,
            detect_query_type_with_logging,
            edges
        )

    # Tools edge
    graph_builder.add_conditional_edges(
        "tools",
        detect_query_type,
        {
            QueryType.CODE.value: "deepseek",
            QueryType.CHAT_MISTRAL.value: "mistral",
            QueryType.CHAT_HAIKU.value: "haiku",
            QueryType.TOOLS.value: "tools",
            QueryType.END.value: END
        }
    )
    checkpointer = MemorySaver()
    kvstore = InMemoryStore()
    # Compile graph
    graph = graph_builder.compile(
        checkpointer=checkpointer,
        store=kvstore
    )

    # Test specific GitLab operations
    test_queries = [
        "list commits",
        "show open merge requests",
        "list branches",
        "show recent issues"
    ]

    for query in test_queries:
        print(f"\nTesting query: {query}")
        state = {
            "messages": [HumanMessage(content=query)]
        }
        try:
            result = gitlab_node.process_state(state)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

    print("Multi-LLM chat system initialized. Type 'quit' to exit.")

    # Run interactive loop
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Error details: {str(e)}")
            break