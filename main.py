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

from tool_docker import create_docker_tools
from tool_graph import create_graph_visualization_tool


class QueryType(Enum):
    CODE = "code"
    CHAT_LLAMA = "chat_llama"
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

    agent = create_openai_tools_agent(llm, toollist, prompt)

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
        return QueryType.CHAT_LLAMA.value  # Changed from CHAT_MISTRAL

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

    # Define system/ops related keywords that should NOT go to the code model
    ops_keywords = [
        "docker", "kubernetes", "k8s", "logs", "monitoring", "deployment",
        "container", "pod", "service", "cluster", "node", "volume",
        "namespace", "config", "configuration", "system", "environment",
        "infrastructure", "server", "cloud", "aws", "azure", "gcp"
    ]

    # Check for ops-related content first
    if any(keyword in content for keyword in ops_keywords):
        return QueryType.CHAT_LLAMA.value  # Changed from CHAT_MISTRAL

    code_keywords = [
        "code", "function", "programming", "debug", "error",
        "python", "javascript", "java", "cpp", "c++",
        "algorithm", "compile", "syntax", "git", "github"
    ]

    if any(keyword in content for keyword in code_keywords):
        return QueryType.CODE.value

    message_count = len([m for m in messages if isinstance(m, (tuple, HumanMessage))])
    return QueryType.CHAT_HAIKU.value if message_count % 2 == 0 else QueryType.CHAT_LLAMA.value  # Changed from CHAT_MISTRAL



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

class GitLabNode:
    """Node for handling GitLab-related operations."""

    def __init__(self, llm: BaseLanguageModel):
        self.agent = create_gitlab_agent(llm)
        self.tools = {tool.name: tool for tool in gitlab_tools}
        print("GitLabNode initialized with new agent configuration")

    def process_state(self, state: State) -> dict:
        """Process GitLab-related queries using the appropriate tool."""
        try:
            last_message = state["messages"][-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            content_lower = content.lower()

            # Map queries to specific tools and format inputs appropriately
            if "show" in content_lower or "list" in content_lower:
                if "issue" in content_lower:
                    return self._execute_tool("list_issues", "")
                elif "detail" in content_lower and "issue" in content_lower:
                    issue_number = self._extract_issue_number(content)
                    return self._execute_tool("get_issue", str(issue_number))
            elif "comment" in content_lower and "issue" in content_lower:
                issue_number = self._extract_issue_number(content)
                comment_text = self._extract_comment_text(content)
                formatted_input = f"{issue_number}\n\n{comment_text}"
                return self._execute_tool("comment_on_issue", formatted_input)
            elif "create" in content_lower:
                if "file" in content_lower:
                    file_path, content = self._extract_file_info(content)
                    formatted_input = f"{file_path}\n{content}"
                    return self._execute_tool("create_file", formatted_input)
                elif "pull request" in content_lower or "pr" in content_lower:
                    title, body = self._extract_pr_info(content)
                    formatted_input = f"{title}\n{body}"
                    return self._execute_tool("create_pull_request", formatted_input)
            elif "read" in content_lower:
                file_path = self._extract_file_path(content)
                return self._execute_tool("read_file", file_path)
            elif "update" in content_lower:
                file_path, old_content, new_content = self._extract_update_info(content)
                formatted_input = f"{file_path}\nOLD <<<<\n{old_content}\n>>>> OLD\nNEW <<<<\n{new_content}\n>>>> NEW"
                return self._execute_tool("update_file", formatted_input)
            elif "delete" in content_lower:
                file_path = self._extract_file_path(content)
                return self._execute_tool("delete_file", file_path)

            return {"messages": [AIMessage(content="Please specify a valid GitLab operation (list issues, create file, etc.)")]}

        except Exception as ex:
            error_msg = f"Error in GitLabNode processing: {type(ex).__name__}: {str(ex)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {"messages": [AIMessage(content=f"Error processing GitLab request:\n{error_msg}")]}

    def _execute_tool(self, tool_name: str, tool_input: str) -> dict:
        """Execute a specific GitLab tool with the given input."""
        if tool_name in self.tools:
            result = self.tools[tool_name].func(tool_input)
            return {"messages": [AIMessage(content=f"GitLab {tool_name} result:\n{result}")]}
        return {"messages": [AIMessage(content="No matching GitLab operation found.")]}

    def _extract_issue_number(self, content: str) -> int:
        """Extract issue number from content."""
        # Simple extraction - in practice, you'd want more robust parsing
        try:
            import re
            numbers = re.findall(r'\d+', content)
            return int(numbers[0]) if numbers else 1
        except:
            return 1

    def _extract_comment_text(self, content: str) -> str:
        """Extract comment text from content."""
        try:
            import re
            match = re.search(r"saying\s+'([^']*)'", content)
            return match.group(1) if match else "No comment text provided"
        except:
            return "No comment text provided"

    def _extract_file_info(self, content: str) -> tuple:
        """Extract file path and content from message."""
        try:
            import re
            file_match = re.search(r"file\s+called\s+(\S+)", content)
            file_path = file_match.group(1) if file_match else "README.md"
            return file_path, "# Auto-generated content\n\nThis file was created via GitLab API."
        except:
            return "README.md", "# Auto-generated content"

    def _extract_pr_info(self, content: str) -> tuple:
        """Extract PR title and body from content."""
        return "Automated PR", "This PR was created via GitLab API"

    def _extract_file_path(self, content: str) -> str:
        """Extract file path from content."""
        try:
            import re
            match = re.search(r"(?:read|update|delete)\s+(?:the\s+)?(?:contents\s+of\s+)?(\S+)", content)
            return match.group(1) if match else ""
        except:
            return ""

    def _extract_update_info(self, content: str) -> tuple:
        """Extract file path and update content information."""
        try:
            file_path = self._extract_file_path(content)
            # In a real implementation, you'd want to fetch the current content
            old_content = "Current content"
            new_content = "Updated content"
            return file_path, old_content, new_content
        except:
            return "", "", ""

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
        print("\n=== Stream Graph Updates ===")
        # print(f"stream_graph_updates {u_input}")
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=u_input)],  # Changed from tuple to HumanMessage
            "query_type": "",
        }
        # print(f"initial_state {initial_state}")
        config = {
            "configurable": {
                "thread_id": str(uuid4()),
                "checkpoint_ns": "chat_interaction"
            }
        }
        # print(f"config {config}")
        print("Starting graph stream...")
        for event in graph.stream(initial_state, config=config):
            print(f"Processing event: {event}")
            for value in event.values():
                if "messages" in value:
                    message = value["messages"][-1]
                    print(f"Message type: {type(message)}")
                    if isinstance(message, (HumanMessage, SystemMessage, AIMessage)):
                        print("Assistant:", message.content)
                    elif isinstance(message, tuple):
                        print("Assistant:", message[1])
                    else:
                        print("Assistant:", message)
    except Exception as ex:
        print(f"Error in stream_graph_updates: {str(ex)}")
        print(f"Full error details: ", ex.__dict__)  # Add more error details for debugging
        import traceback
        traceback.print_exc()

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
    print("Initializing language models...")
    haiku = ChatAnthropic(model="claude-3-5-haiku-20241022")
    llama = ChatOllama(model="llama3.2")
    deepseek = ChatOllama(model="deepseek-coder-v2")

    # Initialize tools
    print("Setting up tools...")
    ddg = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="search",
        func=ddg.run,
        description="Useful for searching the internet for current information."
    )
    # Initialize GitLab tools
    gitlab_tools = [
        Tool(
            name="list_issues",
            description="List all open issues in the repository. Returns issue titles and numbers.",
            func=lambda x: GitLabAction(
                mode="get_issues",
                name="list_issues",
                description="List all open issues"
            ).run("")
        ),

        Tool(
            name="get_issue",
            description="Get details about a specific issue including its title, body, and first 10 comments. Input should be an issue number.",
            func=lambda x: GitLabAction(
                mode="get_issue",
                name="get_issue",
                description="Get specific issue details"
            ).run(x)
        ),

        Tool(
            name="comment_on_issue",
            description="Add a comment to a specific issue. Input format should be: 'issue_number\\n\\ncomment_text'",
            func=lambda x: GitLabAction(
                mode="comment_on_issue",
                name="comment_on_issue",
                description="Comment on an issue"
            ).run(x)
        ),

        Tool(
            name="create_file",
            description="Create a new file in the repository. Input format should be: 'file_path\\nfile_contents'",
            func=lambda x: GitLabAction(
                mode="create_file",
                name="create_file",
                description="Create a new file"
            ).run(x)
        ),

        Tool(
            name="create_pull_request",
            description="Create a new pull request. Input format should be: 'PR_title\\nPR_body'",
            func=lambda x: GitLabAction(
                mode="create_pull_request",
                name="create_pull_request",
                description="Create a pull request"
            ).run(x)
        ),

        Tool(
            name="read_file",
            description="Read the contents of a file from the repository. Input should be the file path.",
            func=lambda x: GitLabAction(
                mode="read_file",
                name="read_file",
                description="Read file contents"
            ).run(x)
        ),

        Tool(
            name="update_file",
            description="Update an existing file. Input format should be: 'file_path\\nOLD <<<<\\nold_content\\n>>>> OLD\\nNEW <<<<\\nnew_content\\n>>>> NEW'",
            func=lambda x: GitLabAction(
                mode="update_file",
                name="update_file",
                description="Update file contents"
            ).run(x)
        ),

        Tool(
            name="delete_file",
            description="Delete a file from the repository. Input should be the file path.",
            func=lambda x: GitLabAction(
                mode="delete_file",
                name="delete_file",
                description="Delete a file"
            ).run(x)
        )
    ]

    # Add graph visualization tool
    graph_viz_tool = create_graph_visualization_tool()
    # docker log
    docker_tools = create_docker_tools()
    # Combine all tools
    tools = [search_tool, graph_viz_tool] + gitlab_tools + docker_tools
    for tool in tools:
        print(f"Tool name: {tool.name}")

    # Create agent executors with specific prompts
    docker_prompt = """When handling Docker-related queries:
    1. For listing containers, use the docker_ps tool
    2. For analyzing logs, use the docker_log_analysis tool with parameters:
       - container_name: Name of the container
       - time_range_minutes: How far back to look (default: 60)
       - filters: Any specific terms to filter for
       - max_lines: Maximum number of lines to analyze (default: 1000)
    3. Provide clear explanations of the results"""

    haiku_prompt = """You are a helpful AI assistant using Claude Haiku. 
    Use the available tools when needed to provide accurate and up-to-date information.
    You can use the visualize_graph tool to create visual representations of graph structures.
    You can use the gitlab tool to interact with GitLab.
    Always respond in a clear and concise manner."""

    llama_prompt = """You are a helpful AI assistant using Llama 3.2.  # Changed from Mistral
    Use the available tools when needed to provide accurate and up-to-date information.
    You can use the visualize_graph tool to create visual representations of graph structures.
    You can use the gitlab tool to interact with GitLab.
    Always respond in a clear and concise manner."""

    llama_prompt += "\n" + docker_prompt
    haiku_prompt += "\n" + docker_prompt

    deepseek_prompt = """You are a code-focused AI assistant using Deepseek Coder.
    Analyze code, fix bugs, and provide programming assistance. Use tools when needed.
    Always respond with clean, well-documented code and clear explanations."""

    # Create agents with simplified prompts
    haiku_agent = create_agent_executor(haiku, tools, haiku_prompt)
    llama_agent = create_agent_executor(llama, tools, llama_prompt)  # Changed from mistral_agent
    gitlab_agent = create_agent_executor(
        llm=haiku,
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
    print("Creating nodes...")
    haiku_node = ChatNode(haiku_agent)
    llama_node = ChatNode(llama_agent)  # Changed from mistral_node
    deepseek_node = CodeNode(deepseek)
    tool_node = ToolNode(tools=tools)
    gitlab_node = GitLabNode(haiku)

    # Build graph
    print("Setting up graph...")
    graph_builder = StateGraph(State)

    # Add nodes
    nodes = {
        "haiku": haiku_node,
        "llama": llama_node,  # Changed from "mistral"
        "deepseek": deepseek_node,
        "tools": tool_node,
        "gitlab": gitlab_node
    }

    # Add each node with logging wrapper
    for node_name, node in nodes.items():
        graph_builder.add_node(node_name, create_node_with_logging(node_name, node))
        print(f"Added node: {node_name}")

    def detect_query_type_with_logging(state: State) -> str:
        print(f"Detecting query type for state: {state}")
        query_type = detect_query_type(state)
        print(f"Detected query type: {query_type}")
        return query_type

    # Update edges
    edges = {
        QueryType.CODE.value: "deepseek",
        QueryType.CHAT_LLAMA.value: "llama",
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

    # Add edges from each node to all possible destinations
    for node_name in nodes.keys():
        graph_builder.add_conditional_edges(
            node_name,
            detect_query_type_with_logging,
            edges
        )
        print(f"Added edges for node: {node_name}")

    checkpointer = MemorySaver()
    kvstore = InMemoryStore()
    # Compile graph
    print("\nCompiling graph...")
    graph = graph_builder.compile(
        checkpointer=checkpointer,
        store=kvstore
    )
    print("Graph compilation complete")
    # Test specific GitLab operations
    # test_queries = [
    #     "show all open issues",  # Tests get_issues mode
    #     # "get details for issue 1",  # Tests get_issue mode
    #     # "add a comment to issue 2 saying 'Working on this'",  # Tests comment_on_issue mode
    #     # "create a new file called README.md with some content",  # Tests create_file mode
    #     # "create a pull request to merge feature branch",  # Tests create_pull_request mode
    #     # "read the contents of config.json",  # Tests read_file mode
    #     # "update the LICENSE file to add current year",  # Tests update_file mode
    #     # "delete the temporary.txt file",  # Tests delete_file mode
    #     # "get details for issue number 123",
    #     # "add a comment to issue 45 saying 'Working on this task'",
    #     # "create a file called docs/README.md with initial documentation",
    #     # "create a pull request titled 'Feature Update' for the feature branch",
    #     # "read the contents of src/config.json",
    #     # "update the LICENSE file to change 2023 to 2024",
    #     # "delete the temporary/test.txt file"
    # ]
    # for query in test_queries:
    #     print(f"\nTesting query: {query}")
    #     state = {
    #         "messages": [HumanMessage(content=query)]
    #     }
    #     try:
    #         result = gitlab_node.process_state(state)
    #         print(f"Result: {result}")
    #     except Exception as e:
    #         print(f"Error: {e}")

    # test_query = "list open mr"
    # print("\nTesting with full debug output:")
    # stream_graph_updates(test_query)
    # LangGraph visualization
    mermaid_diagram = graph_viz_tool.func(graph_builder, "Current System Structure")
    print(mermaid_diagram)

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