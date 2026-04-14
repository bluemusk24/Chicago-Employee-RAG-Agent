import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dotenv import load_dotenv, find_dotenv
from typing import Literal
from evals_tools import RetrieverTools

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
_ = load_dotenv(find_dotenv())


class AgentState(MessagesState):
    query: str


retriever = RetrieverTools()


@tool
def lexical_search(query: str) -> list:
    """BM25 keyword search with CrossEncoder reranking. Use for exact keyword matches."""
    results = retriever.lexical_search(query)
    return [str(r['document'].page_content) for r in results]

@tool
def dense_search(query: str) -> list:
    """Dense vector search with LLM compression and reranking. Use for semantic search."""
    results = retriever.dense_search(query)
    return [str(r['page_content']) for r in results]

@tool
def graph_search(query: str) -> list:
    """Graph traversal retrieval using metadata relationships. Use for related roles and departments."""
    results = retriever.graph_search(query)
    return [str(doc.page_content) for doc in results]

tools = [lexical_search, dense_search, graph_search]
tools_by_name = {tool.name: tool for tool in tools}

llm = ChatOllama(model="qwen3-coder-next:cloud")
llm_with_tools = llm.bind_tools(tools, tool_choice='any')


persona = "You are an expert HR data analyst specializing in municipal employee records and organizational structures.\n"
instruction = "Answer questions about Chicago city employee data by ALWAYS using ALL THREE retrieval tools before giving a final answer.\n"
context = (
    "You have access to three retrieval tools and must call all three for every query:\n"
    "1. lexical_search: for exact keyword matches on names, job titles, and departments.\n"
    "2. dense_search: for contextual compression using LLM extraction, for the semantic understanding of the query.\n"
    "3. graph_search: for finding related roles, departments, and organizational relationships.\n"
    "Do NOT stop after one tool. Always call all three tools and combine their results.\n"
)
data_format = (
    "After calling all three tools, combine and deduplicate the results by employee name. "
    "Present your final answer clearly with employee name, job title, department, full time or part time, and salary. "
    "Only include employees that appear in the tool results. Do NOT invent or assume any employee data. "
    "Exclude employees whose department does not match the query. "
    "Do not give a final answer until all three tools have been called.\n"
)
audience = "The responses are designed for analysts and administrators who need quick, accurate insights from Chicago city employee records.\n"
tone = "The tone should be professional, concise, and data-driven.\n"

sys_msg = SystemMessage(content=persona + instruction + context + data_format + audience + tone)


def llm_assistant(state: AgentState):
    """LLM decides whether to call a tool or not."""
    return {
        "messages": [
            llm_with_tools.invoke([sys_msg] + state["messages"])
        ]
    }


def tool_handler(state: AgentState):
    """Performs the tool call."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}


def should_continue(state: AgentState) -> Literal["tool_handler", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
        return "tool_handler"
    return END


graph_workflow = StateGraph(AgentState)
graph_workflow.add_node("llm_assistant", llm_assistant)
graph_workflow.add_node("tool_handler", tool_handler)
graph_workflow.add_edge(START, "llm_assistant")
graph_workflow.add_conditional_edges(
    "llm_assistant",
    should_continue,
    {
        "tool_handler": "tool_handler",
        END: END,
    },
)
graph_workflow.add_edge("tool_handler", "llm_assistant")

# Compile Graph
graph = graph_workflow.compile()