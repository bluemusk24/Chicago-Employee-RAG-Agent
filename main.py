import gradio as gr
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from loguru import logger
from prometheus_client import start_http_server, Summary
from typing import Literal
from dotenv import load_dotenv, find_dotenv

from agent_tools.evals_tools import RetrieverTools


_ = load_dotenv(find_dotenv())

# Logging
logger.add("chicago_employee.log", rotation="1 MB")

# Prometheus
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

# FastAPI app
app = FastAPI(
    title="Chicago Employee RAG Agent",
    description="Agentic RAG for Chicago Employee Data using Lexical, Dense and Graph Retrieval",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever and tools
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

# LLM
llm = ChatOllama(model="qwen3-coder-next:cloud")
llm_with_tools = llm.bind_tools(tools, tool_choice='any')

# System message
sys_msg = SystemMessage(content=(
    "You are an expert HR data analyst specializing in municipal employee records and organizational structures.\n"
    "Answer questions about Chicago city employee data by ALWAYS using ALL THREE retrieval tools before giving a final answer.\n"
    "You have access to three retrieval tools and must call all three for every query:\n"
    "1. lexical_search: for exact keyword matches on names, job titles, and departments.\n"
    "2. dense_search: for contextual compression using LLM extraction and semantic understanding of the query.\n"
    "3. graph_search: for finding related roles, departments, and organizational relationships.\n"
    "Do NOT stop after one tool. Always call all three tools and combine their results.\n"
    "After calling all three tools, combine and deduplicate the results by employee name. "
    "Present your final answer clearly with employee name, job title, department, full time or part time, and salary. "
    "Only include employees that appear in the tool results. Do NOT invent or assume any employee data. "
    "Exclude employees whose department does not match the query. "
    "Do not give a final answer until all three tools have been called.\n"
    "The responses are designed for analysts and administrators who need quick, accurate insights from Chicago city employee records.\n"
    "The tone should be professional, concise, and data-driven.\n"
))

# Agent state
class AgentState(MessagesState):
    query: str

# LLM node
def llm_assistant(state: AgentState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Tool handler
def tool_handler(state: AgentState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

# Routing
def should_continue(state: AgentState) -> Literal["tool_handler", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
        return "tool_handler"
    return END


# Build graph
graph_workflow = StateGraph(AgentState)
graph_workflow.add_node("llm_assistant", llm_assistant)
graph_workflow.add_node("tool_handler", tool_handler)
graph_workflow.add_edge(START, "llm_assistant")
graph_workflow.add_conditional_edges(
    "llm_assistant",
    should_continue,
    {"tool_handler": "tool_handler", END: END},
)
graph_workflow.add_edge("tool_handler", "llm_assistant")
graph = graph_workflow.compile()


# Core agent function
@REQUEST_TIME.time()
def run_agent(query: str) -> str:
    try:
        logger.info(f"Running agent for query: {query}")
        response = graph.invoke({"messages": [HumanMessage(content=query)]})
        final_message = response["messages"][-1].content
        logger.info(f"Agent response: {final_message}")
        return final_message
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        return f"Error processing query: {str(e)}"
    
    
# FastAPI endpoint
@app.get("/")
def root():
    return {"message": "Chicago Employee RAG Agent is running"}

@app.post("/query")
def query_endpoint(query: str):
    logger.info(f"Received query: {query}")
    result = run_agent(query)
    return {"query": query, "response": result}


# Gradio UI
def gradio_query(query: str) -> str:
    """Gradio interface function."""
    return run_agent(query)


gradio_app = gr.Interface(
    fn=gradio_query,
    inputs=[
        gr.Textbox(
            label="Query",
            placeholder="e.g., List all investigators in the Office of Inspector General",
            lines=3
        )
    ],
    outputs=[
        gr.Textbox(label="Agent Response", lines=15)
    ],
    title="Chicago Employee RAG Agent",
    description="Query Chicago city employee records using Lexical, Dense, and Graph Retrieval"
)

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


def main():
    start_http_server(8000)   # Prometheus metrics
    logger.info("Starting Chicago Employee RAG Agent...")
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()