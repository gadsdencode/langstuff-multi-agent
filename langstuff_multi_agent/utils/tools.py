"""
This module defines utility tools for the LangGraph multi-agent AI project.
Each tool is decorated with @tool for LangGraph compatibility and is fully functional.

Tools include:
  - search_web: Web search via SerpAPI.
  - python_repl: Execute Python code safely.
  - read_file: Read file contents.
  - write_file: Write to a file.
  - calendar_tool: Append to a local calendar file.
  - task_tracker_tool: Manage tasks in SQLite.
  - job_search_tool: Job search via SerpAPI.
  - get_current_weather: Weather data via OpenWeatherMap.
  - calc_tool: Safe mathematical evaluation.
  - news_tool: News headlines via NewsAPI.
  - save_memory: Save conversation memories.
  - search_memories: Search stored memories.
"""

import os
import requests
import sqlite3
import io
import contextlib
from langchain_core.tools import tool
from typing import List, Dict, Any
from langstuff_multi_agent.config import get_llm
from langgraph.prebuilt import ToolNode
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage

# Initialize SQLite for task tracking
def init_task_db():
    conn = sqlite3.connect("tasks.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_details TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_task_db()

def has_tool_calls(messages: List[Any]) -> bool:
    """
    Check if the last message in a list contains tool calls.
    """
    if not messages or not isinstance(messages, list):
        return False
    last_message = messages[-1]
    if isinstance(last_message, dict):
        return bool(last_message.get("tool_calls", []) or last_message.get("function_call"))
    elif hasattr(last_message, "tool_calls"):
        return bool(last_message.tool_calls)
    return False

# --- Tools ---
@tool(return_direct=True)
def search_web(query: str) -> str:
    """Perform a web search using SerpAPI."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not set"
    params = {"engine": "google", "q": query, "api_key": api_key, "num": 5}
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        results = [f"{r.get('title', 'No title')}: {r.get('snippet', '')} ({r.get('link', '')})"
                  for r in data.get("organic_results", [])]
        return "\n".join(results) or "No results found"
    except requests.RequestException as e:
        return f"Error performing web search: {str(e)}"

@tool
def python_repl(code: str) -> str:
    """Execute Python code in a restricted environment."""
    safe_builtins = {"print": print, "range": range, "len": len, "str": str, "int": int,
                     "float": float, "bool": bool, "list": list, "dict": dict, "set": set,
                     "min": min, "max": max, "sum": sum}
    restricted_globals = {"__builtins__": safe_builtins}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, restricted_globals, {})
        return output.getvalue() or "Code executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def read_file(filepath: str) -> str:
    """Read content from a file."""
    try:
        with open(filepath, 'r', encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Error reading file '{filepath}': {str(e)}"

@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(filepath, 'w', encoding="utf-8") as file:
            file.write(content)
        return f"Successfully wrote to '{filepath}'"
    except Exception as e:
        return f"Error writing to file '{filepath}': {str(e)}"

@tool
def calendar_tool(event_details: str) -> str:
    """Add an event to a local calendar file."""
    try:
        with open("calendar.txt", "a", encoding="utf-8") as f:
            f.write(event_details + "\n")
        return f"Event added: {event_details}"
    except Exception as e:
        return f"Error updating calendar: {str(e)}"

@tool
def task_tracker_tool(task_details: str) -> str:
    """Add a task to the SQLite task tracker."""
    try:
        conn = sqlite3.connect("tasks.db")
        c = conn.cursor()
        c.execute("INSERT INTO tasks (task_details) VALUES (?)", (task_details,))
        conn.commit()
        task_id = c.lastrowid
        conn.close()
        return f"Task {task_id} added: {task_details}"
    except Exception as e:
        return f"Error adding task: {str(e)}"

@tool(return_direct=True)
def job_search_tool(query: str) -> str:
    """Perform a job search using SerpAPI."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not set"
    params = {"engine": "google_jobs", "q": query, "api_key": api_key}
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        results = [f"{j.get('title', 'No title')} at {j.get('company', 'Unknown')} in {j.get('location', 'Unknown')}: {j.get('snippet', '')}"
                  for j in data.get("job_results", [])]
        return "\n".join(results) or "No job listings found"
    except requests.RequestException as e:
        return f"Error performing job search: {str(e)}"

@tool(return_direct=True)
def get_current_weather(location: str) -> str:
    """Retrieve current weather using OpenWeatherMap."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not set"
    params = {"q": location, "appid": api_key, "units": "imperial"}
    try:
        response = requests.get("http://api.openweathermap.org/data/2.5/weather", params=params)
        response.raise_for_status()
        data = response.json()
        temp = data["main"]["temp"]
        wind_speed = data["wind"]["speed"]
        wind_direction = data["wind"].get("deg", "N/A")
        return f"Current weather in {location}: {temp}°F, wind {wind_speed} mph at {wind_direction}°"
    except requests.RequestException as e:
        return f"Error fetching weather: {str(e)}"

@tool
def calc_tool(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    safe_builtins = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
    try:
        result = eval(expression, {"__builtins__": safe_builtins}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool(return_direct=True)
def news_tool(topic: str) -> str:
    """Retrieve news headlines using NewsAPI."""
    api_key = os.environ.get("NEWSAPI_API_KEY")
    if not api_key:
        return "Error: NEWSAPI_API_KEY not set"
    params = {"q": topic, "apiKey": api_key, "pageSize": 5, "sortBy": "relevancy"}
    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        data = response.json()
        results = [f"{a.get('title', 'No title')} ({a.get('source', {}).get('name', 'Unknown')})"
                  for a in data.get("articles", [])]
        return "\n".join(results) or "No news found"
    except requests.RequestException as e:
        return f"Error fetching news: {str(e)}"

# Memory tools (will be finalized after memory.py)
memory_manager = None  # Placeholder, set in memory.py

@tool
def save_memory(memories: List[Dict[str, str]], config: RunnableConfig) -> str:
    """Save important facts about users or conversations."""
    global memory_manager
    if memory_manager is None:
        from langstuff_multi_agent.utils.memory import MemoryManager
        memory_manager = MemoryManager()
    user_id = config.get("configurable", {}).get("user_id", "global")
    memory_manager.save_memory(user_id, memories)
    return "Memories saved successfully"

@tool
def search_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search long-term conversation memories."""
    global memory_manager
    if memory_manager is None:
        from langstuff_multi_agent.utils.memory import MemoryManager
        memory_manager = MemoryManager()
    user_id = config.get("configurable", {}).get("user_id", "global")
    results = memory_manager.search_memories(user_id, query)
    return [f"{r['subject']} {r['predicate']} {r['object_']}" for r in results]

# Tool node
def get_tools():
    """Return list of all tools."""
    return [
        search_web, python_repl, read_file, write_file, calendar_tool,
        task_tracker_tool, job_search_tool, get_current_weather, calc_tool,
        news_tool, save_memory, search_memories
    ]

tool_node = ToolNode(get_tools())

__all__ = ["tool_node", "has_tool_calls", "get_tools"] + [t.__name__ for t in get_tools()]