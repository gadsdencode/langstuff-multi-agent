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
from pydantic import BaseModel

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

# --- Tools with Input Schemas ---

class SearchWebInput(BaseModel):
    query: str

@tool(return_direct=True)
def search_web(input: SearchWebInput) -> str:
    """Perform a web search using SerpAPI."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not set"
    params = {"engine": "google", "q": input.query, "api_key": api_key, "num": 5}
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        results = [f"{r.get('title', 'No title')}: {r.get('snippet', '')} ({r.get('link', '')})"
                  for r in data.get("organic_results", [])]
        return "\n".join(results) or "No results found"
    except requests.RequestException as e:
        return f"Error performing web search: {str(e)}"

class PythonREPLInput(BaseModel):
    code: str

@tool
def python_repl(input: PythonREPLInput) -> str:
    """Execute Python code in a restricted environment."""
    safe_builtins = {"print": print, "range": range, "len": len, "str": str, "int": int,
                     "float": float, "bool": bool, "list": list, "dict": dict, "set": set,
                     "min": min, "max": max, "sum": sum}
    restricted_globals = {"__builtins__": safe_builtins}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(input.code, restricted_globals, {})
        return output.getvalue() or "Code executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"

class ReadFileInput(BaseModel):
    filepath: str

@tool
def read_file(input: ReadFileInput) -> str:
    """Read content from a file."""
    try:
        with open(input.filepath, 'r', encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Error reading file '{input.filepath}': {str(e)}"

class WriteFileInput(BaseModel):
    filepath: str
    content: str

@tool
def write_file(input: WriteFileInput) -> str:
    """Write content to a file."""
    try:
        with open(input.filepath, 'w', encoding="utf-8") as file:
            file.write(input.content)
        return f"Successfully wrote to '{input.filepath}'"
    except Exception as e:
        return f"Error writing to file '{input.filepath}': {str(e)}"

class CalendarInput(BaseModel):
    event_details: str

@tool
def calendar_tool(input: CalendarInput) -> str:
    """Add an event to a local calendar file."""
    try:
        with open("calendar.txt", "a", encoding="utf-8") as f:
            f.write(input.event_details + "\n")
        return f"Event added: {input.event_details}"
    except Exception as e:
        return f"Error updating calendar: {str(e)}"

class TaskTrackerInput(BaseModel):
    task_details: str

@tool
def task_tracker_tool(input: TaskTrackerInput) -> str:
    """Add a task to the SQLite task tracker."""
    try:
        conn = sqlite3.connect("tasks.db")
        c = conn.cursor()
        c.execute("INSERT INTO tasks (task_details) VALUES (?)", (input.task_details,))
        conn.commit()
        task_id = c.lastrowid
        conn.close()
        return f"Task {task_id} added: {input.task_details}"
    except Exception as e:
        return f"Error adding task: {str(e)}"

class JobSearchInput(BaseModel):
    query: str

@tool(return_direct=True)
def job_search_tool(input: JobSearchInput) -> str:
    """Perform a job search using SerpAPI."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not set"
    params = {"engine": "google_jobs", "q": input.query, "api_key": api_key}
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        results = [f"{j.get('title', 'No title')} at {j.get('company', 'Unknown')} in {j.get('location', 'Unknown')}: {j.get('snippet', '')}"
                  for j in data.get("job_results", [])]
        return "\n".join(results) or "No job listings found"
    except requests.RequestException as e:
        return f"Error performing job search: {str(e)}"

class GetWeatherInput(BaseModel):
    location: str

@tool(return_direct=True)
def get_current_weather(input: GetWeatherInput) -> str:
    """Retrieve current weather using OpenWeatherMap."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not set"
    params = {"q": input.location, "appid": api_key, "units": "imperial"}
    try:
        response = requests.get("http://api.openweathermap.org/data/2.5/weather", params=params)
        response.raise_for_status()
        data = response.json()
        temp = data["main"]["temp"]
        wind_speed = data["wind"]["speed"]
        wind_direction = data["wind"].get("deg", "N/A")
        return f"Current weather in {input.location}: {temp}°F, wind {wind_speed} mph at {wind_direction}°"
    except requests.RequestException as e:
        return f"Error fetching weather: {str(e)}"

class CalcInput(BaseModel):
    expression: str

@tool
def calc_tool(input: CalcInput) -> str:
    """Evaluate a mathematical expression safely."""
    safe_builtins = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
    try:
        result = eval(input.expression, {"__builtins__": safe_builtins}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

class NewsInput(BaseModel):
    topic: str

@tool(return_direct=True)
def news_tool(input: NewsInput) -> str:
    """Retrieve news headlines using NewsAPI."""
    api_key = os.environ.get("NEWSAPI_API_KEY")
    if not api_key:
        return "Error: NEWSAPI_API_KEY not set"
    params = {"q": input.topic, "apiKey": api_key, "pageSize": 5, "sortBy": "relevancy"}
    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        data = response.json()
        results = [f"{a.get('title', 'No title')} ({a.get('source', {}).get('name', 'Unknown')})"
                  for a in data.get("articles", [])]
        return "\n".join(results) or "No news found"
    except requests.RequestException as e:
        return f"Error fetching news: {str(e)}"

# Memory tools
memory_manager = None  # Initialized lazily

class SaveMemoryInput(BaseModel):
    memories: List[Dict[str, str]]

@tool
def save_memory(input: SaveMemoryInput, config: RunnableConfig) -> str:
    """Save important facts about users or conversations."""
    global memory_manager
    if memory_manager is None:
        from langstuff_multi_agent.utils.memory import MemoryManager
        memory_manager = MemoryManager()
    # Convert config to dict safely and extract user_id
    config_dict = dict(config) if hasattr(config, 'dict') else {}
    configurable = config_dict.get("configurable", {})
    if not isinstance(configurable, dict):
        configurable = {}
    user_id = configurable.get("user_id", "global")
    memory_manager.save_memory(user_id, input.memories)
    return "Memories saved successfully"

class SearchMemoriesInput(BaseModel):
    query: str

@tool
def search_memories(input: SearchMemoriesInput, config: RunnableConfig) -> List[str]:
    """Search long-term conversation memories."""
    global memory_manager
    if memory_manager is None:
        from langstuff_multi_agent.utils.memory import MemoryManager
        memory_manager = MemoryManager()
    # Convert config to dict safely and extract user_id
    config_dict = dict(config) if hasattr(config, 'dict') else {}
    configurable = config_dict.get("configurable", {})
    if not isinstance(configurable, dict):
        configurable = {}
    user_id = configurable.get("user_id", "global")
    results = memory_manager.search_memories(user_id, input.query)
    return [f"{r['subject']} {r['predicate']} {r['object_']}" for r in results]

# Tool collection and node
def get_tools():
    """Return list of all tools."""
    return [
        search_web, python_repl, read_file, write_file, calendar_tool,
        task_tracker_tool, job_search_tool, get_current_weather, calc_tool,
        news_tool, save_memory, search_memories
    ]

tool_node = ToolNode(get_tools())

# Explicitly define tool names to avoid StructuredTool attribute issues
tool_names = [
    "search_web", "python_repl", "read_file", "write_file", "calendar_tool",
    "task_tracker_tool", "job_search_tool", "get_current_weather", "calc_tool",
    "news_tool", "save_memory", "search_memories"
]

__all__ = ["tool_node", "has_tool_calls", "get_tools"] + tool_names
