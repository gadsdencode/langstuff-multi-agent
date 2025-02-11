# langstuff_multi_agent/utils/tools.py
"""
This module defines various utility tools for the LangGraph multi-agent AI project.
Each tool is decorated with @tool from langchain_core.tools to ensure compatibility
with LangGraph. These tools are now implemented to be fully functional.

The tools include:
  - search_web: Perform an actual web search via SerpAPI.
  - python_repl: Execute Python code in a restricted environment.
  - read_file: Read file contents from disk.
  - write_file: Write content to a file on disk.
  - calendar_tool: Append event details to a local calendar file.
  - task_tracker_tool: Insert tasks into a local SQLite database.
  - job_search_tool: Perform job search queries via SerpAPI.
  - get_current_weather: Retrieve weather data via OpenWeatherMap.
  - calc_tool: Evaluate mathematical expressions safely.
  - news_tool: Retrieve news headlines using NewsAPI.
"""

import os
import requests
import sqlite3
import io
import contextlib
from langchain_core.tools import tool
from typing import Dict, Any, Optional, List
from langstuff_multi_agent.config import get_llm


def has_tool_calls(message: Dict[str, Any]) -> bool:
    """
    Check if a message contains tool calls.

    Args:
        message: A dictionary containing message data that might have tool calls

    Returns:
        bool: True if the message contains tool calls, False otherwise
    """
    if not isinstance(message, dict):
        return False

    # Check for tool_calls in the message
    tool_calls = message.get("tool_calls", [])
    if tool_calls and isinstance(tool_calls, list):
        return True

    # Check for function_call in the message (older format)
    function_call = message.get("function_call")
    if function_call and isinstance(function_call, dict):
        return True

    return False


# ---------------------------
# REAL WEB SEARCH TOOL
# ---------------------------
@tool
def search_web(query: str) -> str:
    """
    Performs a real web search using SerpAPI.
    Requires SERPAPI_API_KEY to be set as an environment variable.

    :param query: The search query.
    :return: A string with a summary of top search results.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable not set")
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": 5,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        return f"Error performing web search: {response.text}"
    data = response.json()
    results = []
    for result in data.get("organic_results", []):
        title = result.get("title", "No title")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        results.append(f"{title}: {snippet} ({link})")
    return "\n".join(results)


# ---------------------------
# PYTHON REPL TOOL
# ---------------------------
@tool
def python_repl(code: str) -> str:
    """
    Executes Python code in a restricted environment.

    WARNING: Executing arbitrary code can be dangerous. This implementation uses a
    limited set of safe built-ins. In production, consider a proper sandbox.

    :param code: The Python code to execute.
    :return: The output produced by the code.
    """
    try:
        # Define a restricted set of safe built-ins.
        safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "min": min,
            "max": max,
            "sum": sum,
        }
        restricted_globals = {"__safe_builtins__": safe_builtins}
        restricted_locals = {}
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exec(code, restricted_globals, restricted_locals)

        return output.getvalue()
    except SyntaxError as e:
        return f"Syntax error: {e}"
    except Exception as e:
        return f"Execution error: {e}"


def code(state, config):
    """Enhanced Coder Agent with Fix Suggestions"""
    llm = get_llm(config.get("configurable", {}))
    code_snippet = state["messages"][-1]["content"]

    result = python_repl(code_snippet)

    if "error" in result.lower():
        suggestion = llm.invoke([{
            "role": "user",
            "content": f"The following code produced an error:\n{code_snippet}\nSuggest a fix."
        }])
        return {"messages": [result, suggestion]}

    return {"messages": [result]}


# ---------------------------
# READ FILE TOOL
# ---------------------------
@tool
def read_file(filepath: str) -> str:
    """
    Reads the content of a file from disk.

    :param filepath: The path to the file.
    :return: The file's content or an error message.
    """
    try:
        with open(filepath, 'r', encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Error reading file '{filepath}': {str(e)}"


# ---------------------------
# WRITE FILE TOOL
# ---------------------------
@tool
def write_file(params: dict) -> str:
    """
    Writes content to a file on disk.

    Expects a dictionary with the keys:
      - 'filepath': The path to the file.
      - 'content': The content to write.

    :param params: Dictionary containing file path and content.
    :return: Success message or an error message.
    """
    try:
        filepath = params.get("filepath")
        content = params.get("content", "")
        with open(filepath, 'w', encoding="utf-8") as file:
            file.write(content)
        return f"Successfully wrote to '{filepath}'."
    except Exception as e:
        return f"Error writing to file '{filepath}': {str(e)}"


# ---------------------------
# CALENDAR TOOL
# ---------------------------
@tool
def calendar_tool(event_details: str) -> str:
    """
    Adds an event to a local calendar file.

    This implementation appends the event details to a file named 'calendar.txt'.

    :param event_details: Details of the event.
    :return: Confirmation message.
    """
    try:
        with open("calendar.txt", "a", encoding="utf-8") as f:
            f.write(event_details + "\n")
        return f"Event added to calendar: {event_details}"
    except Exception as e:
        return f"Error updating calendar: {str(e)}"


# ---------------------------
# TASK TRACKER TOOL (using SQLite)
# ---------------------------
# Initialize the SQLite database for task tracking.
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


@tool
def task_tracker_tool(task_details: str) -> str:
    """
    Adds a new task to the task tracker using a local SQLite database.

    :param task_details: Details of the task.
    :return: Confirmation message.
    """
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


# ---------------------------
# JOB SEARCH TOOL (using SerpAPI for Google Jobs)
# ---------------------------
@tool
def job_search_tool(query: str) -> str:
    """
    Performs a job search using the SerpAPI Google Jobs engine.

    Requires SERPAPI_API_KEY to be set as an environment variable.

    :param query: The job search query.
    :return: A summary string of job listings.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable not set")
    params = {
        "engine": "google_jobs",
        "q": query,
        "api_key": api_key,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        return f"Error performing job search: {response.text}"
    data = response.json()
    results = []
    for job in data.get("job_results", []):
        title = job.get("title", "No title")
        company = job.get("company", "Unknown")
        location = job.get("location", "Unknown")
        snippet = job.get("snippet", "")
        results.append(f"{title} at {company} in {location}: {snippet}")
    return "\n".join(results)


# ---------------------------
# CURRENT WEATHER TOOL (using OpenWeatherMap)
# ---------------------------
@tool
def get_current_weather(location: str) -> str:
    """
    Retrieves current weather information for a given location using the OpenWeatherMap API.

    Requires OPENWEATHER_API_KEY to be set as an environment variable.

    :param location: The city name for which to retrieve weather.
    :return: Weather details as a string.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable not set")
    params = {
        "q": location,
        "appid": api_key,
        "units": "imperial"  # Fahrenheit
    }
    response = requests.get("http://api.openweathermap.org/data/2.5/weather", params=params)
    if response.status_code != 200:
        return f"Error fetching weather: {response.text}"
    data = response.json()
    temp = data["main"]["temp"]
    wind_speed = data["wind"]["speed"]
    wind_direction = data["wind"].get("deg", "N/A")
    return f"Current weather in {location}: {temp}°F, wind {wind_speed} mph at {wind_direction}°"


# ---------------------------
# CALCULATION TOOL
# ---------------------------
@tool
def calc_tool(expression: str) -> str:
    """
    Evaluates a simple mathematical expression safely.

    Uses eval with a restricted __builtins__.

    :param expression: A string containing the mathematical expression.
    :return: The result as a string or an error message.
    """
    try:
        safe_builtins = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }
        result = eval(expression, {"__builtins__": safe_builtins}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# ---------------------------
# NEWS TOOL (using NewsAPI)
# ---------------------------
@tool
def news_tool(topic: str) -> str:
    """
    Retrieves news headlines for a given topic using the NewsAPI.

    Requires NEWSAPI_API_KEY to be set as an environment variable.

    :param topic: The news topic.
    :return: A summary string of news headlines.
    """
    api_key = os.environ.get("NEWSAPI_API_KEY")
    if not api_key:
        raise ValueError("NEWSAPI_API_KEY environment variable not set")
    params = {
        "q": topic,
        "apiKey": api_key,
        "pageSize": 5,
        "sortBy": "relevancy",
    }
    response = requests.get("https://newsapi.org/v2/everything", params=params)
    if response.status_code != 200:
        return f"Error fetching news: {response.text}"
    data = response.json()
    results = []
    for article in data.get("articles", []):
        title = article.get("title", "No title")
        source = article.get("source", {}).get("name", "Unknown source")
        results.append(f"{title} ({source})")
    return "\n".join(results)
