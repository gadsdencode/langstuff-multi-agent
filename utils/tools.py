# my_agent/utils/tools.py
"""
This module defines various utility tools for the LangGraph multi-agent AI.
Each tool is decorated with @tool from langchain_core.tools for compatibility
with LangGraph. The tools include:

  - search_web: Simulates a web search.
  - python_repl: Simulates executing Python code.
  - read_file: Reads file contents.
  - write_file: Writes content to a file.
  - calendar_tool: Simulates calendar updates.
  - task_tracker_tool: Simulates task tracking updates.
  - job_search_tool: Simulates job search results.
  - get_current_weather: Simulates current weather information.
  - calc_tool: Evaluates a mathematical expression.
  - news_tool: Simulates retrieving news headlines.
"""

from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """
    Simulates a web search for the given query.

    :param query: The search query.
    :return: A simulated search result.
    """
    return f"Simulated search result for query: '{query}'."


@tool
def python_repl(code: str) -> str:
    """
    Simulates executing Python code.

    :param code: Python code as a string.
    :return: Simulated output from executing the code.
    """
    try:
        # WARNING: In production, executing code via eval/exec can be dangerous.
        # Here, we simulate code execution without running untrusted code.
        return f"Simulated execution output for code: '{code}'."
    except Exception as e:
        return f"Error during simulated code execution: {str(e)}"


@tool
def read_file(filepath: str) -> str:
    """
    Reads the content of a file.

    :param filepath: The path to the file.
    :return: The file's content or an error message.
    """
    try:
        with open(filepath, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file '{filepath}': {str(e)}"


@tool
def write_file(params: dict) -> str:
    """
    Writes content to a file.

    Expects a dictionary with the keys:
      - 'filepath': The path to the file.
      - 'content': The content to write.

    :param params: Dictionary containing file path and content.
    :return: Success message or an error message.
    """
    try:
        filepath = params.get("filepath")
        content = params.get("content", "")
        with open(filepath, 'w') as file:
            file.write(content)
        return f"Successfully wrote to '{filepath}'."
    except Exception as e:
        return f"Error writing to file '{filepath}': {str(e)}"


@tool
def calendar_tool(event_details: str) -> str:
    """
    Simulates updating a calendar with event details.

    :param event_details: Details of the event.
    :return: Confirmation message.
    """
    return f"Calendar updated with event: {event_details}"


@tool
def task_tracker_tool(task_details: str) -> str:
    """
    Simulates updating a task tracker with task details.

    :param task_details: Details of the task.
    :return: Confirmation message.
    """
    return f"Task tracker updated with task: {task_details}"


@tool
def job_search_tool(query: str) -> str:
    """
    Simulates a job search based on the given query.

    :param query: The job search query.
    :return: Simulated job listings.
    """
    return f"Simulated job listings for query: '{query}'."


@tool
def get_current_weather(location: str) -> str:
    """
    Simulates retrieving current weather information for a given location.

    :param location: The location for which to retrieve weather.
    :return: Simulated weather details.
    """
    return f"Simulated weather for {location}: 75°F, Sunny."


@tool
def calc_tool(expression: str) -> str:
    """
    Evaluates a simple mathematical expression.

    :param expression: A string containing the mathematical expression.
    :return: The result as a string or an error message.
    """
    try:
        # Evaluate the expression safely using a restricted namespace.
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def news_tool(topic: str) -> str:
    """
    Simulates retrieving news headlines for a given topic.

    :param topic: The news topic.
    :return: Simulated news headlines.
    """
    return f"Simulated news headlines for topic: '{topic}'."
