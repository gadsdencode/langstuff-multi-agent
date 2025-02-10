# main.py
"""
Main entry point for the LangGraph Multi-Agent AI system.

This script initializes the system by loading configuration from config.py
(which sets up logging and other environment settings) and invoking the
SupervisorAgent to process incoming user requests.
"""

from config import Config  # This import initializes configuration and logging.
from agents.supervisor import SupervisorAgent, compiled_workflows, supervisor_llm


def main():
    print("Welcome to the LangGraph Multi-Agent AI System!")
    # Access a config value to confirm the configuration is loaded.
    print(f"Configuration loaded: Using model {Config.DEFAULT_MODEL} at temperature {Config.DEFAULT_TEMPERATURE}")

    user_request = input("Please enter your request: ")

    # Instantiate the SupervisorAgent with compiled workflows and the classification LLM.
    supervisor_agent = SupervisorAgent(compiled_workflows, supervisor_llm)

    # Route the user request to the appropriate agent workflow.
    response = supervisor_agent.handle_request(user_request)

    print("\nResponse from Agent:")
    print(response)


if __name__ == "__main__":
    main()
