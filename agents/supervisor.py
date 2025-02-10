# agents/supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.

This module uses ChatAnthropic (Claude‑2) to classify incoming user requests and dynamically
routes the request to the appropriate specialized agent workflow. The available agents include:
  DEBUGGER, CONTEXT_MANAGER, PROJECT_MANAGER, PROFESSIONAL_COACH, LIFE_COACH, CODER,
  ANALYST, RESEARCHER, and GENERAL_ASSISTANT.

Each agent workflow is compiled and stored in a dictionary. The SupervisorAgent uses a
classification prompt to decide which agent best fits the user's request, invokes that
workflow with the user's message, and returns the final response.

Reference: LangGraph documentation and examples (&#8203;:contentReference[oaicite:0]{index=0}).
"""

import os
from langgraph.graph import START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Import individual workflows.
from debugger import debugger_workflow
from agents.context_manager import context_manager_workflow
from agents.project_manager import project_manager_workflow
from agents.professional_coach import professional_coach_workflow
from agents.life_coach import life_coach_workflow
from agents.coder import coder_workflow
from agents.analyst import analyst_workflow
from agents.researcher import researcher_workflow
from agents.general_assistant import general_assistant_workflow

# Compile each workflow to obtain runnable instances.
compiled_workflows = {
    "DEBUGGER": debugger_workflow.compile(),
    "CONTEXT_MANAGER": context_manager_workflow.compile(),
    "PROJECT_MANAGER": project_manager_workflow.compile(),
    "PROFESSIONAL_COACH": professional_coach_workflow.compile(),
    "LIFE_COACH": life_coach_workflow.compile(),
    "CODER": coder_workflow.compile(),
    "ANALYST": analyst_workflow.compile(),
    "RESEARCHER": researcher_workflow.compile(),
    "GENERAL_ASSISTANT": general_assistant_workflow.compile(),
}

# Define the available agent options.
AGENT_OPTIONS = [
    "DEBUGGER",
    "CONTEXT_MANAGER",
    "PROJECT_MANAGER",
    "PROFESSIONAL_COACH",
    "LIFE_COACH",
    "CODER",
    "ANALYST",
    "RESEARCHER",
    "GENERAL_ASSISTANT",
]

# Instantiate a classification LLM with zero temperature for deterministic output.
supervisor_llm = ChatAnthropic(model="claude-2", temperature=0)


class SupervisorAgent:
    """
    Routes user requests to the most appropriate specialized agent workflow.

    It uses a classification LLM to analyze the user request and returns one of the
    following agent keys: DEBUGGER, CONTEXT_MANAGER, PROJECT_MANAGER, PROFESSIONAL_COACH,
    LIFE_COACH, CODER, ANALYST, RESEARCHER, or GENERAL_ASSISTANT.
    """
    def __init__(self, workflows: dict, classification_llm):
        """
        :param workflows: Dictionary mapping agent keys to compiled workflow instances.
        :param classification_llm: LLM instance used to classify user requests.
        """
        self.workflows = workflows
        self.classification_llm = classification_llm

    def classify_request(self, user_request: str) -> str:
        """
        Classify the user request to select the best-suited agent.

        Constructs a prompt that lists the available agents and asks the LLM to respond
        with exactly one agent key (case-insensitive).

        :param user_request: The user's input request.
        :return: An agent key from AGENT_OPTIONS.
        """
        prompt = (
            "You are a Supervisor Agent tasked with routing user requests to the most appropriate specialized agent. "
            "The available agents are: " + ", ".join(AGENT_OPTIONS) + ".\n\n"
            "Given the user request below, select the best-suited agent to handle it. "
            "Respond with exactly one of the following options (case-insensitive): DEBUGGER, CONTEXT_MANAGER, "
            "PROJECT_MANAGER, PROFESSIONAL_COACH, LIFE_COACH, CODER, ANALYST, RESEARCHER, GENERAL_ASSISTANT.\n\n"
            f"User Request: \"{user_request}\"\n\n"
            "Your answer:"
        )
        response = self.classification_llm.invoke([HumanMessage(content=prompt)])
        agent_key = response.content.strip().upper()
        if agent_key not in AGENT_OPTIONS:
            agent_key = "GENERAL_ASSISTANT"
        return agent_key

    def handle_request(self, user_request: str) -> str:
        """
        Processes the user request by routing it to the selected agent workflow.

        :param user_request: The user's query.
        :return: The final response from the chosen agent.
        """
        # Build the initial state with the user's message.
        state = {"messages": [HumanMessage(content=user_request)]}

        # Classify the request to choose the appropriate agent.
        agent_key = self.classify_request(user_request)
        print(f"Supervisor routed the request to: {agent_key}")

        # Retrieve the corresponding workflow.
        workflow = self.workflows.get(agent_key)
        if not workflow:
            return "Error: No workflow available for the selected agent."

        # Invoke the agent workflow with the input state.
        result_state = workflow.invoke(state)
        if result_state.get("messages"):
            final_message = result_state["messages"][-1]
            return final_message.content
        else:
            return "Error: The selected agent did not produce any response."


if __name__ == "__main__":
    # Create an instance of SupervisorAgent with the compiled workflows.
    supervisor_agent = SupervisorAgent(compiled_workflows, supervisor_llm)

    # Accept a user request from standard input.
    user_request = input("User Request: ")
    agent_response = supervisor_agent.handle_request(user_request)

    print("\nAgent Response:")
    print(agent_response)
