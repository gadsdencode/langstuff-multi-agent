# agents/supervisor.py
"""
Supervisor Agent module for integrating and routing individual LangGraph agent workflows.

This module uses an LLM—instantiated via the get_llm() factory function from
langstuff_multi_agent/config.py—to classify incoming user requests and dynamically
route the request to the appropriate specialized agent workflow. The available agents include:
  DEBUGGER, CONTEXT_MANAGER, PROJECT_MANAGER, PROFESSIONAL_COACH, LIFE_COACH, CODER,
  ANALYST, RESEARCHER, and GENERAL_ASSISTANT.

Each agent workflow is compiled with persistent checkpointing enabled by explicitly
passing the shared checkpointer (Config.PERSISTENT_CHECKPOINTER) during compilation.
"""

from langgraph.graph import START, END
from langchain_core.messages import HumanMessage
from langstuff_multi_agent.config import Config, get_llm

# Import individual workflows.
from langstuff_multi_agent.agents.debugger import debugger_workflow
from langstuff_multi_agent.agents.context_manager import context_manager_workflow
from langstuff_multi_agent.agents.project_manager import project_manager_workflow
from langstuff_multi_agent.agents.professional_coach import professional_coach_workflow
from langstuff_multi_agent.agents.life_coach import life_coach_workflow
from langstuff_multi_agent.agents.coder import coder_workflow
from langstuff_multi_agent.agents.analyst import analyst_workflow
from langstuff_multi_agent.agents.researcher import researcher_workflow
from langstuff_multi_agent.agents.general_assistant import general_assistant_workflow

# Compile each workflow with the persistent checkpointer.
compiled_workflows = {
    "DEBUGGER": debugger_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER),
    "CONTEXT_MANAGER": context_manager_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER),
    "PROJECT_MANAGER": project_manager_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER),
    "PROFESSIONAL_COACH": professional_coach_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER),
    "LIFE_COACH": life_coach_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER),
    "CODER": coder_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER),
    "ANALYST": analyst_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER),
    "RESEARCHER": researcher_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER),
    "GENERAL_ASSISTANT": general_assistant_workflow.compile(checkpointer=Config.PERSISTENT_CHECKPOINTER)
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
    "GENERAL_ASSISTANT"
]

# Instantiate a classification LLM using the get_llm() factory.
supervisor_llm = get_llm()


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

        Constructs a prompt listing the available agents and asks the LLM to respond
        with exactly one agent key (case-insensitive).

        :param user_request: The user's input request.
        :return: An agent key from AGENT_OPTIONS.
        """
        prompt = (
            "You are a Supervisor Agent tasked with routing user requests to the most appropriate specialized agent. "
            "The available agents are: " + ", ".join(AGENT_OPTIONS) + ".\n\n"
            "Given the user request below, select the best-suited agent to handle it. "
            "Respond with exactly one of the following options (case-insensitive): "
            "DEBUGGER, CONTEXT_MANAGER, PROJECT_MANAGER, PROFESSIONAL_COACH, LIFE_COACH, "
            "CODER, ANALYST, RESEARCHER, GENERAL_ASSISTANT.\n\n"
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
        state = {"messages": [HumanMessage(content=user_request)]}
        agent_key = self.classify_request(user_request)
        print(f"Supervisor routed the request to: {agent_key}")
        workflow = self.workflows.get(agent_key)
        if not workflow:
            return "Error: No workflow available for the selected agent."
        result_state = workflow.invoke(state)
        if result_state.get("messages"):
            final_message = result_state["messages"][-1]
            return final_message.content
        else:
            return "Error: The selected agent did not produce any response."


if __name__ == "__main__":
    supervisor_agent = SupervisorAgent(compiled_workflows, supervisor_llm)
    user_request = input("User Request: ")
    agent_response = supervisor_agent.handle_request(user_request)
    print("\nAgent Response:")
    print(agent_response)
