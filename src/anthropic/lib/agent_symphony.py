"""
Agent Symphony - Orchestration pattern for coordinating multiple agents

This module provides a framework for orchestrating multiple agents working together
sequentially in a symphony-like pattern. Each agent completes its task before the
next agent begins, allowing for coordinated workflows.

Example usage:
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent("py_developer")
    orchestrator.register_agent("code_reviewer")
    orchestrator.register_agent("deployment")
    
    task = {"description": "Build feature X"}
    result = orchestrator.run_symphony(task)
"""

from __future__ import annotations

from typing import Any


class AgentOrchestrator:
    """Orchestrator for managing agent symphony workflow.
    
    This class manages the sequential execution of multiple agents,
    tracking their completion status and aggregating results.
    """

    def __init__(self) -> None:
        """Initialize the orchestrator with empty state."""
        self.agents: list[str] = []
        self.completed_tasks: dict[str, Any] = {}
        self.workflow_status = "initialized"

    def register_agent(self, agent_name: str) -> None:
        """Register an agent in the symphony.
        
        Args:
            agent_name: The name of the agent to register.
        """
        self.agents.append(agent_name)

    def execute_agent(self, agent_name: str, task: Any) -> dict[str, Any]:
        """Execute a single agent task in the workflow.
        
        Args:
            agent_name: The name of the agent to execute.
            task: The task data to pass to the agent.
            
        Returns:
            A dictionary containing the execution result.
            
        Raises:
            ValueError: If the agent is not registered.
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not registered")

        result = {
            "agent": agent_name,
            "task": task,
            "status": "completed",
            "output": f"Task completed by {agent_name}",
        }

        self.completed_tasks[agent_name] = result
        return result

    def run_symphony(self, task: Any) -> dict[str, Any]:
        """Execute the full agent symphony workflow.
        
        Runs all registered agents sequentially, passing the same task
        through each agent in order.
        
        Args:
            task: The task data to process through the symphony.
            
        Returns:
            A dictionary summarizing the workflow execution.
        """
        self.workflow_status = "running"

        for agent in self.agents:
            self.execute_agent(agent, task)

        self.workflow_status = "completed"
        return {
            "workflow_status": self.workflow_status,
            "completed_agents": list(self.completed_tasks.keys()),
            "total_agents": len(self.agents),
        }

    def get_status(self) -> str:
        """Get the current workflow status.
        
        Returns:
            The current workflow status.
        """
        return self.workflow_status

    def get_agent_result(self, agent_name: str) -> dict[str, Any] | None:
        """Get the result from a specific agent.
        
        Args:
            agent_name: The name of the agent.
            
        Returns:
            The agent's task result or None if not found.
        """
        return self.completed_tasks.get(agent_name)

    def reset(self) -> None:
        """Reset the orchestrator to initial state."""
        self.agents = []
        self.completed_tasks = {}
        self.workflow_status = "initialized"


__all__ = ["AgentOrchestrator"]
