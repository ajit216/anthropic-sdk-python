"""
Agent Symphony Testing - RAP-439

This module demonstrates the agent symphony orchestration pattern where multiple
agents work sequentially to complete a workflow: py_developer -> py_review -> git_pr
"""

from __future__ import annotations

from typing import Any

import pytest


class AgentOrchestrator:
    """Orchestrator for managing agent symphony workflow."""

    def __init__(self) -> None:
        self.agents: list[str] = []
        self.completed_tasks: dict[str, Any] = {}
        self.workflow_status = "initialized"

    def register_agent(self, agent_name: str) -> None:
        """Register an agent in the symphony."""
        self.agents.append(agent_name)

    def execute_agent(self, agent_name: str, task: Any) -> dict[str, Any]:
        """Execute a single agent task in the workflow."""
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
        """Execute the full agent symphony workflow."""
        self.workflow_status = "running"

        for agent in self.agents:
            self.execute_agent(agent, task)

        self.workflow_status = "completed"
        return {
            "workflow_status": self.workflow_status,
            "completed_agents": list(self.completed_tasks.keys()),
            "total_agents": len(self.agents),
        }


class TestAgentSymphony:
    """Test suite for agent symphony orchestration."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.orchestrator = AgentOrchestrator()

    def test_orchestrator_initialization(self) -> None:
        """Test that orchestrator initializes correctly."""
        assert self.orchestrator.workflow_status == "initialized"
        assert len(self.orchestrator.agents) == 0
        assert len(self.orchestrator.completed_tasks) == 0

    def test_register_agent(self) -> None:
        """Test registering agents."""
        self.orchestrator.register_agent("py_developer")
        self.orchestrator.register_agent("py_review")
        self.orchestrator.register_agent("git_pr")

        assert len(self.orchestrator.agents) == 3
        assert "py_developer" in self.orchestrator.agents
        assert "py_review" in self.orchestrator.agents
        assert "git_pr" in self.orchestrator.agents

    def test_execute_single_agent(self) -> None:
        """Test executing a single agent task."""
        self.orchestrator.register_agent("py_developer")
        task = {"issue_key": "RAP-439", "type": "story"}

        result = self.orchestrator.execute_agent("py_developer", task)

        assert result["agent"] == "py_developer"
        assert result["status"] == "completed"
        assert "py_developer" in self.orchestrator.completed_tasks

    def test_execute_agent_not_registered(self) -> None:
        """Test that executing unregistered agent raises error."""
        task = {"issue_key": "RAP-439"}

        with pytest.raises(ValueError, match="Agent unknown_agent not registered"):
            self.orchestrator.execute_agent("unknown_agent", task)

    def test_run_full_symphony(self) -> None:
        """Test running the complete agent symphony workflow."""
        # Register agents in order
        self.orchestrator.register_agent("py_developer")
        self.orchestrator.register_agent("py_review")
        self.orchestrator.register_agent("git_pr")

        task = {
            "issue_key": "RAP-439",
            "title": "Agent Symphony Testing - part1",
            "description": "Testing symphony agents",
        }

        result = self.orchestrator.run_symphony(task)

        assert result["workflow_status"] == "completed"
        assert result["total_agents"] == 3
        assert len(result["completed_agents"]) == 3
        assert "py_developer" in result["completed_agents"]
        assert "py_review" in result["completed_agents"]
        assert "git_pr" in result["completed_agents"]

    def test_symphony_workflow_order(self) -> None:
        """Test that agents execute in registered order."""
        execution_order = []

        class TrackingOrchestrator(AgentOrchestrator):
            def execute_agent(self, agent_name: str, task: Any) -> dict[str, Any]:
                execution_order.append(agent_name)
                return super().execute_agent(agent_name, task)

        orchestrator = TrackingOrchestrator()
        orchestrator.register_agent("py_developer")
        orchestrator.register_agent("py_review")
        orchestrator.register_agent("git_pr")

        orchestrator.run_symphony({"test": "task"})

        assert execution_order == ["py_developer", "py_review", "git_pr"]

    def test_completed_tasks_tracking(self) -> None:
        """Test that completed tasks are properly tracked."""
        self.orchestrator.register_agent("py_developer")
        self.orchestrator.register_agent("py_review")

        task1 = {"id": 1}
        task2 = {"id": 2}

        self.orchestrator.execute_agent("py_developer", task1)
        self.orchestrator.execute_agent("py_review", task2)

        assert len(self.orchestrator.completed_tasks) == 2
        assert self.orchestrator.completed_tasks["py_developer"]["task"] == task1
        assert self.orchestrator.completed_tasks["py_review"]["task"] == task2


class TestAgentSymphonyIntegration:
    """Integration tests for agent symphony workflow."""

    def test_rap_439_workflow(self) -> None:
        """Test the RAP-439 agent symphony workflow."""
        orchestrator = AgentOrchestrator()

        # Register all agents
        orchestrator.register_agent("py_developer")
        orchestrator.register_agent("py_review")
        orchestrator.register_agent("git_pr")

        # Define RAP-439 task
        rap_439_task = {
            "issue_key": "RAP-439",
            "status": "READY TO IMPLEMENT",
            "title": "Agent Symphony Testing - part1",
            "description": "This story is being created for testing of symphony agents",
            "acceptance_criteria": "This is acceptance criteria",
        }

        # Execute symphony
        result = orchestrator.run_symphony(rap_439_task)

        # Verify workflow completion
        assert result["workflow_status"] == "completed"
        assert len(orchestrator.completed_tasks) == 3

        # Verify each agent has task data
        for agent_name in ["py_developer", "py_review", "git_pr"]:
            task_record = orchestrator.completed_tasks[agent_name]
            assert task_record["status"] == "completed"
            assert task_record["task"]["issue_key"] == "RAP-439"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
