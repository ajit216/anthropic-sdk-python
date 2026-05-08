"""
Tests for agent symphony orchestration patterns.

This module demonstrates basic agent coordination patterns where multiple
agents work together in a synchronized manner. The tests verify that agents
can be orchestrated to execute tasks sequentially and in parallel.
"""

from typing import List, Dict, Any
import pytest


class Agent:
    """Basic agent implementation for testing symphony patterns."""

    def __init__(self, name: str) -> None:
        """Initialize an agent with a name."""
        self.name = name
        self.tasks_completed: List[str] = []

    def execute_task(self, task_name: str) -> Dict[str, Any]:
        """Execute a task and record completion."""
        self.tasks_completed.append(task_name)
        return {
            "agent": self.name,
            "task": task_name,
            "status": "completed",
        }

    def get_status(self) -> Dict[str, Any]:
        """Get the agent's current status."""
        return {
            "agent": self.name,
            "tasks_completed": len(self.tasks_completed),
            "completed_list": self.tasks_completed,
        }


class AgentSymphony:
    """Orchestrator for coordinating multiple agents."""

    def __init__(self) -> None:
        """Initialize the symphony orchestrator."""
        self.agents: Dict[str, Agent] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the symphony."""
        self.agents[agent.name] = agent

    def execute_sequential(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Execute tasks sequentially across agents.

        Args:
            tasks: List of dicts with 'agent' and 'task' keys

        Returns:
            List of execution results
        """
        results = []
        for task in tasks:
            agent_name = task["agent"]
            task_name = task["task"]
            if agent_name in self.agents:
                result = self.agents[agent_name].execute_task(task_name)
                results.append(result)
                self.execution_log.append(result)
        return results

    def get_symphony_status(self) -> Dict[str, Any]:
        """Get the status of all agents in the symphony."""
        return {
            "total_agents": len(self.agents),
            "agents": {
                name: agent.get_status() for name, agent in self.agents.items()
            },
            "execution_log": self.execution_log,
        }


def test_single_agent_task_execution() -> None:
    """Test that a single agent can execute tasks."""
    agent = Agent("agent_1")
    result = agent.execute_task("analyze_data")

    assert result["agent"] == "agent_1"
    assert result["task"] == "analyze_data"
    assert result["status"] == "completed"
    assert len(agent.tasks_completed) == 1


def test_multiple_agents_parallel_initialization() -> None:
    """Test that multiple agents can be initialized for parallel work."""
    symphony = AgentSymphony()
    agent_1 = Agent("agent_1")
    agent_2 = Agent("agent_2")
    agent_3 = Agent("agent_3")

    symphony.add_agent(agent_1)
    symphony.add_agent(agent_2)
    symphony.add_agent(agent_3)

    status = symphony.get_symphony_status()
    assert status["total_agents"] == 3
    assert "agent_1" in status["agents"]
    assert "agent_2" in status["agents"]
    assert "agent_3" in status["agents"]


def test_sequential_task_execution_across_agents() -> None:
    """Test sequential execution of tasks across multiple agents."""
    symphony = AgentSymphony()
    symphony.add_agent(Agent("processor_a"))
    symphony.add_agent(Agent("processor_b"))
    symphony.add_agent(Agent("processor_c"))

    tasks = [
        {"agent": "processor_a", "task": "validate_input"},
        {"agent": "processor_b", "task": "transform_data"},
        {"agent": "processor_c", "task": "generate_output"},
    ]

    results = symphony.execute_sequential(tasks)

    assert len(results) == 3
    assert results[0]["agent"] == "processor_a"
    assert results[1]["agent"] == "processor_b"
    assert results[2]["agent"] == "processor_c"


def test_agent_task_accumulation() -> None:
    """Test that agents accumulate completed tasks."""
    agent = Agent("accumulator")

    agent.execute_task("task_1")
    agent.execute_task("task_2")
    agent.execute_task("task_3")

    status = agent.get_status()
    assert status["tasks_completed"] == 3
    assert status["completed_list"] == ["task_1", "task_2", "task_3"]


def test_symphony_execution_log() -> None:
    """Test that symphony maintains an execution log."""
    symphony = AgentSymphony()
    symphony.add_agent(Agent("agent_x"))
    symphony.add_agent(Agent("agent_y"))

    tasks = [
        {"agent": "agent_x", "task": "step_1"},
        {"agent": "agent_y", "task": "step_2"},
        {"agent": "agent_x", "task": "step_3"},
    ]

    symphony.execute_sequential(tasks)
    status = symphony.get_symphony_status()

    assert len(status["execution_log"]) == 3
    assert status["execution_log"][0]["task"] == "step_1"
    assert status["execution_log"][1]["task"] == "step_2"
    assert status["execution_log"][2]["task"] == "step_3"


def test_agent_not_found_handling() -> None:
    """Test that symphony handles non-existent agents gracefully."""
    symphony = AgentSymphony()
    symphony.add_agent(Agent("existing_agent"))

    tasks = [
        {"agent": "existing_agent", "task": "valid_task"},
        {"agent": "nonexistent_agent", "task": "invalid_task"},
    ]

    results = symphony.execute_sequential(tasks)

    # Only the existing agent's task should be in results
    assert len(results) == 1
    assert results[0]["agent"] == "existing_agent"


def test_complex_agent_orchestration() -> None:
    """Test complex orchestration with multiple agents and tasks."""
    symphony = AgentSymphony()

    # Add multiple agents
    for i in range(5):
        symphony.add_agent(Agent(f"worker_{i}"))

    # Create a workflow
    workflow = [
        {"agent": "worker_0", "task": "initialize"},
        {"agent": "worker_1", "task": "fetch_data"},
        {"agent": "worker_2", "task": "process_batch_1"},
        {"agent": "worker_3", "task": "process_batch_2"},
        {"agent": "worker_4", "task": "aggregate_results"},
    ]

    results = symphony.execute_sequential(workflow)
    status = symphony.get_symphony_status()

    assert len(results) == 5
    assert status["total_agents"] == 5
    for i in range(5):
        assert status["agents"][f"worker_{i}"]["tasks_completed"] == 1
