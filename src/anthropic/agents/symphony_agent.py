"""
Symphony Agent Module - Part 1 Testing

This module implements a simple symphony agent for testing integration
with issue tracking systems (e.g., JIRA).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class IssueInfo:
    """Data class to hold issue information."""
    issue_key: str
    summary: str
    issue_type: str
    description: str
    status: str
    priority: str
    assignee: Optional[str] = None
    acceptance_criteria: Optional[str] = None


class SymphonyAgent:
    """
    A simple symphony agent that processes and manages issue information.
    """

    def __init__(self, name: str = "SymphonyAgent"):
        """Initialize the symphony agent."""
        self.name = name
        self.issue_cache: Dict[str, IssueInfo] = {}

    def register_issue(self, issue_data: Dict[str, Any]) -> IssueInfo:
        """
        Register an issue with the agent.

        Args:
            issue_data: Dictionary containing issue information

        Returns:
            IssueInfo: The registered issue information
        """
        issue = IssueInfo(
            issue_key=issue_data.get("issue_key", ""),
            summary=issue_data.get("summary", ""),
            issue_type=issue_data.get("issue_type", ""),
            description=issue_data.get("description", ""),
            status=issue_data.get("status", ""),
            priority=issue_data.get("priority", ""),
            assignee=issue_data.get("assignee"),
            acceptance_criteria=issue_data.get("acceptance_criteria"),
        )
        self.issue_cache[issue.issue_key] = issue
        return issue

    def get_issue(self, issue_key: str) -> Optional[IssueInfo]:
        """
        Retrieve issue information by key.

        Args:
            issue_key: The issue key to retrieve

        Returns:
            IssueInfo if found, None otherwise
        """
        return self.issue_cache.get(issue_key)

    def display_issue(self, issue: IssueInfo) -> str:
        """
        Format issue information for display.

        Args:
            issue: The issue to display

        Returns:
            Formatted issue information
        """
        output = f"""
        ========================================
        Issue: {issue.issue_key}
        ========================================
        Summary: {issue.summary}
        Type: {issue.issue_type}
        Priority: {issue.priority}
        Status: {issue.status}
        Description: {issue.description}
        Assignee: {issue.assignee or 'Unassigned'}
        Acceptance Criteria: {issue.acceptance_criteria or 'N/A'}
        ========================================
        """
        return output

    def list_issues(self) -> list:
        """Get all registered issues."""
        return list(self.issue_cache.values())

    def process_issue(self, issue_key: str) -> bool:
        """
        Process an issue by key.

        Args:
            issue_key: The issue key to process

        Returns:
            True if processing was successful, False otherwise
        """
        issue = self.get_issue(issue_key)
        if not issue:
            return False
        print(self.display_issue(issue))
        return True
