"""
Test cases for Symphony Agent Module (RAP-439)
"""

import pytest
from anthropic.agents import SymphonyAgent, IssueInfo


class TestIssueInfo:
    """Test cases for IssueInfo dataclass."""

    def test_issue_info_creation(self):
        """Test creating an IssueInfo object."""
        issue = IssueInfo(
            issue_key="RAP-439",
            summary="Agent Symphony Testing - part1",
            issue_type="Story",
            description="This story is being created for testing of symphony agents",
            status="READY TO IMPLEMENT",
            priority="Major",
            assignee="harshal.more@forcepoint.com",
            acceptance_criteria="This is acceptance criteria",
        )
        assert issue.issue_key == "RAP-439"
        assert issue.summary == "Agent Symphony Testing - part1"
        assert issue.issue_type == "Story"
        assert issue.status == "READY TO IMPLEMENT"
        assert issue.priority == "Major"

    def test_issue_info_optional_fields(self):
        """Test IssueInfo with optional fields as None."""
        issue = IssueInfo(
            issue_key="RAP-440",
            summary="Test Issue",
            issue_type="Bug",
            description="Test description",
            status="OPEN",
            priority="Low",
        )
        assert issue.assignee is None
        assert issue.acceptance_criteria is None


class TestSymphonyAgent:
    """Test cases for SymphonyAgent class."""

    @pytest.fixture
    def agent(self):
        """Fixture providing a SymphonyAgent instance."""
        return SymphonyAgent()

    @pytest.fixture
    def sample_issue_data(self):
        """Fixture providing sample issue data."""
        return {
            "issue_key": "RAP-439",
            "summary": "Agent Symphony Testing - part1",
            "issue_type": "Story",
            "description": "This story is being created for testing of symphony agents",
            "status": "READY TO IMPLEMENT",
            "priority": "Major",
            "assignee": "harshal.more@forcepoint.com",
            "acceptance_criteria": "This is acceptance criteria",
        }

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.name == "SymphonyAgent"
        assert agent.issue_cache == {}

    def test_agent_custom_name(self):
        """Test agent initialization with custom name."""
        agent = SymphonyAgent(name="CustomAgent")
        assert agent.name == "CustomAgent"

    def test_register_issue(self, agent, sample_issue_data):
        """Test registering an issue."""
        issue = agent.register_issue(sample_issue_data)
        assert issue.issue_key == "RAP-439"
        assert issue.summary == "Agent Symphony Testing - part1"
        assert issue.issue_key in agent.issue_cache

    def test_get_issue(self, agent, sample_issue_data):
        """Test retrieving a registered issue."""
        agent.register_issue(sample_issue_data)
        retrieved_issue = agent.get_issue("RAP-439")
        assert retrieved_issue is not None
        assert retrieved_issue.issue_key == "RAP-439"

    def test_get_nonexistent_issue(self, agent):
        """Test retrieving a non-existent issue."""
        result = agent.get_issue("RAP-999")
        assert result is None

    def test_list_issues(self, agent, sample_issue_data):
        """Test listing all registered issues."""
        agent.register_issue(sample_issue_data)
        issues = agent.list_issues()
        assert len(issues) == 1
        assert issues[0].issue_key == "RAP-439"

    def test_display_issue(self, agent, sample_issue_data):
        """Test displaying issue information."""
        issue = agent.register_issue(sample_issue_data)
        display = agent.display_issue(issue)
        assert "RAP-439" in display
        assert "Agent Symphony Testing - part1" in display
        assert "READY TO IMPLEMENT" in display

    def test_process_issue(self, agent, sample_issue_data, capsys):
        """Test processing an issue."""
        agent.register_issue(sample_issue_data)
        result = agent.process_issue("RAP-439")
        assert result is True
        captured = capsys.readouterr()
        assert "RAP-439" in captured.out

    def test_process_nonexistent_issue(self, agent):
        """Test processing a non-existent issue."""
        result = agent.process_issue("RAP-999")
        assert result is False

    def test_multiple_issues(self, agent):
        """Test handling multiple issues."""
        issues_data = [
            {
                "issue_key": "RAP-439",
                "summary": "Issue 1",
                "issue_type": "Story",
                "description": "Description 1",
                "status": "OPEN",
                "priority": "High",
            },
            {
                "issue_key": "RAP-440",
                "summary": "Issue 2",
                "issue_type": "Bug",
                "description": "Description 2",
                "status": "IN_PROGRESS",
                "priority": "Medium",
            },
        ]

        for issue_data in issues_data:
            agent.register_issue(issue_data)

        all_issues = agent.list_issues()
        assert len(all_issues) == 2
        assert agent.get_issue("RAP-439") is not None
        assert agent.get_issue("RAP-440") is not None
