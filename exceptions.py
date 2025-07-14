"""
Custom exception classes for the Allora RAG system
"""


class AlloraAgentError(Exception):
    """Base exception for Allora agent errors"""
    pass


class ToolExecutionError(AlloraAgentError):
    """Error during tool execution"""
    pass


class ChartGenerationError(ToolExecutionError):
    """Error during chart generation"""
    pass


class ImageGenerationError(ToolExecutionError):
    """Error during image generation"""
    pass


class RAGQueryError(AlloraAgentError):
    """Error during RAG query processing"""
    pass


class SlackIntegrationError(AlloraAgentError):
    """Error in Slack integration"""
    pass


class ConfigurationError(AlloraAgentError):
    """Error in system configuration"""
    pass


class ValidationError(AlloraAgentError):
    """Error in input validation"""
    pass