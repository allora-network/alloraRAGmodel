"""
Configuration management for Allora RAG system
Centralizes all hardcoded values and provides environment-based configuration
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from exceptions import ConfigurationError


@dataclass
class AgentConfig:
    """Configuration for the Allora RAG Agent"""
    # LLM Configuration
    model: str = "gpt-5.2"  # or "gpt-5.1", "gpt-4o", etc.
    temperature: float = 0.3
    max_tokens: int = 4096
    max_tokens_multiplier: int = 2  # For agent LLM (max_tokens * multiplier)
    
    # Memory Configuration
    memory_token_limit: int = 8000
    
    # Client Configuration
    reuse_client: bool = True
    verbose: bool = True


@dataclass
class RAGConfig:
    """Configuration for RAG tools and queries"""
    # Query Configuration
    similarity_top_k: int = 5
    temperature: float = 0.5
    max_tokens: int = 1000
    
    # Index Configuration
    default_indices: List[str] = None
    all_indices: List[str] = None
    
    def __post_init__(self):
        if self.default_indices is None:
            self.default_indices = ["alloradocs"]
        if self.all_indices is None:
            self.all_indices = ["alloradocs", "allora_production"]


@dataclass
class ChartConfig:
    figure_size: tuple = (10, 6)
    dpi: int = 300


@dataclass
class SlackConfig:
    max_sources_displayed: int = 3
    bot_token: str = ""  # Will be populated from environment


@dataclass
class WizardConfig:
    """Configuration for Wizard API tools"""
    api_url: str = "http://localhost:3000"
    timeout: float = 60.0
    api_key: Optional[str] = None
    # Anthropic config for wizard tool reasoning
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-opus-4-5-20251101"


@dataclass
class ServerConfig:
    """Configuration for FastAPI server"""
    # Server Settings
    debug: bool = False
    port: int = 8000
    host: str = "0.0.0.0"
    
    # CORS Settings
    allowed_origins: List[str] = None
    allowed_methods: List[str] = None
    allowed_headers: List[str] = None
    allow_credentials: bool = True
    
    # Rate Limiting
    requests_per_minute: int = 60
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:3000", "http://localhost:8000"]
        if self.allowed_methods is None:
            self.allowed_methods = ["GET", "POST"]
        if self.allowed_headers is None:
            self.allowed_headers = ["*"]


@dataclass
class Config:
    """Main configuration object containing all subsystem configurations"""
    agent: AgentConfig
    rag: RAGConfig
    chart: ChartConfig
    slack: SlackConfig
    server: ServerConfig
    wizard: WizardConfig

    # Environment Variables (required)
    llama_cloud_api_key: str
    llama_cloud_org_id: str
    openai_api_key: str
    slack_bot_token: str
    base_url: str
    image_dir: str
    
    @classmethod
    def from_environment(cls) -> 'Config':
        """Create configuration from environment variables"""
        
        # Required environment variables
        required_vars = {
            'llama_cloud_api_key': 'LLAMA_CLOUD_API_KEY',
            'llama_cloud_org_id': 'LLAMA_CLOUD_ORG_ID',
            'openai_api_key': 'OPENAI_API_KEY',
            'slack_bot_token': 'SLACK_BOT_TOKEN',
            'base_url': 'BASE_URL',
            'image_dir': 'IMAGE_DIR'
        }
        
        # Check for missing required variables
        missing_vars = []
        env_values = {}
        
        for config_key, env_var in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                missing_vars.append(env_var)
            else:
                env_values[config_key] = value
        
        if missing_vars:
            raise ConfigurationError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Create subsystem configurations with environment overrides
        agent_config = AgentConfig(
            model=os.getenv('AGENT_MODEL', 'gpt-5.2'),
            temperature=float(os.getenv('AGENT_TEMPERATURE', '0.3')),
            max_tokens=int(os.getenv('AGENT_MAX_TOKENS', '4096')),
            max_tokens_multiplier=int(os.getenv('AGENT_MAX_TOKENS_MULTIPLIER', '2')),
            memory_token_limit=int(os.getenv('AGENT_MEMORY_TOKEN_LIMIT', '8000')),
            reuse_client=os.getenv('AGENT_REUSE_CLIENT', 'true').lower() == 'true',
            verbose=os.getenv('AGENT_VERBOSE', 'true').lower() == 'true'
        )
        
        rag_config = RAGConfig(
            similarity_top_k=int(os.getenv('RAG_SIMILARITY_TOP_K', '5')),
            temperature=float(os.getenv('RAG_TEMPERATURE', '0.5')),
            max_tokens=int(os.getenv('RAG_MAX_TOKENS', '1000'))
        )
        
        chart_config = ChartConfig(
            figure_size=eval(os.getenv('CHART_FIGURE_SIZE', '(10, 6)')),
            dpi=int(os.getenv('CHART_DPI', '300'))
        )
        
        slack_config = SlackConfig(
            max_sources_displayed=int(os.getenv('SLACK_MAX_SOURCES_DISPLAYED', '3')),
            bot_token=env_values['slack_bot_token'],
        )
        
        server_config = ServerConfig(
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            port=int(os.getenv('PORT', '8000')),
            host=os.getenv('HOST', '0.0.0.0'),
            allowed_origins=os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:8000').split(','),
            requests_per_minute=int(os.getenv('REQUESTS_PER_MINUTE', '60'))
        )

        wizard_config = WizardConfig(
            api_url=os.getenv('WIZARD_API_URL', 'http://localhost:3000'),
            timeout=float(os.getenv('WIZARD_TIMEOUT', '60.0')),
            api_key=os.getenv('WIZARD_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            anthropic_model=os.getenv('WIZARD_ANTHROPIC_MODEL', 'claude-opus-4-5-20251101')
        )

        return cls(
            agent=agent_config,
            rag=rag_config,
            chart=chart_config,
            slack=slack_config,
            server=server_config,
            wizard=wizard_config,
            **env_values
        )
    
# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config.from_environment()
    return _config

