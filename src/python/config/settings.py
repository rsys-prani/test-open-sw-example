"""
Configuration management for AI MongoDB Query Agent.

This module provides centralized configuration management using Pydantic
for MongoDB connection strings, Ollama model settings, and Streamlit app
configuration with environment variable support.
"""

import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings, Field, validator
from pydantic.env_settings import SettingsSourceCallable


class MongoDBSettings(BaseSettings):
    """MongoDB connection configuration."""
    
    host: str = Field(default="localhost", description="MongoDB host address")
    port: int = Field(default=27017, ge=1, le=65535, description="MongoDB port number")
    username: Optional[str] = Field(default=None, description="MongoDB username")
    password: Optional[str] = Field(default=None, description="MongoDB password")
    database: Optional[str] = Field(default=None, description="Default database name")
    auth_source: str = Field(default="admin", description="Authentication database")
    
    # Connection pool settings
    max_pool_size: int = Field(default=100, ge=1, description="Maximum connection pool size")
    min_pool_size: int = Field(default=0, ge=0, description="Minimum connection pool size")
    max_idle_time_ms: int = Field(default=30000, ge=1000, description="Maximum idle time in milliseconds")
    server_selection_timeout_ms: int = Field(default=5000, ge=1000, description="Server selection timeout in milliseconds")
    connect_timeout_ms: int = Field(default=10000, ge=1000, description="Connection timeout in milliseconds")
    socket_timeout_ms: int = Field(default=20000, ge=1000, description="Socket timeout in milliseconds")
    
    # SSL/TLS settings
    use_ssl: bool = Field(default=False, description="Use SSL/TLS connection")
    ssl_cert_reqs: str = Field(default="CERT_REQUIRED", description="SSL certificate requirements")
    ssl_ca_certs: Optional[str] = Field(default=None, description="Path to CA certificates file")
    
    class Config:
        env_prefix = "MONGODB_"
        case_sensitive = False
        
    @validator('port')
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('ssl_cert_reqs')
    def validate_ssl_cert_reqs(cls, v):
        valid_options = ['CERT_NONE', 'CERT_OPTIONAL', 'CERT_REQUIRED']
        if v not in valid_options:
            raise ValueError(f'ssl_cert_reqs must be one of: {valid_options}')
        return v
    
    def get_connection_string(self) -> str:
        """Generate MongoDB connection string."""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
        else:
            auth_part = ""
        
        connection_string = f"mongodb://{auth_part}{self.host}:{self.port}/"
        
        # Add query parameters
        params = []
        if self.username and self.password:
            params.append(f"authSource={self.auth_source}")
        if self.use_ssl:
            params.append("ssl=true")
            if self.ssl_ca_certs:
                params.append(f"ssl_ca_certs={self.ssl_ca_certs}")
            params.append(f"ssl_cert_reqs={self.ssl_cert_reqs}")
        
        if params:
            connection_string += "?" + "&".join(params)
        
        return connection_string
    
    def get_connection_options(self) -> Dict[str, Any]:
        """Get connection options for pymongo client."""
        return {
            "maxPoolSize": self.max_pool_size,
            "minPoolSize": self.min_pool_size,
            "maxIdleTimeMS": self.max_idle_time_ms,
            "serverSelectionTimeoutMS": self.server_selection_timeout_ms,
            "connectTimeoutMS": self.connect_timeout_ms,
            "socketTimeoutMS": self.socket_timeout_ms,
        }


class OllamaSettings(BaseSettings):
    """Ollama client configuration."""
    
    host: str = Field(default="http://localhost:11434", description="Ollama server host URL")
    model: str = Field(default="llama2", description="Default model name")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    
    # Generation parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=40, ge=1, description="Top-k sampling parameter")
    max_tokens: int = Field(default=2048, ge=1, description="Maximum tokens to generate")
    
    # Model management
    auto_pull_models: bool = Field(default=False, description="Automatically pull missing models")
    preferred_models: List[str] = Field(default=["llama2", "codellama", "mistral"], description="Preferred model list in order")
    
    class Config:
        env_prefix = "OLLAMA_"
        case_sensitive = False
    
    @validator('host')
    def validate_host(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Host must start with http:// or https://')
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not (0.0 <= v <= 2.0):
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Top-p must be between 0.0 and 1.0')
        return v


class QueryTranslatorSettings(BaseSettings):
    """Query translator configuration."""
    
    max_result_limit: int = Field(default=1000, ge=1, description="Maximum allowed result limit")
    default_limit: int = Field(default=50, ge=1, description="Default limit for queries")
    enable_aggregation: bool = Field(default=True, description="Enable aggregation queries")
    enable_text_search: bool = Field(default=True, description="Enable text search queries")
    
    # Safety settings
    enable_query_validation: bool = Field(default=True, description="Enable query validation")
    enable_query_sanitization: bool = Field(default=True, description="Enable query sanitization")
    max_query_complexity: int = Field(default=10, ge=1, description="Maximum query complexity score")
    
    # Performance settings
    query_cache_size: int = Field(default=100, ge=0, description="Query cache size (0 to disable)")
    query_cache_ttl: int = Field(default=3600, ge=60, description="Query cache TTL in seconds")
    
    class Config:
        env_prefix = "QUERY_TRANSLATOR_"
        case_sensitive = False
    
    @validator('default_limit')
    def validate_default_limit(cls, v, values):
        max_limit = values.get('max_result_limit', 1000)
        if v > max_limit:
            raise ValueError(f'Default limit cannot exceed max_result_limit ({max_limit})')
        return v


class StreamlitSettings(BaseSettings):
    """Streamlit application configuration."""
    
    # App settings
    page_title: str = Field(default="AI MongoDB Query Agent", description="Application page title")
    page_icon: str = Field(default="🤖", description="Application page icon")
    layout: str = Field(default="wide", description="Streamlit layout mode")
    initial_sidebar_state: str = Field(default="expanded", description="Initial sidebar state")
    
    # UI settings
    theme_primary_color: str = Field(default="#1f77b4", description="Primary theme color")
    theme_background_color: str = Field(default="#ffffff", description="Background color")
    theme_secondary_background_color: str = Field(default="#f0f2f6", description="Secondary background color")
    theme_text_color: str = Field(default="#262730", description="Text color")
    
    # Chat settings
    max_chat_history: int = Field(default=50, ge=1, description="Maximum chat history to display")
    enable_query_suggestions: bool = Field(default=True, description="Enable query suggestions")
    max_query_suggestions: int = Field(default=5, ge=1, description="Maximum number of query suggestions")
    
    # Result display settings
    max_result_preview: int = Field(default=5, ge=1, description="Maximum results to show in preview")
    enable_result_download: bool = Field(default=True, description="Enable result download")
    result_download_format: str = Field(default="csv", description="Default download format")
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable Streamlit caching")
    cache_ttl: int = Field(default=300, ge=60, description="Cache TTL in seconds")
    
    class Config:
        env_prefix = "STREAMLIT_"
        case_sensitive = False
    
    @validator('layout')
    def validate_layout(cls, v):
        valid_layouts = ['centered', 'wide']
        if v not in valid_layouts:
            raise ValueError(f'Layout must be one of: {valid_layouts}')
        return v
    
    @validator('initial_sidebar_state')
    def validate_sidebar_state(cls, v):
        valid_states = ['auto', 'expanded', 'collapsed']
        if v not in valid_states:
            raise ValueError(f'Initial sidebar state must be one of: {valid_states}')
        return v
    
    @validator('result_download_format')
    def validate_download_format(cls, v):
        valid_formats = ['csv', 'json', 'xlsx']
        if v not in valid_formats:
            raise ValueError(f'Download format must be one of: {valid_formats}')
        return v


class AgentSettings(BaseSettings):
    """AI Agent configuration."""
    
    # Agent behavior
    max_conversation_history: int = Field(default=50, ge=1, description="Maximum conversation history")
    enable_conversation_memory: bool = Field(default=True, description="Enable conversation memory")
    conversation_context_window: int = Field(default=5, ge=1, description="Number of recent interactions for context")
    
    # Query processing
    max_retry_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts for failed queries")
    query_timeout: int = Field(default=30, ge=5, description="Query timeout in seconds")
    enable_query_explanation: bool = Field(default=True, description="Enable query explanations")
    
    # Schema management
    enable_schema_caching: bool = Field(default=True, description="Enable database schema caching")
    schema_cache_ttl: int = Field(default=3600, ge=300, description="Schema cache TTL in seconds")
    schema_sample_size: int = Field(default=5, ge=1, description="Number of documents to sample for schema inference")
    
    # Safety and security
    enable_query_logging: bool = Field(default=True, description="Enable query logging")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    
    class Config:
        env_prefix = "AGENT_"
        case_sensitive = False
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Component settings
    mongodb: MongoDBSettings = Field(default_factory=MongoDBSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    query_translator: QueryTranslatorSettings = Field(default_factory=QueryTranslatorSettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    
    # Application metadata
    app_name: str = Field(default="AI MongoDB Query Agent", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    app_description: str = Field(default="Natural language interface for MongoDB querying", description="Application description")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('environment')
    def validate_environment(cls, v):
        valid_environments = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_environments:
            raise ValueError(f'Environment must be one of: {valid_environments}')
        return v.lower()
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def get_log_level(self) -> str:
        """Get appropriate log level based on environment."""
        if self.debug or self.is_development():
            return "DEBUG"
        return self.agent.log_level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict()
    
    def to_json(self) -> str:
        """Convert settings to JSON string."""
        return self.json(indent=2)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance (singleton pattern).
    
    Returns:
        Settings: Global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables and config files.
    
    Returns:
        Settings: Reloaded settings instance
    """
    global _settings
    _settings = Settings()
    return _settings


def create_env_template(file_path: str = ".env.template") -> None:
    """
    Create environment variable template file.
    
    Args:
        file_path: Path to create the template file
    """
    template_content = """# AI MongoDB Query Agent Configuration Template
# Copy this file to .env and modify the values as needed

# Environment
ENVIRONMENT=development
DEBUG=false

# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USERNAME=
MONGODB_PASSWORD=
MONGODB_DATABASE=
MONGODB_AUTH_SOURCE=admin
MONGODB_MAX_POOL_SIZE=100
MONGODB_MIN_POOL_SIZE=0
MONGODB_MAX_IDLE_TIME_MS=30000
MONGODB_SERVER_SELECTION_TIMEOUT_MS=5000
MONGODB_CONNECT_TIMEOUT_MS=10000
MONGODB_SOCKET_TIMEOUT_MS=20000
MONGODB_USE_SSL=false
MONGODB_SSL_CERT_REQS=CERT_REQUIRED
MONGODB_SSL_CA_CERTS=

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TIMEOUT=30
OLLAMA_TEMPERATURE=0.1
OLLAMA_TOP_P=0.9
OLLAMA_TOP_K=40
OLLAMA_MAX_TOKENS=2048
OLLAMA_AUTO_PULL_MODELS=false

# Query Translator Configuration
QUERY_TRANSLATOR_MAX_RESULT_LIMIT=1000
QUERY_TRANSLATOR_DEFAULT_LIMIT=50
QUERY_TRANSLATOR_ENABLE_AGGREGATION=true
QUERY_TRANSLATOR_ENABLE_TEXT_SEARCH=true
QUERY_TRANSLATOR_ENABLE_QUERY_VALIDATION=true
QUERY_TRANSLATOR_ENABLE_QUERY_SANITIZATION=true
QUERY_TRANSLATOR_MAX_QUERY_COMPLEXITY=10
QUERY_TRANSLATOR_QUERY_CACHE_SIZE=100
QUERY_TRANSLATOR_QUERY_CACHE_TTL=3600

# Streamlit Configuration
STREAMLIT_PAGE_TITLE=AI MongoDB Query Agent
STREAMLIT_PAGE_ICON=🤖
STREAMLIT_LAYOUT=wide
STREAMLIT_INITIAL_SIDEBAR_STATE=expanded
STREAMLIT_THEME_PRIMARY_COLOR=#1f77b4
STREAMLIT_THEME_BACKGROUND_COLOR=#ffffff
STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR=#f0f2f6
STREAMLIT_THEME_TEXT_COLOR=#262730
STREAMLIT_MAX_CHAT_HISTORY=50
STREAMLIT_ENABLE_QUERY_SUGGESTIONS=true
STREAMLIT_MAX_QUERY_SUGGESTIONS=5
STREAMLIT_MAX_RESULT_PREVIEW=5
STREAMLIT_ENABLE_RESULT_DOWNLOAD=true
STREAMLIT_RESULT_DOWNLOAD_FORMAT=csv
STREAMLIT_ENABLE_CACHING=true
STREAMLIT_CACHE_TTL=300

# Agent Configuration
AGENT_MAX_CONVERSATION_HISTORY=50
AGENT_ENABLE_CONVERSATION_MEMORY=true
AGENT_CONVERSATION_CONTEXT_WINDOW=5
AGENT_MAX_RETRY_ATTEMPTS=3
AGENT_QUERY_TIMEOUT=30
AGENT_ENABLE_QUERY_EXPLANATION=true
AGENT_ENABLE_SCHEMA_CACHING=true
AGENT_SCHEMA_CACHE_TTL=3600
AGENT_SCHEMA_SAMPLE_SIZE=5
AGENT_ENABLE_QUERY_LOGGING=true
AGENT_LOG_LEVEL=INFO
AGENT_ENABLE_PERFORMANCE_MONITORING=true

# Application Metadata
APP_NAME=AI MongoDB Query Agent
APP_VERSION=0.1.0
APP_DESCRIPTION=Natural language interface for MongoDB querying
"""
    
    with open(file_path, 'w') as f:
        f.write(template_content)
    
    print(f"Environment template created at: {file_path}")


if __name__ == "__main__":
    # Create environment template when run directly
    create_env_template()
    
    # Display current settings
    settings = get_settings()
    print("\nCurrent Settings:")
    print(settings.to_json())
