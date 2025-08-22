"""
Core AI agent for natural language MongoDB querying.

This module provides the main AI agent that orchestrates query translation,
MongoDB query execution, result formatting, and conversation memory management
for natural language database interactions.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from ..database.mongo_client import MongoConnectionManager
from ..llm.ollama_client import OllamaClient
from .query_translator import QueryTranslator


logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history and context for the AI agent.
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of conversation turns to remember
        """
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
    
    def add_interaction(
        self,
        user_query: str,
        generated_query: Optional[Dict[str, Any]],
        results: Optional[Union[List[Dict], Dict, int]],
        execution_time: float,
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """
        Add an interaction to conversation history.
        
        Args:
            user_query: Original natural language query
            generated_query: Generated MongoDB query
            results: Query execution results
            execution_time: Query execution time in seconds
            success: Whether the query was successful
            error_message: Error message if query failed
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'generated_query': generated_query,
            'results': results,
            'execution_time': execution_time,
            'success': success,
            'error_message': error_message,
            'result_count': self._get_result_count(results) if success else 0
        }
        
        self.history.append(interaction)
        
        # Maintain history size limit
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_recent_context(self, num_interactions: int = 5) -> str:
        """
        Get recent conversation context as a formatted string.
        
        Args:
            num_interactions: Number of recent interactions to include
            
        Returns:
            str: Formatted context string
        """
        if not self.history:
            return ""
        
        recent_history = self.history[-num_interactions:]
        context_parts = ["Recent conversation context:"]
        
        for i, interaction in enumerate(recent_history, 1):
            context_parts.append(f"\n{i}. User asked: {interaction['user_query']}")
            if interaction['success']:
                context_parts.append(f"   Result: Found {interaction['result_count']} items")
            else:
                context_parts.append(f"   Error: {interaction['error_message']}")
        
        return "\n".join(context_parts)
    
    def get_successful_queries(self) -> List[Dict[str, Any]]:
        """Get list of successful queries for learning."""
        return [
            {
                'user_query': interaction['user_query'],
                'generated_query': interaction['generated_query']
            }
            for interaction in self.history
            if interaction['success'] and interaction['generated_query']
        ]
    
    def _get_result_count(self, results: Union[List[Dict], Dict, int, None]) -> int:
        """Get count of results."""
        if results is None:
            return 0
        elif isinstance(results, int):
            return results
        elif isinstance(results, list):
            return len(results)
        elif isinstance(results, dict):
            return 1
        else:
            return 0
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        self.context.clear()


class MongoAgent:
    """
    Core AI agent for natural language MongoDB querying.
    
    This agent orchestrates the entire process of converting natural language
    queries into MongoDB operations, executing them, and formatting results
    with conversation memory management.
    """
    
    def __init__(
        self,
        mongo_client: MongoConnectionManager,
        ollama_client: OllamaClient,
        default_database: Optional[str] = None,
        max_conversation_history: int = 50,
        enable_query_caching: bool = True,
        result_limit: int = 100
    ):
        """
        Initialize MongoDB AI agent.
        
        Args:
            mongo_client: MongoDB connection manager
            ollama_client: Ollama client for LLM operations
            default_database: Default database name
            max_conversation_history: Maximum conversation history to maintain
            enable_query_caching: Whether to enable query result caching
            result_limit: Maximum number of results to return
        """
        self.mongo_client = mongo_client
        self.ollama_client = ollama_client
        self.default_database = default_database
        self.result_limit = result_limit
        
        # Initialize query translator
        self.query_translator = QueryTranslator(
            ollama_client=ollama_client,
            max_result_limit=result_limit,
            default_limit=min(50, result_limit)
        )
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(max_conversation_history)
        
        # Query caching
        self.enable_query_caching = enable_query_caching
        self.query_cache: Dict[str, Any] = {}
        
        # Agent state
        self.current_database = default_database
        self.current_collection = None
        self.database_schema_cache: Dict[str, Dict[str, Any]] = {}
    
    def initialize(self) -> bool:
        """
        Initialize the AI agent and all its components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize MongoDB connection
            if not self.mongo_client.is_connected():
                if not self.mongo_client.connect():
                    logger.error("Failed to connect to MongoDB")
                    return False
            
            # Initialize Ollama client
            if not self.ollama_client.is_available():
                if not self.ollama_client.initialize():
                    logger.error("Failed to initialize Ollama client")
                    return False
            
            logger.info("MongoDB AI Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MongoDB AI Agent: {e}")
            return False
    
    def set_database(self, database_name: str) -> bool:
        """
        Set the current working database.
        
        Args:
            database_name: Database name to use
            
        Returns:
            bool: True if database exists and was set, False otherwise
        """
        try:
            databases = self.mongo_client.list_databases()
            if database_name not in databases:
                logger.warning(f"Database '{database_name}' not found. Available: {databases}")
                return False
            
            self.current_database = database_name
            self.current_collection = None  # Reset collection when changing database
            logger.info(f"Set current database to: {database_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting database: {e}")
            return False
    
    def set_collection(self, collection_name: str) -> bool:
        """
        Set the current working collection.
        
        Args:
            collection_name: Collection name to use
            
        Returns:
            bool: True if collection exists and was set, False otherwise
        """
        if not self.current_database:
            logger.error("No database selected")
            return False
        
        try:
            collections = self.mongo_client.list_collections(self.current_database)
            if collection_name not in collections:
                logger.warning(f"Collection '{collection_name}' not found in database '{self.current_database}'. Available: {collections}")
                return False
            
            self.current_collection = collection_name
            logger.info(f"Set current collection to: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting collection: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about available databases and collections.
        
        Returns:
            Dict: Database and collection information
        """
        try:
            databases = self.mongo_client.list_databases()
            db_info = {
                'databases': databases,
                'current_database': self.current_database,
                'current_collection': self.current_collection,
                'collections': {}
            }
            
            # Get collections for each database
            for db_name in databases:
                try:
                    collections = self.mongo_client.list_collections(db_name)
                    db_info['collections'][db_name] = collections
                except Exception as e:
                    logger.warning(f"Could not list collections for database {db_name}: {e}")
                    db_info['collections'][db_name] = []
            
            return db_info
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {'databases': [], 'collections': {}, 'current_database': None, 'current_collection': None}
    
    def _get_collection_schema(self, collection_name: str, database_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get schema information for a collection by sampling documents.
        
        Args:
            collection_name: Collection name
            database_name: Database name (uses current if not provided)
            
        Returns:
            Optional[Dict]: Collection schema information
        """
        db_name = database_name or self.current_database
        if not db_name:
            return None
        
        cache_key = f"{db_name}.{collection_name}"
        if cache_key in self.database_schema_cache:
            return self.database_schema_cache[cache_key]
        
        try:
            # Sample a few documents to infer schema
            sample_query = {
                'operation': 'find',
                'filter': {},
                'limit': 5
            }
            
            results = self.mongo_client.execute_query(
                collection_name=collection_name,
                operation=sample_query['operation'],
                query=sample_query,
                database_name=db_name
            )
            
            if not results:
                return None
            
            # Analyze document structure
            schema = {
                'fields': set(),
                'sample_documents': results[:2],  # Keep only 2 for schema
                'document_count': len(results)
            }
            
            for doc in results:
                if isinstance(doc, dict):
                    schema['fields'].update(doc.keys())
            
            schema['fields'] = list(schema['fields'])  # Convert set to list
            
            # Cache the schema
            self.database_schema_cache[cache_key] = schema
            return schema
            
        except Exception as e:
            logger.error(f"Error getting collection schema: {e}")
            return None
    
    def process_query(
        self,
        user_query: str,
        collection_name: Optional[str] = None,
        database_name: Optional[str] = None,
        include_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Process a natural language query and return results.
        
        Args:
            user_query: Natural language query from user
            collection_name: Target collection (auto-detected if not provided)
            database_name: Target database (uses current if not provided)
            include_explanation: Whether to include query explanation
            
        Returns:
            Dict: Query results and metadata
        """
        start_time = datetime.now()
        
        try:
            # Use current database if not specified
            target_database = database_name or self.current_database
            if not target_database:
                return self._create_error_response(
                    user_query, "No database selected", start_time
                )
            
            # Use current collection if not specified
            target_collection = collection_name or self.current_collection
            
            # Get collection schema for better query generation
            collection_info = None
            if target_collection:
                collection_info = self._get_collection_schema(target_collection, target_database)
            
            # Get conversation context
            context = self.conversation_memory.get_recent_context(3)
            
            # Translate natural language to MongoDB query
            mongodb_query = self.query_translator.translate_query(
                user_query=user_query,
                collection_name=target_collection,
                database_schema=self.database_schema_cache.get(f"{target_database}.{target_collection}"),
                collection_info=collection_info,
                additional_context=context if context else None
            )
            
            if not mongodb_query:
                return self._create_error_response(
                    user_query, "Failed to translate query", start_time
                )
            
            # Update metadata with timestamp
            mongodb_query['_metadata']['generated_at'] = datetime.now().isoformat()
            
            # Determine target collection from query if not specified
            if not target_collection:
                target_collection = mongodb_query['_metadata'].get('collection_hint')
                if not target_collection:
                    return self._create_error_response(
                        user_query, "Could not determine target collection", start_time
                    )
            
            # Execute MongoDB query
            results = self.mongo_client.execute_query(
                collection_name=target_collection,
                operation=mongodb_query['operation'],
                query=mongodb_query,
                database_name=target_database
            )
            
            if results is None:
                return self._create_error_response(
                    user_query, "Query execution failed", start_time, mongodb_query
                )
            
            # Format results
            formatted_results = self._format_results(results, mongodb_query['operation'])
            
            # Generate explanation if requested
            explanation = None
            if include_explanation:
                explanation = self.ollama_client.explain_query(mongodb_query)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            response = {
                'success': True,
                'user_query': user_query,
                'generated_query': mongodb_query,
                'results': formatted_results,
                'result_count': self._get_result_count(results),
                'execution_time': execution_time,
                'database': target_database,
                'collection': target_collection,
                'explanation': explanation,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to conversation memory
            self.conversation_memory.add_interaction(
                user_query=user_query,
                generated_query=mongodb_query,
                results=results,
                execution_time=execution_time,
                success=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(
                user_query, f"Unexpected error: {str(e)}", start_time
            )
    
    def _create_error_response(
        self,
        user_query: str,
        error_message: str,
        start_time: datetime,
        generated_query: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Add to conversation memory
        self.conversation_memory.add_interaction(
            user_query=user_query,
            generated_query=generated_query,
            results=None,
            execution_time=execution_time,
            success=False,
            error_message=error_message
        )
        
        return {
            'success': False,
            'user_query': user_query,
            'generated_query': generated_query,
            'results': None,
            'result_count': 0,
            'execution_time': execution_time,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_results(self, results: Union[List[Dict], Dict, int], operation: str) -> Dict[str, Any]:
        """
        Format query results for display.
        
        Args:
            results: Raw query results
            operation: MongoDB operation type
            
        Returns:
            Dict: Formatted results
        """
        if operation == 'count':
            return {
                'type': 'count',
                'count': results,
                'data': None
            }
        elif isinstance(results, list):
            return {
                'type': 'documents',
                'count': len(results),
                'data': results,
                'sample': results[:5] if len(results) > 5 else results  # Show first 5 for preview
            }
        elif isinstance(results, dict):
            return {
                'type': 'document',
                'count': 1,
                'data': [results],
                'sample': [results]
            }
        else:
            return {
                'type': 'unknown',
                'count': 0,
                'data': results,
                'sample': None
            }
    
    def _get_result_count(self, results: Union[List[Dict], Dict, int, None]) -> int:
        """Get count of results."""
        if results is None:
            return 0
        elif isinstance(results, int):
            return results
        elif isinstance(results, list):
            return len(results)
        elif isinstance(results, dict):
            return 1
        else:
            return 0
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List: Recent conversation history
        """
        return self.conversation_memory.history[-limit:] if self.conversation_memory.history else []
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_memory.clear_history()
        logger.info("Conversation history cleared")
    
    def get_query_suggestions(self, partial_query: str = "") -> List[str]:
        """
        Get query suggestions based on conversation history and database schema.
        
        Args:
            partial_query: Partial query text for context
            
        Returns:
            List[str]: List of suggested queries
        """
        suggestions = []
        
        # Add suggestions based on current database/collection
        if self.current_database and self.current_collection:
            collection_info = self._get_collection_schema(self.current_collection, self.current_database)
            if collection_info and 'fields' in collection_info:
                fields = collection_info['fields'][:5]  # Limit to first 5 fields
                suggestions.extend([
                    f"Find all documents in {self.current_collection}",
                    f"Count documents in {self.current_collection}",
                    f"Show me documents with {fields[0]} greater than 10" if fields else "",
                    f"Find documents where {fields[0]} contains 'text'" if fields else ""
                ])
        
        # Add suggestions from successful past queries
        successful_queries = self.conversation_memory.get_successful_queries()
        if successful_queries:
            recent_queries = successful_queries[-3:]  # Last 3 successful queries
            for query_info in recent_queries:
                if query_info['user_query'] not in suggestions:
                    suggestions.append(f"Similar to: {query_info['user_query']}")
        
        # Filter out empty suggestions
        suggestions = [s for s in suggestions if s.strip()]
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current agent status and configuration.
        
        Returns:
            Dict: Agent status information
        """
        return {
            'mongodb_connected': self.mongo_client.is_connected(),
            'ollama_available': self.ollama_client.is_available(),
            'current_database': self.current_database,
            'current_collection': self.current_collection,
            'conversation_history_count': len(self.conversation_memory.history),
            'available_models': self.ollama_client.get_available_models(),
            'current_model': self.ollama_client.model,
            'result_limit': self.result_limit,
            'query_caching_enabled': self.enable_query_caching
        }
