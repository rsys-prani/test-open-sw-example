"""
Ollama client wrapper for natural language to MongoDB query generation.

This module provides a high-level interface for interacting with Ollama models
to generate MongoDB queries from natural language input with proper prompt
formatting, response parsing, and error handling.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

import ollama
from ollama import Client


logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Ollama client wrapper for MongoDB query generation.
    
    This class provides methods for initializing Ollama models, formatting
    prompts for MongoDB query generation, and parsing responses with
    comprehensive error handling.
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: int = 30,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama server host URL
            model: Model name to use for generation
            timeout: Request timeout in seconds
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
        """
        self.host = host
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        self._client: Optional[Client] = None
        self._available_models: List[str] = []
        
    def initialize(self) -> bool:
        """
        Initialize the Ollama client and check model availability.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self._client = Client(host=self.host)
            
            # Test connection and get available models
            models_response = self._client.list()
            self._available_models = [model['name'] for model in models_response['models']]
            
            if self.model not in self._available_models:
                logger.warning(f"Model '{self.model}' not found. Available models: {self._available_models}")
                if self._available_models:
                    self.model = self._available_models[0]
                    logger.info(f"Using available model: {self.model}")
                else:
                    logger.error("No models available on Ollama server")
                    return False
            
            logger.info(f"Ollama client initialized successfully with model: {self.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is available.
        
        Returns:
            bool: True if server is available, False otherwise
        """
        if not self._client:
            return False
            
        try:
            self._client.list()
            return True
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List[str]: List of available model names
        """
        return self._available_models.copy()
    
    def _create_mongodb_query_prompt(
        self,
        user_query: str,
        database_schema: Optional[Dict[str, Any]] = None,
        collection_info: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Create a formatted prompt for MongoDB query generation.
        
        Args:
            user_query: Natural language query from user
            database_schema: Database schema information
            collection_info: Collection structure information
            examples: Example query pairs for few-shot learning
            
        Returns:
            str: Formatted prompt for the LLM
        """
        prompt_parts = [
            "You are an expert MongoDB query generator. Your task is to convert natural language queries into valid MongoDB queries.",
            "",
            "IMPORTANT RULES:",
            "1. Generate ONLY valid MongoDB query syntax",
            "2. Use proper MongoDB operators ($eq, $gt, $lt, $in, $regex, etc.)",
            "3. Return queries in JSON format",
            "4. For find operations, return: {'operation': 'find', 'filter': {...}, 'projection': {...}, 'limit': N, 'sort': {...}}",
            "5. For aggregate operations, return: {'operation': 'aggregate', 'pipeline': [...]}",
            "6. For count operations, return: {'operation': 'count', 'filter': {...}}",
            "7. Always include the 'operation' field to specify the query type",
            "8. Use case-insensitive regex for text searches: {'$regex': 'pattern', '$options': 'i'}",
            "9. Be conservative with limits - use reasonable defaults (e.g., 10-100 documents)",
            ""
        ]
        
        # Add schema information if available
        if database_schema:
            prompt_parts.extend([
                "DATABASE SCHEMA:",
                json.dumps(database_schema, indent=2),
                ""
            ])
        
        # Add collection information if available
        if collection_info:
            prompt_parts.extend([
                "COLLECTION INFORMATION:",
                json.dumps(collection_info, indent=2),
                ""
            ])
        
        # Add examples if provided
        if examples:
            prompt_parts.append("EXAMPLES:")
            for i, example in enumerate(examples, 1):
                prompt_parts.extend([
                    f"Example {i}:",
                    f"Natural Language: {example.get('natural', '')}",
                    f"MongoDB Query: {example.get('query', '')}",
                    ""
                ])
        
        # Add the actual user query
        prompt_parts.extend([
            "USER QUERY:",
            f"Convert this natural language query to MongoDB: {user_query}",
            "",
            "MongoDB Query (JSON format only):"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_mongodb_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to extract MongoDB query.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Optional[Dict]: Parsed MongoDB query or None if parsing failed
        """
        try:
            # Clean the response - remove markdown code blocks and extra text
            cleaned_response = response.strip()
            
            # Remove markdown code blocks
            if "```json" in cleaned_response:
                cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
            elif "```" in cleaned_response:
                cleaned_response = re.sub(r'```\s*', '', cleaned_response)
                cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = cleaned_response
            
            # Parse JSON
            query = json.loads(json_str)
            
            # Validate required fields
            if not isinstance(query, dict):
                logger.error("Response is not a valid dictionary")
                return None
                
            if 'operation' not in query:
                logger.error("Missing 'operation' field in query")
                return None
                
            # Validate operation type
            valid_operations = ['find', 'find_one', 'count', 'aggregate']
            if query['operation'] not in valid_operations:
                logger.error(f"Invalid operation: {query['operation']}")
                return None
            
            logger.info(f"Successfully parsed MongoDB query: {query}")
            return query
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None
    
    def generate_mongodb_query(
        self,
        user_query: str,
        database_schema: Optional[Dict[str, Any]] = None,
        collection_info: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Generate MongoDB query from natural language input.
        
        Args:
            user_query: Natural language query from user
            database_schema: Database schema information
            collection_info: Collection structure information
            examples: Example query pairs for few-shot learning
            max_retries: Maximum number of retry attempts
            
        Returns:
            Optional[Dict]: Generated MongoDB query or None if failed
        """
        if not self._client:
            logger.error("Ollama client not initialized")
            return None
        
        prompt = self._create_mongodb_query_prompt(
            user_query, database_schema, collection_info, examples
        )
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Generating MongoDB query (attempt {attempt + 1})")
                
                response = self._client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'top_p': self.top_p,
                        'num_predict': self.max_tokens,
                    }
                )
                
                if 'response' not in response:
                    logger.error("No response field in Ollama response")
                    continue
                
                parsed_query = self._parse_mongodb_response(response['response'])
                if parsed_query:
                    return parsed_query
                
                logger.warning(f"Failed to parse response on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Error generating query on attempt {attempt + 1}: {e}")
                
        logger.error(f"Failed to generate valid MongoDB query after {max_retries + 1} attempts")
        return None
    
    def explain_query(self, query: Dict[str, Any]) -> Optional[str]:
        """
        Generate human-readable explanation of a MongoDB query.
        
        Args:
            query: MongoDB query dictionary
            
        Returns:
            Optional[str]: Human-readable explanation or None if failed
        """
        if not self._client:
            logger.error("Ollama client not initialized")
            return None
        
        prompt = f"""
        Explain this MongoDB query in simple, human-readable terms:
        
        Query: {json.dumps(query, indent=2)}
        
        Provide a clear, concise explanation of what this query does, including:
        1. What operation it performs
        2. What data it searches for or filters
        3. Any sorting, limiting, or projection applied
        4. The expected result format
        
        Explanation:
        """
        
        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_predict': 512,
                }
            )
            
            if 'response' in response:
                return response['response'].strip()
            else:
                logger.error("No response field in Ollama response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating query explanation: {e}")
            return None
    
    def validate_query_safety(self, query: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate MongoDB query for safety and potential issues.
        
        Args:
            query: MongoDB query dictionary
            
        Returns:
            tuple: (is_safe, list_of_warnings)
        """
        warnings = []
        is_safe = True
        
        try:
            # Check for potentially dangerous operations
            if query.get('operation') == 'aggregate':
                pipeline = query.get('pipeline', [])
                for stage in pipeline:
                    if isinstance(stage, dict):
                        # Check for $out or $merge stages that modify data
                        if '$out' in stage or '$merge' in stage:
                            warnings.append("Query contains data modification operations")
                            is_safe = False
            
            # Check for overly broad queries without limits
            if query.get('operation') in ['find', 'aggregate']:
                if 'limit' not in query and query.get('operation') == 'find':
                    warnings.append("Query has no limit - may return large result sets")
                
                # Check for empty filters
                if query.get('filter') == {} or not query.get('filter'):
                    warnings.append("Query has no filter - will return all documents")
            
            # Check for regex without proper escaping
            def check_regex_in_dict(d):
                if isinstance(d, dict):
                    for key, value in d.items():
                        if key == '$regex' and isinstance(value, str):
                            # Check for potentially dangerous regex patterns
                            if '.*' in value or '.+' in value:
                                warnings.append("Regex pattern may be inefficient")
                        elif isinstance(value, (dict, list)):
                            check_regex_in_dict(value)
                elif isinstance(d, list):
                    for item in d:
                        check_regex_in_dict(item)
            
            check_regex_in_dict(query)
            
        except Exception as e:
            logger.error(f"Error validating query safety: {e}")
            warnings.append("Could not validate query safety")
            is_safe = False
        
        return is_safe, warnings
