"""
Natural language to MongoDB query translator.

This module provides a high-level translator that converts natural language
queries into valid MongoDB operations using the Ollama client, with proper
validation, sanitization, and query optimization.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..llm.ollama_client import OllamaClient


logger = logging.getLogger(__name__)


class QueryTranslator:
    """
    Natural language to MongoDB query translator.
    
    This class provides methods for translating natural language queries
    into MongoDB operations with validation, sanitization, and optimization.
    """
    
    # Dangerous MongoDB operators that should be restricted
    RESTRICTED_OPERATORS = {
        '$where',  # JavaScript execution
        '$function',  # Server-side JavaScript
        '$accumulator',  # Custom accumulator functions
        '$expr',  # Expression evaluation (can be dangerous)
    }
    
    # Safe MongoDB operators for query validation
    SAFE_QUERY_OPERATORS = {
        # Comparison operators
        '$eq', '$ne', '$gt', '$gte', '$lt', '$lte', '$in', '$nin',
        # Logical operators
        '$and', '$or', '$not', '$nor',
        # Element operators
        '$exists', '$type',
        # Evaluation operators
        '$regex', '$text', '$mod',
        # Array operators
        '$all', '$elemMatch', '$size',
        # Projection operators
        '$slice', '$elemMatch', '$',
        # Update operators (for reference, not used in queries)
        '$set', '$unset', '$inc', '$mul', '$rename', '$min', '$max',
        '$currentDate', '$addToSet', '$pop', '$pull', '$push', '$pullAll'
    }
    
    # Safe aggregation pipeline stages
    SAFE_AGGREGATION_STAGES = {
        '$match', '$project', '$sort', '$limit', '$skip', '$unwind',
        '$group', '$lookup', '$addFields', '$replaceRoot', '$facet',
        '$bucket', '$bucketAuto', '$sortByCount', '$count', '$sample'
    }
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        max_result_limit: int = 1000,
        default_limit: int = 50,
        enable_aggregation: bool = True,
        enable_text_search: bool = True
    ):
        """
        Initialize query translator.
        
        Args:
            ollama_client: Initialized Ollama client instance
            max_result_limit: Maximum allowed result limit
            default_limit: Default limit for queries without explicit limit
            enable_aggregation: Whether to allow aggregation queries
            enable_text_search: Whether to allow text search queries
        """
        self.ollama_client = ollama_client
        self.max_result_limit = max_result_limit
        self.default_limit = default_limit
        self.enable_aggregation = enable_aggregation
        self.enable_text_search = enable_text_search
        
        # Query examples for few-shot learning
        self.query_examples = [
            {
                "natural": "Find all users with age greater than 25",
                "query": '{"operation": "find", "filter": {"age": {"$gt": 25}}, "limit": 50}'
            },
            {
                "natural": "Get users named John or Jane",
                "query": '{"operation": "find", "filter": {"name": {"$in": ["John", "Jane"]}}, "limit": 50}'
            },
            {
                "natural": "Count all active users",
                "query": '{"operation": "count", "filter": {"status": "active"}}'
            },
            {
                "natural": "Find users with email containing gmail, sorted by name",
                "query": '{"operation": "find", "filter": {"email": {"$regex": "gmail", "$options": "i"}}, "sort": {"name": 1}, "limit": 50}'
            },
            {
                "natural": "Get the first 10 products with price between 100 and 500",
                "query": '{"operation": "find", "filter": {"price": {"$gte": 100, "$lte": 500}}, "limit": 10}'
            }
        ]
    
    def _sanitize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize MongoDB query to remove dangerous operations.
        
        Args:
            query: Raw MongoDB query dictionary
            
        Returns:
            Dict: Sanitized query dictionary
        """
        sanitized_query = query.copy()
        
        def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively sanitize dictionary."""
            sanitized = {}
            for key, value in d.items():
                # Remove restricted operators
                if key in self.RESTRICTED_OPERATORS:
                    logger.warning(f"Removed restricted operator: {key}")
                    continue
                
                # Sanitize nested dictionaries
                if isinstance(value, dict):
                    sanitized[key] = sanitize_dict(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        sanitize_dict(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    sanitized[key] = value
            
            return sanitized
        
        # Sanitize filter
        if 'filter' in sanitized_query and isinstance(sanitized_query['filter'], dict):
            sanitized_query['filter'] = sanitize_dict(sanitized_query['filter'])
        
        # Sanitize aggregation pipeline
        if 'pipeline' in sanitized_query and isinstance(sanitized_query['pipeline'], list):
            sanitized_pipeline = []
            for stage in sanitized_query['pipeline']:
                if isinstance(stage, dict):
                    # Check if stage is allowed
                    stage_name = next(iter(stage.keys())) if stage else None
                    if stage_name and stage_name not in self.SAFE_AGGREGATION_STAGES:
                        logger.warning(f"Removed unsafe aggregation stage: {stage_name}")
                        continue
                    sanitized_pipeline.append(sanitize_dict(stage))
                else:
                    sanitized_pipeline.append(stage)
            sanitized_query['pipeline'] = sanitized_pipeline
        
        return sanitized_query
    
    def _validate_query_structure(self, query: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate MongoDB query structure and constraints.
        
        Args:
            query: MongoDB query dictionary
            
        Returns:
            Tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required operation field
        if 'operation' not in query:
            errors.append("Missing required 'operation' field")
            return False, errors
        
        operation = query['operation']
        valid_operations = ['find', 'find_one', 'count', 'aggregate']
        
        if operation not in valid_operations:
            errors.append(f"Invalid operation '{operation}'. Must be one of: {valid_operations}")
            return False, errors
        
        # Validate operation-specific requirements
        if operation in ['find', 'find_one', 'count']:
            # These operations should have a filter
            if 'filter' in query and not isinstance(query['filter'], dict):
                errors.append("Filter must be a dictionary")
        
        elif operation == 'aggregate':
            if not self.enable_aggregation:
                errors.append("Aggregation queries are disabled")
                return False, errors
            
            if 'pipeline' not in query:
                errors.append("Aggregation operation requires 'pipeline' field")
            elif not isinstance(query['pipeline'], list):
                errors.append("Pipeline must be a list")
        
        # Validate limit constraints
        if 'limit' in query:
            limit = query['limit']
            if not isinstance(limit, int) or limit <= 0:
                errors.append("Limit must be a positive integer")
            elif limit > self.max_result_limit:
                errors.append(f"Limit {limit} exceeds maximum allowed limit {self.max_result_limit}")
        
        # Validate sort structure
        if 'sort' in query:
            sort_spec = query['sort']
            if not isinstance(sort_spec, dict):
                errors.append("Sort specification must be a dictionary")
            else:
                for field, direction in sort_spec.items():
                    if direction not in [1, -1, "asc", "desc"]:
                        errors.append(f"Invalid sort direction for field '{field}': {direction}")
        
        # Validate projection structure
        if 'projection' in query:
            projection = query['projection']
            if not isinstance(projection, dict):
                errors.append("Projection must be a dictionary")
        
        return len(errors) == 0, errors
    
    def _optimize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize MongoDB query for better performance.
        
        Args:
            query: MongoDB query dictionary
            
        Returns:
            Dict: Optimized query dictionary
        """
        optimized_query = query.copy()
        
        # Add default limit if not specified for find operations
        if query['operation'] == 'find' and 'limit' not in query:
            optimized_query['limit'] = self.default_limit
            logger.info(f"Added default limit: {self.default_limit}")
        
        # Optimize regex queries for case-insensitive search
        def optimize_regex_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively optimize regex patterns."""
            optimized = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    if '$regex' in value and '$options' not in value:
                        # Add case-insensitive option if not specified
                        value = value.copy()
                        value['$options'] = 'i'
                        logger.info(f"Added case-insensitive option to regex for field: {key}")
                    optimized[key] = optimize_regex_in_dict(value)
                elif isinstance(value, list):
                    optimized[key] = [
                        optimize_regex_in_dict(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    optimized[key] = value
            return optimized
        
        # Optimize filter
        if 'filter' in optimized_query and isinstance(optimized_query['filter'], dict):
            optimized_query['filter'] = optimize_regex_in_dict(optimized_query['filter'])
        
        # Optimize aggregation pipeline
        if 'pipeline' in optimized_query and isinstance(optimized_query['pipeline'], list):
            optimized_pipeline = []
            for stage in optimized_query['pipeline']:
                if isinstance(stage, dict):
                    optimized_pipeline.append(optimize_regex_in_dict(stage))
                else:
                    optimized_pipeline.append(stage)
            optimized_query['pipeline'] = optimized_pipeline
        
        return optimized_query
    
    def _extract_collection_hint(self, user_query: str) -> Optional[str]:
        """
        Extract potential collection name from user query.
        
        Args:
            user_query: Natural language query
            
        Returns:
            Optional[str]: Potential collection name or None
        """
        # Common collection name patterns
        collection_patterns = [
            r'\b(users?|customers?|clients?)\b',
            r'\b(products?|items?|goods?)\b',
            r'\b(orders?|purchases?|transactions?)\b',
            r'\b(posts?|articles?|blogs?)\b',
            r'\b(comments?|reviews?|feedback?)\b',
            r'\b(categories?|tags?|labels?)\b',
            r'\b(files?|documents?|records?)\b',
            r'\b(events?|activities?|logs?)\b',
        ]
        
        user_query_lower = user_query.lower()
        
        for pattern in collection_patterns:
            match = re.search(pattern, user_query_lower)
            if match:
                collection_name = match.group(1)
                # Convert to common collection naming convention
                if collection_name.endswith('s'):
                    return collection_name
                else:
                    return collection_name + 's'
        
        return None
    
    def translate_query(
        self,
        user_query: str,
        collection_name: Optional[str] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        collection_info: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Translate natural language query to MongoDB query.
        
        Args:
            user_query: Natural language query from user
            collection_name: Target collection name (auto-detected if not provided)
            database_schema: Database schema information
            collection_info: Collection structure information
            additional_context: Additional context for query generation
            
        Returns:
            Optional[Dict]: Translated and validated MongoDB query or None if failed
        """
        try:
            # Extract collection hint if not provided
            if not collection_name:
                collection_name = self._extract_collection_hint(user_query)
                if collection_name:
                    logger.info(f"Auto-detected collection: {collection_name}")
            
            # Enhance query with context
            enhanced_query = user_query
            if additional_context:
                enhanced_query = f"{additional_context}\n\nUser Query: {user_query}"
            
            # Generate query using Ollama client
            raw_query = self.ollama_client.generate_mongodb_query(
                user_query=enhanced_query,
                database_schema=database_schema,
                collection_info=collection_info,
                examples=self.query_examples
            )
            
            if not raw_query:
                logger.error("Failed to generate query from Ollama client")
                return None
            
            # Sanitize the query
            sanitized_query = self._sanitize_query(raw_query)
            
            # Validate query structure
            is_valid, validation_errors = self._validate_query_structure(sanitized_query)
            if not is_valid:
                logger.error(f"Query validation failed: {validation_errors}")
                return None
            
            # Optimize the query
            optimized_query = self._optimize_query(sanitized_query)
            
            # Add metadata
            optimized_query['_metadata'] = {
                'original_query': user_query,
                'collection_hint': collection_name,
                'generated_at': None,  # Will be set by the agent
                'sanitized': sanitized_query != raw_query,
                'optimized': optimized_query != sanitized_query
            }
            
            logger.info(f"Successfully translated query: {user_query}")
            return optimized_query
            
        except Exception as e:
            logger.error(f"Error translating query: {e}")
            return None
    
    def validate_and_explain_query(
        self, 
        query: Dict[str, Any]
    ) -> Tuple[bool, List[str], Optional[str]]:
        """
        Validate query and generate explanation.
        
        Args:
            query: MongoDB query dictionary
            
        Returns:
            Tuple: (is_valid, validation_errors, explanation)
        """
        # Validate structure
        is_valid, errors = self._validate_query_structure(query)
        
        # Check safety
        is_safe, safety_warnings = self.ollama_client.validate_query_safety(query)
        if not is_safe:
            errors.extend(safety_warnings)
            is_valid = False
        
        # Generate explanation
        explanation = None
        if is_valid:
            explanation = self.ollama_client.explain_query(query)
        
        return is_valid, errors, explanation
    
    def suggest_query_improvements(
        self, 
        query: Dict[str, Any], 
        performance_hints: Optional[List[str]] = None
    ) -> List[str]:
        """
        Suggest improvements for a MongoDB query.
        
        Args:
            query: MongoDB query dictionary
            performance_hints: Additional performance hints
            
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        
        # Check for missing indexes hint
        if 'filter' in query and query['filter']:
            filter_fields = self._extract_filter_fields(query['filter'])
            if filter_fields:
                suggestions.append(
                    f"Consider creating indexes on fields: {', '.join(filter_fields)}"
                )
        
        # Check for large result sets
        if query.get('operation') == 'find':
            if 'limit' not in query:
                suggestions.append("Consider adding a limit to prevent large result sets")
            elif query.get('limit', 0) > 100:
                suggestions.append("Large limit may impact performance")
        
        # Check for inefficient regex
        if self._has_inefficient_regex(query):
            suggestions.append("Regex patterns without anchors may be slow on large collections")
        
        # Add performance hints if provided
        if performance_hints:
            suggestions.extend(performance_hints)
        
        return suggestions
    
    def _extract_filter_fields(self, filter_dict: Dict[str, Any]) -> Set[str]:
        """Extract field names from filter dictionary."""
        fields = set()
        
        def extract_fields(d: Dict[str, Any], prefix: str = ""):
            for key, value in d.items():
                if key.startswith('$'):
                    # Skip operators, but process their values
                    if isinstance(value, dict):
                        extract_fields(value, prefix)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                extract_fields(item, prefix)
                else:
                    # This is a field name
                    field_name = f"{prefix}.{key}" if prefix else key
                    fields.add(field_name)
                    
                    if isinstance(value, dict):
                        extract_fields(value, field_name)
        
        extract_fields(filter_dict)
        return fields
    
    def _has_inefficient_regex(self, query: Dict[str, Any]) -> bool:
        """Check if query contains potentially inefficient regex patterns."""
        def check_regex_in_dict(d: Dict[str, Any]) -> bool:
            for key, value in d.items():
                if key == '$regex' and isinstance(value, str):
                    # Check for unanchored regex patterns
                    if not value.startswith('^') and ('.*' in value or '.+' in value):
                        return True
                elif isinstance(value, dict):
                    if check_regex_in_dict(value):
                        return True
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and check_regex_in_dict(item):
                            return True
            return False
        
        return check_regex_in_dict(query)
