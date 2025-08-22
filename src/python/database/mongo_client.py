"""
MongoDB connection manager with connection pooling and database operations.

This module provides a robust MongoDB client wrapper that handles connection
pooling, database selection, collection access, and basic CRUD operations
with proper error handling and logging.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    OperationFailure,
    PyMongoError,
)


logger = logging.getLogger(__name__)


class MongoConnectionManager:
    """
    MongoDB connection manager with connection pooling and database operations.
    
    This class provides a high-level interface for MongoDB operations with
    automatic connection management, error handling, and connection pooling.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database_name: Optional[str] = None,
        auth_source: str = "admin",
        max_pool_size: int = 100,
        min_pool_size: int = 0,
        max_idle_time_ms: int = 30000,
        server_selection_timeout_ms: int = 5000,
        connect_timeout_ms: int = 10000,
        socket_timeout_ms: int = 20000,
        **kwargs
    ):
        """
        Initialize MongoDB connection manager.
        
        Args:
            host: MongoDB host address
            port: MongoDB port number
            username: Username for authentication
            password: Password for authentication
            database_name: Default database name
            auth_source: Authentication database
            max_pool_size: Maximum number of connections in the pool
            min_pool_size: Minimum number of connections in the pool
            max_idle_time_ms: Maximum idle time for connections
            server_selection_timeout_ms: Server selection timeout
            connect_timeout_ms: Connection timeout
            socket_timeout_ms: Socket timeout
            **kwargs: Additional MongoDB client options
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database_name
        self.auth_source = auth_source
        
        # Connection pool settings
        self.connection_options = {
            "maxPoolSize": max_pool_size,
            "minPoolSize": min_pool_size,
            "maxIdleTimeMS": max_idle_time_ms,
            "serverSelectionTimeoutMS": server_selection_timeout_ms,
            "connectTimeoutMS": connect_timeout_ms,
            "socketTimeoutMS": socket_timeout_ms,
            **kwargs
        }
        
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        
    def _build_connection_string(self) -> str:
        """Build MongoDB connection string."""
        if self.username and self.password:
            # URL encode username and password to handle special characters
            encoded_username = quote_plus(self.username)
            encoded_password = quote_plus(self.password)
            auth_part = f"{encoded_username}:{encoded_password}@"
        else:
            auth_part = ""
            
        connection_string = f"mongodb://{auth_part}{self.host}:{self.port}/"
        
        if self.username and self.password:
            connection_string += f"?authSource={self.auth_source}"
            
        return connection_string
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            connection_string = self._build_connection_string()
            
            self._client = MongoClient(
                connection_string,
                **self.connection_options
            )
            
            # Test the connection
            self._client.admin.command('ping')
            
            if self.database_name:
                self._database = self._client[self.database_name]
                
            logger.info(f"Successfully connected to MongoDB at {self.host}:{self.port}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB connection: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("Disconnected from MongoDB")
    
    def is_connected(self) -> bool:
        """
        Check if MongoDB connection is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self._client:
            return False
            
        try:
            self._client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def get_client(self) -> Optional[MongoClient]:
        """
        Get the MongoDB client instance.
        
        Returns:
            Optional[MongoClient]: MongoDB client or None if not connected
        """
        return self._client
    
    def get_database(self, database_name: Optional[str] = None) -> Optional[Database]:
        """
        Get database instance.
        
        Args:
            database_name: Database name (uses default if not provided)
            
        Returns:
            Optional[Database]: Database instance or None if not connected
        """
        if not self._client:
            logger.error("Not connected to MongoDB")
            return None
            
        db_name = database_name or self.database_name
        if not db_name:
            logger.error("No database name provided")
            return None
            
        return self._client[db_name]
    
    def get_collection(
        self, 
        collection_name: str, 
        database_name: Optional[str] = None
    ) -> Optional[Collection]:
        """
        Get collection instance.
        
        Args:
            collection_name: Collection name
            database_name: Database name (uses default if not provided)
            
        Returns:
            Optional[Collection]: Collection instance or None if not available
        """
        database = self.get_database(database_name)
        if not database:
            return None
            
        return database[collection_name]
    
    def list_databases(self) -> List[str]:
        """
        List all databases.
        
        Returns:
            List[str]: List of database names
        """
        if not self._client:
            logger.error("Not connected to MongoDB")
            return []
            
        try:
            return [db['name'] for db in self._client.list_databases()]
        except Exception as e:
            logger.error(f"Error listing databases: {e}")
            return []
    
    def list_collections(self, database_name: Optional[str] = None) -> List[str]:
        """
        List all collections in a database.
        
        Args:
            database_name: Database name (uses default if not provided)
            
        Returns:
            List[str]: List of collection names
        """
        database = self.get_database(database_name)
        if not database:
            return []
            
        try:
            return database.list_collection_names()
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def execute_query(
        self,
        collection_name: str,
        operation: str,
        query: Dict[str, Any],
        database_name: Optional[str] = None,
        **kwargs
    ) -> Optional[Union[List[Dict], Dict, int]]:
        """
        Execute a MongoDB query operation.
        
        Args:
            collection_name: Collection name
            operation: Operation type ('find', 'find_one', 'count', 'aggregate')
            query: Query parameters
            database_name: Database name (uses default if not provided)
            **kwargs: Additional operation parameters
            
        Returns:
            Query results or None if error occurred
        """
        collection = self.get_collection(collection_name, database_name)
        if not collection:
            return None
            
        try:
            if operation == 'find':
                cursor = collection.find(query.get('filter', {}), query.get('projection'))
                if 'limit' in query:
                    cursor = cursor.limit(query['limit'])
                if 'skip' in query:
                    cursor = cursor.skip(query['skip'])
                if 'sort' in query:
                    cursor = cursor.sort(query['sort'])
                return list(cursor)
                
            elif operation == 'find_one':
                return collection.find_one(
                    query.get('filter', {}), 
                    query.get('projection')
                )
                
            elif operation == 'count':
                return collection.count_documents(query.get('filter', {}))
                
            elif operation == 'aggregate':
                pipeline = query.get('pipeline', [])
                return list(collection.aggregate(pipeline))
                
            else:
                logger.error(f"Unsupported operation: {operation}")
                return None
                
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed: {e}")
            return None
        except PyMongoError as e:
            logger.error(f"PyMongo error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during query execution: {e}")
            return None
    
    def get_collection_stats(
        self, 
        collection_name: str, 
        database_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Collection name
            database_name: Database name (uses default if not provided)
            
        Returns:
            Optional[Dict]: Collection statistics or None if error occurred
        """
        database = self.get_database(database_name)
        if not database:
            return None
            
        try:
            stats = database.command("collStats", collection_name)
            return {
                'count': stats.get('count', 0),
                'size': stats.get('size', 0),
                'avgObjSize': stats.get('avgObjSize', 0),
                'storageSize': stats.get('storageSize', 0),
                'indexes': stats.get('nindexes', 0),
                'totalIndexSize': stats.get('totalIndexSize', 0)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
