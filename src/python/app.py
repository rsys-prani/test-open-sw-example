"""
Streamlit web application for AI MongoDB querying agent.

This module provides a comprehensive web interface for natural language
MongoDB querying with chat interface, query history, database configuration,
and result visualization components.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from streamlit.runtime.state import SessionState

from .agents.mongo_agent import MongoAgent
from .database.mongo_client import MongoConnectionManager
from .llm.ollama_client import OllamaClient


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI MongoDB Query Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    .agent-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    .error-message {
        background-color: #ffe6e6;
        border-left-color: #ff4444;
        color: #cc0000;
    }
    .success-message {
        background-color: #e6ffe6;
        border-left-color: #44ff44;
        color: #006600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .query-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = {
            'mongodb': False,
            'ollama': False,
            'agent_initialized': False
        }
    
    if 'current_database' not in st.session_state:
        st.session_state.current_database = None
    
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    
    if 'query_suggestions' not in st.session_state:
        st.session_state.query_suggestions = []


def create_mongodb_client(host: str, port: int, username: str, password: str, database: str) -> MongoConnectionManager:
    """Create MongoDB client with provided configuration."""
    return MongoConnectionManager(
        host=host,
        port=port,
        username=username if username else None,
        password=password if password else None,
        database_name=database if database else None
    )


def create_ollama_client(host: str, model: str) -> OllamaClient:
    """Create Ollama client with provided configuration."""
    return OllamaClient(
        host=host,
        model=model,
        temperature=0.1,
        max_tokens=2048
    )


def initialize_agent(mongo_config: Dict[str, Any], ollama_config: Dict[str, Any]) -> Optional[MongoAgent]:
    """Initialize the AI agent with provided configurations."""
    try:
        # Create MongoDB client
        mongo_client = create_mongodb_client(**mongo_config)
        
        # Create Ollama client
        ollama_client = create_ollama_client(**ollama_config)
        
        # Create agent
        agent = MongoAgent(
            mongo_client=mongo_client,
            ollama_client=ollama_client,
            default_database=mongo_config.get('database'),
            max_conversation_history=50,
            result_limit=100
        )
        
        # Initialize agent
        if agent.initialize():
            st.session_state.connection_status = {
                'mongodb': mongo_client.is_connected(),
                'ollama': ollama_client.is_available(),
                'agent_initialized': True
            }
            return agent
        else:
            st.error("Failed to initialize AI agent")
            return None
            
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        logger.error(f"Error initializing agent: {e}")
        return None


def render_sidebar():
    """Render the sidebar with configuration and status."""
    st.sidebar.markdown("## 🔧 Configuration")
    
    # MongoDB Configuration
    st.sidebar.markdown("### MongoDB Settings")
    mongo_host = st.sidebar.text_input("MongoDB Host", value="localhost", key="mongo_host")
    mongo_port = st.sidebar.number_input("MongoDB Port", value=27017, min_value=1, max_value=65535, key="mongo_port")
    mongo_username = st.sidebar.text_input("Username (optional)", key="mongo_username")
    mongo_password = st.sidebar.text_input("Password (optional)", type="password", key="mongo_password")
    mongo_database = st.sidebar.text_input("Default Database (optional)", key="mongo_database")
    
    # Ollama Configuration
    st.sidebar.markdown("### Ollama Settings")
    ollama_host = st.sidebar.text_input("Ollama Host", value="http://localhost:11434", key="ollama_host")
    ollama_model = st.sidebar.text_input("Model Name", value="llama2", key="ollama_model")
    
    # Initialize/Reconnect button
    if st.sidebar.button("🔄 Initialize/Reconnect", type="primary"):
        mongo_config = {
            'host': mongo_host,
            'port': mongo_port,
            'username': mongo_username,
            'password': mongo_password,
            'database': mongo_database
        }
        
        ollama_config = {
            'host': ollama_host,
            'model': ollama_model
        }
        
        with st.spinner("Initializing AI agent..."):
            agent = initialize_agent(mongo_config, ollama_config)
            if agent:
                st.session_state.agent = agent
                st.session_state.current_database = mongo_database
                st.success("✅ Agent initialized successfully!")
                st.rerun()
    
    # Connection Status
    st.sidebar.markdown("### 📊 Connection Status")
    status = st.session_state.connection_status
    
    mongo_status = "🟢 Connected" if status['mongodb'] else "🔴 Disconnected"
    ollama_status = "🟢 Available" if status['ollama'] else "🔴 Unavailable"
    agent_status = "🟢 Ready" if status['agent_initialized'] else "🔴 Not Initialized"
    
    st.sidebar.markdown(f"**MongoDB:** {mongo_status}")
    st.sidebar.markdown(f"**Ollama:** {ollama_status}")
    st.sidebar.markdown(f"**Agent:** {agent_status}")
    
    # Database and Collection Selection
    if st.session_state.agent and status['agent_initialized']:
        st.sidebar.markdown("### 🗄️ Database Selection")
        
        try:
            db_info = st.session_state.agent.get_database_info()
            databases = db_info.get('databases', [])
            
            if databases:
                selected_db = st.sidebar.selectbox(
                    "Select Database",
                    options=databases,
                    index=databases.index(st.session_state.current_database) if st.session_state.current_database in databases else 0,
                    key="selected_database"
                )
                
                if selected_db != st.session_state.current_database:
                    if st.session_state.agent.set_database(selected_db):
                        st.session_state.current_database = selected_db
                        st.rerun()
                
                # Collection selection
                collections = db_info.get('collections', {}).get(selected_db, [])
                if collections:
                    selected_collection = st.sidebar.selectbox(
                        "Select Collection (optional)",
                        options=["Auto-detect"] + collections,
                        key="selected_collection"
                    )
                    
                    if selected_collection != "Auto-detect":
                        if st.session_state.agent.set_collection(selected_collection):
                            st.session_state.current_collection = selected_collection
                    else:
                        st.session_state.current_collection = None
        
        except Exception as e:
            st.sidebar.error(f"Error loading database info: {str(e)}")
    
    # Clear History Button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.agent:
            st.session_state.agent.clear_conversation_history()
        st.rerun()


def render_chat_interface():
    """Render the main chat interface."""
    st.markdown('<h1 class="main-header">🤖 AI MongoDB Query Agent</h1>', unsafe_allow_html=True)
    
    if not st.session_state.agent or not st.session_state.connection_status['agent_initialized']:
        st.warning("⚠️ Please configure and initialize the agent in the sidebar first.")
        return
    
    # Query suggestions
    if st.session_state.query_suggestions:
        st.markdown("### 💡 Query Suggestions")
        cols = st.columns(min(len(st.session_state.query_suggestions), 3))
        for i, suggestion in enumerate(st.session_state.query_suggestions[:3]):
            with cols[i]:
                if st.button(f"💭 {suggestion[:50]}...", key=f"suggestion_{i}"):
                    st.session_state.user_input = suggestion
                    st.rerun()
    
    # Chat history
    st.markdown("### 💬 Conversation")
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            
            elif message['type'] == 'agent':
                st.markdown(f"""
                <div class="chat-message agent-message">
                    <strong>🤖 Agent:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show query results if available
                if 'results' in message and message['results']:
                    render_query_results(message['results'])
            
            elif message['type'] == 'error':
                st.markdown(f"""
                <div class="chat-message error-message">
                    <strong>❌ Error:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Query input
    st.markdown("### ✍️ Ask a Question")
    
    # Get query suggestions
    if st.session_state.agent:
        try:
            suggestions = st.session_state.agent.get_query_suggestions()
            st.session_state.query_suggestions = suggestions
        except Exception as e:
            logger.error(f"Error getting query suggestions: {e}")
    
    # Input form
    with st.form("query_form", clear_on_submit=True):
        user_input = st.text_area(
            "Enter your natural language query:",
            placeholder="e.g., Find all users with age greater than 25",
            height=100,
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit_button = st.form_submit_button("🚀 Send Query", type="primary")
        with col2:
            include_explanation = st.checkbox("Include explanation", value=True)
        
        if submit_button and user_input.strip():
            process_user_query(user_input.strip(), include_explanation)


def process_user_query(user_query: str, include_explanation: bool = True):
    """Process user query and display results."""
    # Add user message to chat history
    st.session_state.chat_history.append({
        'type': 'user',
        'content': user_query,
        'timestamp': datetime.now().isoformat()
    })
    
    # Process query with agent
    with st.spinner("🔍 Processing your query..."):
        try:
            response = st.session_state.agent.process_query(
                user_query=user_query,
                include_explanation=include_explanation
            )
            
            if response['success']:
                # Create agent response message
                agent_message = f"Found {response['result_count']} results"
                if response.get('explanation'):
                    agent_message += f"\n\n**Query Explanation:**\n{response['explanation']}"
                
                st.session_state.chat_history.append({
                    'type': 'agent',
                    'content': agent_message,
                    'results': response,
                    'timestamp': datetime.now().isoformat()
                })
                
            else:
                # Add error message
                st.session_state.chat_history.append({
                    'type': 'error',
                    'content': response.get('error_message', 'Unknown error occurred'),
                    'timestamp': datetime.now().isoformat()
                })
        
        except Exception as e:
            st.session_state.chat_history.append({
                'type': 'error',
                'content': f"Error processing query: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })
            logger.error(f"Error processing query: {e}")
    
    st.rerun()


def render_query_results(response: Dict[str, Any]):
    """Render query results in a formatted way."""
    if not response or not response.get('success'):
        return
    
    results = response.get('results', {})
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Results Found", response.get('result_count', 0))
    
    with col2:
        execution_time = response.get('execution_time', 0)
        st.metric("Execution Time", f"{execution_time:.3f}s")
    
    with col3:
        st.metric("Database", response.get('database', 'N/A'))
    
    with col4:
        st.metric("Collection", response.get('collection', 'N/A'))
    
    # Show generated MongoDB query
    if response.get('generated_query'):
        with st.expander("🔍 Generated MongoDB Query", expanded=False):
            query_without_metadata = {k: v for k, v in response['generated_query'].items() if k != '_metadata'}
            st.code(json.dumps(query_without_metadata, indent=2), language='json')
    
    # Display results based on type
    if results.get('type') == 'count':
        st.success(f"📊 Count result: **{results.get('count', 0)}** documents")
    
    elif results.get('type') in ['documents', 'document']:
        data = results.get('data', [])
        
        if data:
            # Show sample data in expandable section
            with st.expander(f"📄 Results Preview (showing {len(results.get('sample', []))} of {results.get('count', 0)})", expanded=True):
                
                # Try to display as DataFrame if possible
                try:
                    if isinstance(data, list) and len(data) > 0:
                        # Convert to DataFrame for better display
                        df = pd.json_normalize(data)
                        
                        # Limit columns for better display
                        if len(df.columns) > 10:
                            st.info(f"Showing first 10 columns out of {len(df.columns)} total columns")
                            df = df.iloc[:, :10]
                        
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button for full results
                        if len(data) > 5:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name=f"mongodb_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    else:
                        st.json(data)
                
                except Exception as e:
                    # Fallback to JSON display
                    st.json(results.get('sample', data))
                    logger.warning(f"Could not display as DataFrame: {e}")
        
        else:
            st.info("No results found for your query.")
    
    else:
        # Unknown result type, display as JSON
        st.json(results)


def render_query_history():
    """Render query history tab."""
    st.markdown("### 📚 Query History")
    
    if not st.session_state.agent:
        st.warning("Agent not initialized")
        return
    
    try:
        history = st.session_state.agent.get_conversation_history(20)
        
        if not history:
            st.info("No query history available")
            return
        
        # Create DataFrame for history
        history_data = []
        for interaction in history:
            history_data.append({
                'Timestamp': interaction.get('timestamp', ''),
                'Query': interaction.get('user_query', '')[:100] + '...' if len(interaction.get('user_query', '')) > 100 else interaction.get('user_query', ''),
                'Success': '✅' if interaction.get('success') else '❌',
                'Results': interaction.get('result_count', 0),
                'Time (s)': f"{interaction.get('execution_time', 0):.3f}"
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed view
        st.markdown("### 🔍 Detailed History")
        for i, interaction in enumerate(reversed(history[-10:])):  # Show last 10 in detail
            with st.expander(f"Query {len(history) - i}: {interaction.get('user_query', '')[:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**User Query:**")
                    st.text(interaction.get('user_query', ''))
                    
                    st.markdown("**Status:**")
                    status = "✅ Success" if interaction.get('success') else "❌ Failed"
                    st.markdown(status)
                
                with col2:
                    st.markdown("**Results:**")
                    st.text(f"Count: {interaction.get('result_count', 0)}")
                    st.text(f"Execution Time: {interaction.get('execution_time', 0):.3f}s")
                    
                    if interaction.get('error_message'):
                        st.markdown("**Error:**")
                        st.error(interaction['error_message'])
                
                if interaction.get('generated_query'):
                    st.markdown("**Generated MongoDB Query:**")
                    query_without_metadata = {k: v for k, v in interaction['generated_query'].items() if k != '_metadata'}
                    st.code(json.dumps(query_without_metadata, indent=2), language='json')
    
    except Exception as e:
        st.error(f"Error loading query history: {str(e)}")
        logger.error(f"Error loading query history: {e}")


def main():
    """Main application function."""
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📚 History"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_query_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 **AI MongoDB Query Agent** - Natural language interface for MongoDB querying powered by Ollama"
    )


if __name__ == "__main__":
    main()
