# 🤖 AI MongoDB Query Agent

A powerful natural language interface for MongoDB querying powered by Ollama and Streamlit. This application allows users to interact with MongoDB databases using plain English queries, which are automatically translated into MongoDB operations using AI.

## ✨ Features

- **Natural Language Querying**: Ask questions in plain English and get MongoDB results
- **AI-Powered Translation**: Uses Ollama LLMs to convert natural language to MongoDB queries
- **Interactive Web Interface**: Beautiful Streamlit-based chat interface
- **Query History**: Track and review your previous queries and results
- **Database Management**: Easy database and collection selection
- **Result Visualization**: Smart formatting and visualization of query results
- **Safety & Validation**: Query sanitization and validation for secure operations
- **Conversation Memory**: Context-aware conversations for better query understanding
- **Export Capabilities**: Download query results in CSV format

## 🏗️ Architecture

The application consists of several key components:

- **MongoDB Connection Manager**: Handles database connections with connection pooling
- **Ollama Client Wrapper**: Manages LLM interactions for query generation
- **Query Translator**: Converts natural language to validated MongoDB queries
- **AI Agent**: Orchestrates the entire query process with conversation memory
- **Streamlit Web App**: Provides the user interface and visualization
- **Configuration Management**: Centralized settings with environment variable support

## 📋 Prerequisites

Before setting up the AI MongoDB Query Agent, ensure you have the following installed:

### Required Software

1. **Python 3.9+**
   ```bash
   python --version  # Should be 3.9 or higher
   ```

2. **MongoDB Server**
   - Install MongoDB Community Edition: https://docs.mongodb.com/manual/installation/
   - Or use MongoDB Atlas (cloud): https://www.mongodb.com/atlas
   - Ensure MongoDB is running and accessible

3. **Ollama**
   - Install Ollama: https://ollama.ai/
   - Pull a compatible model (e.g., llama2, codellama, mistral)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-mongodb-agent
```

### 2. Install Python Dependencies

```bash
# Using the startup script (recommended)
python run_app.py --install-deps

# Or manually with pip
pip install -r requirements.txt
```

### 3. Set Up Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama2
```

### 4. Configure MongoDB

Ensure MongoDB is running:

```bash
# For local MongoDB
mongod

# Or check if MongoDB service is running
sudo systemctl status mongod
```

### 5. Configure Environment

```bash
# Create environment configuration
python run_app.py --create-env

# Copy and edit the environment file
cp .env.template .env
# Edit .env with your specific settings
```

### 6. Launch the Application

```bash
# Run with dependency checks
python run_app.py

# Or run on a custom port
python run_app.py --port 8080

# Check dependencies only
python run_app.py --check-only
```

The application will be available at `http://localhost:8501` (or your specified port).

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with your configuration:

```env
# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USERNAME=your_username
MONGODB_PASSWORD=your_password
MONGODB_DATABASE=your_database

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TEMPERATURE=0.1

# Application Settings
ENVIRONMENT=development
DEBUG=false
```

### MongoDB Setup Examples

#### Local MongoDB (No Authentication)
```env
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=myapp
```

#### Local MongoDB (With Authentication)
```env
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USERNAME=myuser
MONGODB_PASSWORD=mypassword
MONGODB_DATABASE=myapp
MONGODB_AUTH_SOURCE=admin
```

#### MongoDB Atlas (Cloud)
```env
MONGODB_HOST=cluster0.xxxxx.mongodb.net
MONGODB_PORT=27017
MONGODB_USERNAME=myuser
MONGODB_PASSWORD=mypassword
MONGODB_DATABASE=myapp
MONGODB_USE_SSL=true
```

### Ollama Model Configuration

The application supports various Ollama models:

```env
# Recommended models
OLLAMA_MODEL=llama2          # General purpose, good balance
OLLAMA_MODEL=codellama       # Better for technical queries
OLLAMA_MODEL=mistral         # Fast and efficient
OLLAMA_MODEL=llama2:13b      # More accurate, slower
```

## 📖 Usage Examples

### Basic Queries

Once the application is running, you can ask questions like:

**Finding Documents:**
- "Find all users with age greater than 25"
- "Show me products with price between 100 and 500"
- "Get customers from New York"

**Counting Documents:**
- "How many active users do we have?"
- "Count orders from last month"
- "How many products are in stock?"

**Text Search:**
- "Find users with email containing gmail"
- "Search for products with 'laptop' in the name"
- "Show posts containing the word 'mongodb'"

**Sorting and Limiting:**
- "Show me the top 10 highest priced products"
- "Get the 5 most recent orders"
- "Find users sorted by registration date"

### Advanced Queries

**Aggregation:**
- "Group users by city and count them"
- "Show average order value by month"
- "Find the most popular product categories"

**Date Queries:**
- "Find orders created after January 1st, 2024"
- "Show users registered in the last 30 days"
- "Get sales data for this year"

### Using the Web Interface

1. **Configure Connection**: Use the sidebar to set up your MongoDB and Ollama connections
2. **Select Database**: Choose your target database and collection
3. **Ask Questions**: Type your natural language queries in the chat interface
4. **Review Results**: View formatted results with export options
5. **Check History**: Review previous queries in the History tab

## 🛠️ Development

### Project Structure

```
ai-mongodb-agent/
├── src/python/                 # Python source code
│   ├── agents/                 # AI agents and query translation
│   │   ├── mongo_agent.py      # Core AI agent
│   │   └── query_translator.py # Natural language translator
│   ├── database/               # Database connection management
│   │   └── mongo_client.py     # MongoDB client wrapper
│   ├── llm/                    # LLM integration
│   │   └── ollama_client.py    # Ollama client wrapper
│   ├── config/                 # Configuration management
│   │   └── settings.py         # Pydantic settings
│   └── app.py                  # Streamlit web application
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Python project configuration
├── run_app.py                 # Application startup script
└── README.md                  # This file
```

### Running in Development Mode

```bash
# Set development environment
export ENVIRONMENT=development
export DEBUG=true

# Run with verbose logging
python run_app.py --verbose

# Skip dependency checks for faster startup
python run_app.py --skip-checks
```

### Adding New Features

1. **Extend Query Translator**: Add new query patterns in `query_translator.py`
2. **Enhance UI**: Modify the Streamlit interface in `app.py`
3. **Add Database Support**: Extend the MongoDB client in `mongo_client.py`
4. **Improve AI Responses**: Customize prompts in `ollama_client.py`

## 🔧 Troubleshooting

### Common Issues

**MongoDB Connection Failed**
```bash
# Check if MongoDB is running
sudo systemctl status mongod

# Check connection manually
mongo --host localhost --port 27017
```

**Ollama Server Not Available**
```bash
# Start Ollama server
ollama serve

# Check if model is available
ollama list

# Pull required model
ollama pull llama2
```

**Python Dependencies Missing**
```bash
# Install dependencies
python run_app.py --install-deps

# Or manually
pip install -r requirements.txt
```

**Port Already in Use**
```bash
# Run on different port
python run_app.py --port 8080

# Check what's using the port
lsof -i :8501
```

### Performance Optimization

**For Better Query Performance:**
- Create appropriate indexes in MongoDB
- Use specific collection names in queries
- Limit result sets for large collections

**For Better AI Performance:**
- Use faster Ollama models (e.g., mistral)
- Adjust temperature settings for consistency
- Provide clear, specific queries

## 📚 API Reference

### Command Line Options

```bash
python run_app.py [OPTIONS]

Options:
  --port, -p INTEGER     Port to run the application (default: 8501)
  --host TEXT           Host address to bind to (default: 0.0.0.0)
  --check-only          Only check dependencies without launching
  --install-deps        Install Python dependencies and exit
  --create-env          Create environment template file and exit
  --skip-checks         Skip dependency checks and launch directly
  --verbose, -v         Enable verbose logging
  --help               Show help message and exit
```

### Environment Variables Reference

See the generated `.env.template` file for a complete list of available configuration options.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for providing the LLM infrastructure
- [Streamlit](https://streamlit.io/) for the web application framework
- [MongoDB](https://www.mongodb.com/) for the database platform
- [Pydantic](https://pydantic-docs.helpmanual.io/) for configuration management

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [Issues](../../issues) page
3. Create a new issue with detailed information about your problem

---

**Happy Querying! 🚀**

