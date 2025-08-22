#!/usr/bin/env python3
"""
Startup script for AI MongoDB Query Agent.

This script initializes the environment, checks dependencies (Ollama server,
MongoDB connection), and launches the Streamlit application with proper
error handling and user feedback.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Add src/python to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

try:
    from config.settings import get_settings, create_env_template
except ImportError as e:
    print(f"❌ Error importing configuration: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencyChecker:
    """Handles checking and validation of application dependencies."""
    
    def __init__(self, settings):
        self.settings = settings
        self.mongodb_settings = settings.mongodb
        self.ollama_settings = settings.ollama
    
    def check_python_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Check if required Python packages are installed.
        
        Returns:
            Tuple[bool, List[str]]: (success, missing_packages)
        """
        required_packages = [
            'streamlit',
            'pymongo',
            'ollama',
            'pydantic',
            'python-dotenv',
            'pandas',
            'numpy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        return len(missing_packages) == 0, missing_packages
    
    def check_mongodb_connection(self) -> Tuple[bool, str]:
        """
        Check MongoDB connection.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Build connection string
            if self.mongodb_settings.username and self.mongodb_settings.password:
                auth_part = f"{self.mongodb_settings.username}:{self.mongodb_settings.password}@"
            else:
                auth_part = ""
            
            connection_string = f"mongodb://{auth_part}{self.mongodb_settings.host}:{self.mongodb_settings.port}/"
            
            # Create client with short timeout for quick check
            client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000
            )
            
            # Test connection
            client.admin.command('ping')
            client.close()
            
            return True, f"✅ MongoDB connection successful at {self.mongodb_settings.host}:{self.mongodb_settings.port}"
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            return False, f"❌ MongoDB connection failed: {str(e)}"
        except Exception as e:
            return False, f"❌ MongoDB connection error: {str(e)}"
    
    def check_ollama_server(self) -> Tuple[bool, str]:
        """
        Check Ollama server availability.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Check if Ollama server is running
            response = requests.get(
                f"{self.ollama_settings.host}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                # Check if the configured model is available
                if self.ollama_settings.model in model_names:
                    return True, f"✅ Ollama server running with model '{self.ollama_settings.model}' available"
                else:
                    return False, f"⚠️ Ollama server running but model '{self.ollama_settings.model}' not found. Available models: {model_names}"
            else:
                return False, f"❌ Ollama server responded with status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, f"❌ Cannot connect to Ollama server at {self.ollama_settings.host}"
        except requests.exceptions.Timeout:
            return False, f"❌ Timeout connecting to Ollama server at {self.ollama_settings.host}"
        except Exception as e:
            return False, f"❌ Ollama server check failed: {str(e)}"
    
    def check_all_dependencies(self) -> Dict[str, Tuple[bool, str]]:
        """
        Check all application dependencies.
        
        Returns:
            Dict[str, Tuple[bool, str]]: Dependency check results
        """
        results = {}
        
        # Check Python dependencies
        python_ok, missing_packages = self.check_python_dependencies()
        if python_ok:
            results['python'] = (True, "✅ All Python dependencies installed")
        else:
            results['python'] = (False, f"❌ Missing Python packages: {', '.join(missing_packages)}")
        
        # Check MongoDB
        results['mongodb'] = self.check_mongodb_connection()
        
        # Check Ollama
        results['ollama'] = self.check_ollama_server()
        
        return results


class AppLauncher:
    """Handles application launch and environment setup."""
    
    def __init__(self, settings):
        self.settings = settings
    
    def setup_environment(self) -> bool:
        """
        Set up environment variables and configuration.
        
        Returns:
            bool: Success status
        """
        try:
            # Set Streamlit configuration
            os.environ['STREAMLIT_SERVER_PORT'] = '8501'
            os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
            os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
            os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
            
            # Set logging level
            log_level = self.settings.get_log_level()
            os.environ['PYTHONPATH'] = str(Path(__file__).parent / "src" / "python")
            
            logger.info(f"Environment configured with log level: {log_level}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            return False
    
    def launch_streamlit(self, port: int = 8501, host: str = "0.0.0.0") -> None:
        """
        Launch Streamlit application.
        
        Args:
            port: Port to run the application on
            host: Host address to bind to
        """
        try:
            app_path = Path(__file__).parent / "src" / "python" / "app.py"
            
            if not app_path.exists():
                raise FileNotFoundError(f"Streamlit app not found at {app_path}")
            
            # Build streamlit command
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                "--server.port", str(port),
                "--server.address", host,
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            logger.info(f"Launching Streamlit application on {host}:{port}")
            print(f"\n🚀 Starting AI MongoDB Query Agent...")
            print(f"📱 Open your browser and navigate to: http://localhost:{port}")
            print(f"🛑 Press Ctrl+C to stop the application\n")
            
            # Launch Streamlit
            subprocess.run(cmd, check=True)
            
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to launch Streamlit: {e}")
            print(f"❌ Failed to launch Streamlit application: {e}")
        except Exception as e:
            logger.error(f"Unexpected error launching application: {e}")
            print(f"❌ Unexpected error: {e}")


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🤖 AI MongoDB Query Agent                             ║
║                                                              ║
║        Natural Language Interface for MongoDB Querying      ║
║        Powered by Ollama and Streamlit                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_dependency_status(results: Dict[str, Tuple[bool, str]]):
    """Print dependency check results."""
    print("\n📋 Dependency Check Results:")
    print("=" * 50)
    
    all_ok = True
    for name, (success, message) in results.items():
        print(f"{name.upper()}: {message}")
        if not success:
            all_ok = False
    
    print("=" * 50)
    
    if all_ok:
        print("✅ All dependencies are ready!")
    else:
        print("❌ Some dependencies need attention before launching the application.")
    
    return all_ok


def install_python_dependencies() -> bool:
    """
    Install Python dependencies using pip.
    
    Returns:
        bool: Success status
    """
    try:
        requirements_file = Path(__file__).parent / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        print("📦 Installing Python dependencies...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Python dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_sample_env_file():
    """Create sample environment file if it doesn't exist."""
    env_file = Path(__file__).parent / ".env"
    env_template = Path(__file__).parent / ".env.template"
    
    if not env_file.exists():
        print("📝 Creating sample environment file...")
        try:
            create_env_template(str(env_template))
            print(f"✅ Environment template created at {env_template}")
            print(f"💡 Copy {env_template} to .env and configure your settings")
        except Exception as e:
            print(f"❌ Failed to create environment template: {e}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="AI MongoDB Query Agent Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                    # Run with default settings
  python run_app.py --port 8080        # Run on custom port
  python run_app.py --check-only       # Only check dependencies
  python run_app.py --install-deps     # Install Python dependencies
  python run_app.py --create-env       # Create environment template
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run the Streamlit application (default: 8501)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies without launching the application"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies and exit"
    )
    
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create environment template file and exit"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency checks and launch directly"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print_banner()
    
    # Handle special commands
    if args.install_deps:
        success = install_python_dependencies()
        sys.exit(0 if success else 1)
    
    if args.create_env:
        create_sample_env_file()
        sys.exit(0)
    
    try:
        # Load settings
        print("⚙️ Loading configuration...")
        settings = get_settings()
        
        if settings.is_development():
            print("🔧 Running in development mode")
        
        # Create sample env file if needed
        create_sample_env_file()
        
        # Check dependencies unless skipped
        if not args.skip_checks:
            print("🔍 Checking dependencies...")
            checker = DependencyChecker(settings)
            results = checker.check_all_dependencies()
            
            all_dependencies_ok = print_dependency_status(results)
            
            if args.check_only:
                sys.exit(0 if all_dependencies_ok else 1)
            
            if not all_dependencies_ok:
                print("\n💡 Suggestions:")
                
                # Check specific failures and provide suggestions
                python_ok, python_msg = results.get('python', (True, ''))
                if not python_ok:
                    print("   • Run: python run_app.py --install-deps")
                
                mongodb_ok, mongodb_msg = results.get('mongodb', (True, ''))
                if not mongodb_ok:
                    print("   • Make sure MongoDB is running and accessible")
                    print("   • Check your MongoDB configuration in .env file")
                
                ollama_ok, ollama_msg = results.get('ollama', (True, ''))
                if not ollama_ok:
                    print("   • Install and start Ollama server: https://ollama.ai/")
                    print("   • Pull the required model: ollama pull llama2")
                
                print("\n❓ Do you want to continue anyway? (y/N): ", end="")
                response = input().strip().lower()
                if response not in ['y', 'yes']:
                    print("👋 Exiting...")
                    sys.exit(1)
        
        # Setup environment and launch application
        launcher = AppLauncher(settings)
        
        if launcher.setup_environment():
            launcher.launch_streamlit(port=args.port, host=args.host)
        else:
            print("❌ Failed to setup environment")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n👋 Application startup cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        print(f"❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
