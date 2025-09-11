"""
Production Launcher for Flower Detection System
Launches all production services: training, web interface, and API server.
"""

import subprocess
import sys
import time
import threading
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    """Run the training pipeline."""
    logger.info("🚀 Starting training pipeline...")
    try:
        # Change to the correct directory
        script_dir = Path(__file__).parent
        subprocess.run([sys.executable, "run_pipeline.py"], cwd=script_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

def run_web_interface():
    """Run the web interface."""
    logger.info("🌐 Starting web interface...")
    try:
        # Change to the correct directory
        script_dir = Path(__file__).parent
        subprocess.run([sys.executable, "-m", "src.verification_ui.web_interface"], cwd=script_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Web interface failed: {e}")
    except KeyboardInterrupt:
        logger.info("Web interface interrupted by user")

def run_api_server():
    """Run the API server."""
    logger.info("🔌 Starting API server...")
    try:
        # Change to the correct directory
        script_dir = Path(__file__).parent
        subprocess.run([sys.executable, "-m", "src.inference.api_server"], cwd=script_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"API server failed: {e}")
    except KeyboardInterrupt:
        logger.info("API server interrupted by user")

def main():
    """Main launcher function."""
    print("🌟 FLOWER DETECTION SYSTEM - PRODUCTION LAUNCHER")
    print("=" * 60)
    print("Choose an option:")
    print("1. Run Training Pipeline")
    print("2. Start Web Interface")
    print("3. Start API Server")
    print("4. Start All Services (training + web + API)")
    print("5. Exit")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                run_training()
                break
            elif choice == "2":
                run_web_interface()
                break
            elif choice == "3":
                run_api_server()
                break
            elif choice == "4":
                print("\n🚀 Starting all services...")
                print("Note: This will start training, web interface, and API server")
                print("Press Ctrl+C to stop all services")
                
                # Start services in separate threads
                training_thread = threading.Thread(target=run_training, daemon=True)
                web_thread = threading.Thread(target=run_web_interface, daemon=True)
                api_thread = threading.Thread(target=run_api_server, daemon=True)
                
                training_thread.start()
                time.sleep(2)  # Wait for training to start
                
                web_thread.start()
                time.sleep(1)  # Wait for web interface to start
                
                api_thread.start()
                
                print("\n✅ All services started!")
                print("🌐 Web Interface: http://localhost:5000")
                print("🔌 API Server: http://localhost:8000")
                print("📊 TensorBoard: http://localhost:6006")
                print("\nPress Ctrl+C to stop all services...")
                
                try:
                    # Keep main thread alive
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Stopping all services...")
                    break
                    
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
