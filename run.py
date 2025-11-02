#!/usr/bin/env python3
"""
Main entry point for the Medical Document Assistant application.

This module imports and runs the Flask app with proper configuration.
"""

import os
import sys
from app import app

def main():
    """Main entry point for the application."""
    print("🏥 Starting Medical Document Assistant...")
    print("📋 Optimized for accurate medical document analysis")
    print("🔍 Enhanced RAG pipeline with intelligent query classification")
    print("=" * 60)
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5006))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"🌐 Server starting at: http://{host}:{port}")
    print(f"🔧 Debug mode: {'Enabled' if debug else 'Disabled'}")
    print("=" * 60)
    
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Medical Document Assistant shutting down...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
