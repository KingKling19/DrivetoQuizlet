#!/usr/bin/env python3
"""
Setup script for Army ADA BOLC Flashcard App
Run this script to initialize the project environment
"""

import os
import shutil
from pathlib import Path

def setup_project():
    """Initialize the project structure and configuration"""
    print("🚀 Setting up Army ADA BOLC Flashcard App...")
    
    # Create required directories
    directories = [
        "config",
        "outputs", 
        "logs",
        "temp",
        "static/css",
        "static/js", 
        "static/images",
        "lessons"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Copy .env template if .env doesn't exist
    if not Path(".env").exists():
        if Path(".env.template").exists():
            shutil.copy(".env.template", ".env")
            print("✓ Created .env file from template")
            print("⚠️  Please edit .env file with your API keys and configuration")
        else:
            print("❌ .env.template not found")
    else:
        print("✓ .env file already exists")
    
    # Check for Google credentials template
    credentials_template = Path("config/credentials.json.template")
    if not credentials_template.exists():
        create_credentials_template()
    
    print("\n📋 Next steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Set up Google Drive API credentials:")
    print("   - Go to Google Cloud Console")
    print("   - Create a new project or select existing")
    print("   - Enable Google Drive API")
    print("   - Create credentials (OAuth 2.0)")
    print("   - Download credentials.json to config/ directory")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run authentication test: python scripts/drive_test.py")
    print("5. Configure Google Drive folder IDs in config/drive_config.json")
    print("\n🎯 Ready to process military training materials!")

def create_credentials_template():
    """Create a template for Google credentials"""
    template_content = '''{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}'''
    
    with open("config/credentials.json.template", "w") as f:
        f.write(template_content)
    print("✓ Created config/credentials.json.template")

if __name__ == "__main__":
    setup_project()