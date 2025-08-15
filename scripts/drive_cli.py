#!/usr/bin/env python3
"""
DriveToQuizlet - Google Drive Automation CLI
Command-line interface for managing Google Drive file automation
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from drive_automation import DriveAutomation

def print_banner():
    """Print application banner"""
    print("""
🚀 DriveToQuizlet - Google Drive Automation
===========================================
Automatically download and organize military training files from Google Drive
""")

def test_connection():
    """Test Google Drive connection"""
    print("🔍 Testing Google Drive connection...")
    try:
        automation = DriveAutomation()
        print("✅ Google Drive connection successful!")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("\n💡 Make sure you have:")
        print("   1. Run 'python scripts/drive_test.py' to authenticate")
        print("   2. Created 'config/token.json' file")
        print("   3. Set up 'config/drive_config.json' with folder IDs")
        return False

def scan_once():
    """Perform a single scan and download"""
    print("🔄 Performing single scan and download...")
    try:
        automation = DriveAutomation()
        results = automation.scan_and_download()
        
        print(f"\n📊 Scan Results:")
        print(f"   Total files found: {results['new_files']}")
        print(f"   New files to download: {results.get('new_downloads', results['new_files'])}")
        print(f"   Files downloaded: {results['downloaded']}")
        print(f"   Failed downloads: {results['failed']}")
        
        if results['errors']:
            print(f"\n❌ Errors:")
            for error in results['errors']:
                print(f"   - {error}")
        
        return True
    except Exception as e:
        print(f"❌ Scan failed: {str(e)}")
        return False

def show_queue():
    """Show download queue"""
    try:
        automation = DriveAutomation()
        queue = automation.get_download_queue()
        
        if not queue:
            print("📭 Download queue is empty")
            return
        
        print(f"\n📋 Download Queue ({len(queue)} files):")
        print("-" * 80)
        for i, file_info in enumerate(queue, 1):
            print(f"{i:2d}. {file_info['name']}")
            print(f"    📁 Lesson: {file_info['lesson']}")
            print(f"    📂 Type: {file_info['type']}")
            print(f"    📅 Queued: {file_info['queued_date']}")
            print(f"    📊 Status: {file_info['status']}")
            if file_info['error_message']:
                print(f"    ❌ Error: {file_info['error_message']}")
            print()
        
    except Exception as e:
        print(f"❌ Error showing queue: {str(e)}")

def show_downloaded():
    """Show downloaded files"""
    try:
        automation = DriveAutomation()
        files = automation.get_downloaded_files()
        
        if not files:
            print("📭 No files have been downloaded yet")
            return
        
        print(f"\n📥 Downloaded Files ({len(files)} files):")
        print("-" * 80)
        for i, file_info in enumerate(files, 1):
            size_mb = file_info['file_size'] / 1024 / 1024 if file_info['file_size'] else 0
            print(f"{i:2d}. {file_info['name']}")
            print(f"    📁 Lesson: {file_info['lesson']}")
            print(f"    📂 Type: {file_info['type']}")
            print(f"    📏 Size: {size_mb:.1f} MB")
            print(f"    📅 Downloaded: {file_info['download_date']}")
            print(f"    📍 Path: {file_info['local_path']}")
            print()
        
    except Exception as e:
        print(f"❌ Error showing downloaded files: {str(e)}")

def process_queue():
    """Process download queue manually with approval prompts"""
    print("🚀 Processing download queue...")
    try:
        automation = DriveAutomation()
        
        # Get current configuration
        config_path = Path("config/drive_config.json")
        if not config_path.exists():
            print("❌ Configuration file not found")
            return False
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Check if approval is required
        require_approval = config['processing'].get('require_approval', True)
        
        # Get files in queue
        queued_files = automation.get_download_queue()
        queued_files = [f for f in queued_files if f['status'] == 'queued']
        
        if not queued_files:
            print("📭 No files in download queue")
            return True
        
        print(f"\n📋 Found {len(queued_files)} files in download queue:")
        print("-" * 80)
        
        # Show files that will be downloaded
        for i, file_info in enumerate(queued_files, 1):
            print(f"{i:2d}. {file_info['name']}")
            print(f"    📁 Lesson: {file_info['lesson']}")
            print(f"    📂 Type: {file_info['type']}")
            print(f"    📅 Queued: {file_info['queued_date']}")
            print()
        
        # If approval is required, ask for confirmation
        if require_approval:
            print("⚠️  Approval Required: Files will be downloaded to your local lessons folder")
            print("   This will create lesson folders and organize files automatically.")
            print()
            
            while True:
                response = input("❓ Do you want to download these files? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    print("✅ Proceeding with download...")
                    break
                elif response in ['n', 'no']:
                    print("❌ Download cancelled by user")
                    return False
                else:
                    print("❓ Please enter 'y' for yes or 'n' for no")
        else:
            print("✅ Auto-approval enabled, proceeding with download...")
        
        # Process the download queue
        results = automation.process_download_queue()
        
        print(f"\n📊 Processing Results:")
        print(f"   Successful: {results['success']}")
        print(f"   Failed: {results['failed']}")
        
        if results['errors']:
            print(f"\n❌ Errors:")
            for error in results['errors']:
                print(f"   - {error}")
        
        return True
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        return False

def show_config():
    """Show current configuration"""
    try:
        config_path = Path("config/drive_config.json")
        if not config_path.exists():
            print("❌ Configuration file not found")
            return
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print("\n⚙️  Current Configuration:")
        print("-" * 40)
        print(f"📁 Personal Notes/Audio Folder: {config['drive_folders']['personal_notes_audio']}")
        print(f"📁 Instructor Presentations Folder: {config['drive_folders']['instructor_presentations']}")
        print(f"⏰ Check Interval: {config['processing']['check_interval_hours']} hour(s)")
        print(f"📥 Auto Download: {config['processing']['auto_download']}")
        print(f"🔄 Auto Process: {config['processing']['auto_process']}")
        print(f"✅ Require Approval: {config['processing']['require_approval']}")
        
        print(f"\n📄 Supported File Types:")
        for category, mime_types in config['file_types'].items():
            print(f"   {category}: {', '.join(mime_types)}")
        
    except Exception as e:
        print(f"❌ Error showing config: {str(e)}")

def approve_files():
    """Interactive file approval - approve files one by one"""
    print("✅ Interactive File Approval")
    print("=" * 50)
    
    try:
        automation = DriveAutomation()
        
        # Get files in queue
        queued_files = automation.get_download_queue()
        queued_files = [f for f in queued_files if f['status'] == 'queued']
        
        if not queued_files:
            print("📭 No files in download queue")
            return True
        
        print(f"📋 Found {len(queued_files)} files pending approval:")
        print()
        
        approved_files = []
        
        for i, file_info in enumerate(queued_files, 1):
            print(f"📄 File {i}/{len(queued_files)}:")
            print(f"   📝 Name: {file_info['name']}")
            print(f"   📁 Lesson: {file_info['lesson']}")
            print(f"   📂 Type: {file_info['type']}")
            print(f"   📅 Queued: {file_info['queued_date']}")
            print()
            
            while True:
                response = input(f"❓ Approve this file for download? (y/n/s for skip): ").lower().strip()
                if response in ['y', 'yes']:
                    approved_files.append(file_info)
                    print("✅ Approved")
                    break
                elif response in ['n', 'no']:
                    print("❌ Rejected")
                    break
                elif response in ['s', 'skip']:
                    print("⏭️  Skipped")
                    break
                else:
                    print("❓ Please enter 'y' for yes, 'n' for no, or 's' for skip")
            print()
        
        if not approved_files:
            print("📭 No files approved for download")
            return True
        
        print(f"🚀 Downloading {len(approved_files)} approved files...")
        
        # Download approved files
        success_count = 0
        failed_count = 0
        errors = []
        
        for file_info in approved_files:
            success, error = automation.download_file(file_info)
            if success:
                success_count += 1
                print(f"✅ Downloaded: {file_info['name']}")
            else:
                failed_count += 1
                errors.append(f"{file_info['name']}: {error}")
                print(f"❌ Failed: {file_info['name']} - {error}")
        
        print(f"\n📊 Download Summary:")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {failed_count}")
        
        if errors:
            print(f"\n❌ Errors:")
            for error in errors:
                print(f"   - {error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Approval process failed: {str(e)}")
        return False

def start_service():
    """Start the automation service"""
    print("🚀 Starting DriveToQuizlet automation service...")
    print("Press Ctrl+C to stop")
    
    try:
        automation = DriveAutomation()
        automation.start_automation()
    except KeyboardInterrupt:
        print("\n⏹️  Service stopped by user")
    except Exception as e:
        print(f"❌ Service error: {str(e)}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="DriveToQuizlet - Google Drive Automation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drive_cli.py test              # Test Google Drive connection
  python drive_cli.py scan              # Perform single scan
  python drive_cli.py queue             # Show download queue
  python drive_cli.py process           # Process download queue (with approval prompt)
  python drive_cli.py approve           # Interactive file approval (one by one)
  python drive_cli.py downloaded        # Show downloaded files
  python drive_cli.py config            # Show configuration
  python drive_cli.py start             # Start automation service
        """
    )
    
    parser.add_argument('command', choices=[
        'test', 'scan', 'queue', 'process', 'approve', 'downloaded', 'config', 'start'
    ], help='Command to execute')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'test':
        test_connection()
    elif args.command == 'scan':
        scan_once()
    elif args.command == 'queue':
        show_queue()
    elif args.command == 'process':
        process_queue()
    elif args.command == 'approve':
        approve_files()
    elif args.command == 'downloaded':
        show_downloaded()
    elif args.command == 'config':
        show_config()
    elif args.command == 'start':
        start_service()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
