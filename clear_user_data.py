#!/usr/bin/env python3
"""
Clear User Data Script for Medical Document Assistant

This script clears user-uploaded files and data while preserving the database structure.
It does NOT delete the database file itself, only the user data within it.
"""

import os
import shutil

def clear_user_data():
    """Clear user data while preserving database structure."""
    
    print("🧹 Starting user data cleanup...")
    print("=" * 50)
    
    # 1. Clear user uploaded files
    user_data_dir = "UserData"
    if os.path.exists(user_data_dir):
        print(f"📁 Clearing user uploaded files from {user_data_dir}...")
        try:
            # Remove all user upload directories but keep the main UserData folder
            for item in os.listdir(user_data_dir):
                item_path = os.path.join(user_data_dir, item)
                if os.path.isdir(item_path) and item.endswith('_Upload'):
                    shutil.rmtree(item_path)
                    print(f"   ✅ Removed: {item}")
                elif os.path.isfile(item_path) and item.endswith('.json'):
                    os.remove(item_path)
                    print(f"   ✅ Removed: {item}")
            print("✅ User uploaded files cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing user files: {e}")
    
    # 2. Clear chroma vector store data
    chroma_dir = "chroma_store"
    if os.path.exists(chroma_dir):
        print(f"🔍 Clearing vector store data from {chroma_dir}...")
        try:
            shutil.rmtree(chroma_dir)
            os.makedirs(chroma_dir, exist_ok=True)
            print("✅ Vector store data cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing vector store: {e}")
    
    # 3. Clear uploads directory
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        print(f"📤 Clearing uploads directory...")
        try:
            for item in os.listdir(uploads_dir):
                item_path = os.path.join(uploads_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"   ✅ Removed: {item}")
            print("✅ Uploads directory cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing uploads: {e}")
    
    print("=" * 50)
    print("🎉 User data cleanup completed!")
    print("\n📋 What was cleared:")
    print("   • User uploaded documents and files")
    print("   • Vector store embeddings")
    print("   • Temporary uploads")
    print("\n🔒 What was preserved:")
    print("   • Database file and structure")
    print("   • User accounts and login history")
    print("   • Application configuration")
    print("\n💡 Note: To clear user accounts from database, use the Flask app admin interface")

def main():
    """Main function to run the cleanup."""
    print("🏥 Medical Document Assistant - User Data Cleanup")
    print("⚠️  This will clear all user data but preserve the database structure.")
    
    # Ask for confirmation
    confirm = input("\nDo you want to proceed? (yes/no): ").lower().strip()
    
    if confirm in ['yes', 'y']:
        clear_user_data()
    else:
        print("❌ Cleanup cancelled.")

if __name__ == "__main__":
    main()
