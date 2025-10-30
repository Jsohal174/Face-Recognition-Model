#!/usr/bin/env python3
"""
Setup Database - Add all people from images folder
"""

import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the access system
from face_access_system import FaceAccessSystem

def setup_database():
    """Add all people from images folder to database."""
    print("\n" + "="*70)
    print(" "*18 + "🗄️  DATABASE SETUP")
    print("="*70)

    # Initialize system
    system = FaceAccessSystem()

    # Find all images
    images_dir = Path('images')
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))

    if not image_files:
        print("\n❌ No images found in images/ directory")
        return

    print(f"\n📁 Found {len(image_files)} images:")
    for img in image_files:
        print(f"   • {img.stem} ({img.name})")

    print("\n" + "─"*70)
    print("Adding all people to database...")
    print("─"*70)

    success_count = 0
    for img_path in image_files:
        name = img_path.stem
        print(f"\n🔄 Processing: {name}")

        if system.add_person(name, str(img_path), verbose=False):
            print(f"   ✓ Added successfully")
            success_count += 1
        else:
            print(f"   ✗ Failed to add")

    print("\n" + "="*70)
    print(f"✅ Database Setup Complete!")
    print(f"   Added: {success_count}/{len(image_files)} people")
    print("="*70)

    # Show final database
    system.list_people()


if __name__ == "__main__":
    setup_database()
