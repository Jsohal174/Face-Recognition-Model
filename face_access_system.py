#!/usr/bin/env python3
"""
Face Recognition Access Control System
--------------------------------------
1. Recognize faces and grant/deny access
2. Add new people to the database
"""

import os
import sys
from pathlib import Path
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from deepface import DeepFace
    import numpy as np
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    print("❌ Error: DeepFace not installed")
    print("Run: ./setup_and_run.sh")
    sys.exit(1)


class FaceAccessSystem:
    """Face recognition system for access control."""

    def __init__(self, database_path='database/face_database.json', threshold=10.0):
        """
        Initialize the access control system.

        Args:
            database_path: Path to database file
            threshold: Distance threshold for recognition (default: 10.0)
        """
        self.database_path = Path(database_path)
        self.threshold = threshold
        self.database = {}
        self.load_database()

    def load_database(self):
        """Load face database from file."""
        if self.database_path.exists():
            with open(self.database_path, 'r') as f:
                data = json.load(f)
                self.database = {name: np.array(encoding)
                               for name, encoding in data.items()}
            print(f"✓ Loaded database with {len(self.database)} people")
        else:
            print("ℹ️  No existing database found. Starting fresh.")
            self.database = {}

    def save_database(self):
        """Save face database to file."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON
        db_serializable = {name: enc.tolist() for name, enc in self.database.items()}

        with open(self.database_path, 'w') as f:
            json.dump(db_serializable, f, indent=2)

        print(f"✓ Database saved ({len(self.database)} people)")

    def generate_encoding(self, image_path):
        """
        Generate face encoding for an image using FaceNet.

        Args:
            image_path: Path to face image

        Returns:
            numpy array: 128-dimensional encoding, or None if failed
        """
        try:
            embedding_objs = DeepFace.represent(
                img_path=str(image_path),
                model_name="Facenet",
                enforce_detection=False
            )

            if embedding_objs and len(embedding_objs) > 0:
                return np.array(embedding_objs[0]['embedding'])
            else:
                return None

        except Exception as e:
            print(f"❌ Error generating encoding: {e}")
            return None

    def recognize(self, image_path, verbose=True):
        """
        Recognize a person from their face and grant/deny access.

        Args:
            image_path: Path to face image
            verbose: Whether to print detailed results

        Returns:
            tuple: (identity, distance, access_granted)
        """
        if verbose:
            print("\n" + "="*70)
            print(" "*20 + "🔒 ACCESS CONTROL SYSTEM")
            print("="*70)

        # Check if database is empty
        if not self.database:
            if verbose:
                print("\n❌ ACCESS DENIED")
                print("   Reason: Database is empty (no authorized personnel)")
            return None, None, False

        if verbose:
            print(f"\n📷 Processing image: {Path(image_path).name}")
            print(f"👥 Checking against {len(self.database)} authorized personnel...")

        # Generate encoding for input image
        if verbose:
            print("\n🔄 Generating face encoding...")

        test_encoding = self.generate_encoding(image_path)

        if test_encoding is None:
            if verbose:
                print("\n❌ ACCESS DENIED")
                print("   Reason: Could not detect face in image")
            return None, None, False

        if verbose:
            print("✓ Encoding generated (128-dimensional vector)")

        # Compare with all people in database
        if verbose:
            print("\n🔍 Comparing with authorized personnel:")

        min_distance = float('inf')
        best_match = None

        for name, db_encoding in self.database.items():
            distance = np.linalg.norm(test_encoding - db_encoding)

            if verbose:
                print(f"   {name:<20} Distance: {distance:.4f}")

            if distance < min_distance:
                min_distance = distance
                best_match = name

        # Determine access
        access_granted = min_distance < self.threshold

        if verbose:
            print("\n" + "─"*70)
            print(f"Best match: {best_match}")
            print(f"Distance:   {min_distance:.4f}")
            print(f"Threshold:  {self.threshold}")
            print("─"*70)

        # Display result
        if access_granted:
            if verbose:
                print("\n✅ ACCESS GRANTED")
                print(f"   Welcome, {best_match}!")
        else:
            if verbose:
                print("\n❌ ACCESS DENIED")
                if best_match:
                    print(f"   Closest match: {best_match} (distance: {min_distance:.4f})")
                print(f"   Reason: No match found (all distances > {self.threshold})")

        if verbose:
            print("="*70 + "\n")

        return best_match, min_distance, access_granted

    def add_person(self, name, image_path, verbose=True):
        """
        Add a new person to the database.

        Args:
            name: Person's name
            image_path: Path to their face image
            verbose: Whether to print progress

        Returns:
            bool: True if added successfully
        """
        if verbose:
            print("\n" + "="*70)
            print(" "*20 + "➕ ADD NEW PERSON")
            print("="*70)
            print(f"\n👤 Name:  {name}")
            print(f"📷 Image: {image_path}")

        # Check if person already exists
        if name in self.database:
            if verbose:
                print(f"\n⚠️  Warning: '{name}' already exists in database")
                response = input("   Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print("✗ Cancelled")
                    return False

        # Generate encoding
        if verbose:
            print("\n🔄 Generating face encoding...")

        encoding = self.generate_encoding(image_path)

        if encoding is None:
            if verbose:
                print("\n❌ Failed to add person")
                print("   Reason: Could not detect face in image")
            return False

        # Add to database
        self.database[name] = encoding

        if verbose:
            print(f"✓ Encoding generated (128-dimensional vector)")
            print(f"   Sample: [{encoding[0]:.4f}, {encoding[1]:.4f}, {encoding[2]:.4f}, ...]")

        # Save database
        self.save_database()

        if verbose:
            print(f"\n✅ Successfully added '{name}' to database!")
            print(f"   Total authorized personnel: {len(self.database)}")
            print("="*70 + "\n")

        return True

    def list_people(self):
        """List all people in the database."""
        print("\n" + "="*70)
        print(" "*20 + "📋 AUTHORIZED PERSONNEL")
        print("="*70)

        if not self.database:
            print("\n   (empty)")
        else:
            for i, name in enumerate(self.database.keys(), 1):
                print(f"   {i}. {name}")

        print(f"\n   Total: {len(self.database)} people")
        print("="*70 + "\n")


def main():
    """Main function with interactive menu."""
    print("\n" + "="*70)
    print(" "*15 + "🤖 FACE RECOGNITION ACCESS CONTROL")
    print("="*70)

    # Initialize system
    system = FaceAccessSystem()

    while True:
        print("\n" + "─"*70)
        print("Choose an option:")
        print("─"*70)
        print("  1. Recognize face (check access)")
        print("  2. Add new person")
        print("  3. List all people")
        print("  4. Exit")
        print("─"*70)

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            # Recognize face
            image_path = input("\n📷 Enter path to face image: ").strip()

            if not Path(image_path).exists():
                print(f"❌ Error: File not found: {image_path}")
                continue

            identity, distance, access = system.recognize(image_path)

        elif choice == '2':
            # Add new person
            name = input("\n👤 Enter person's name: ").strip()

            if not name:
                print("❌ Error: Name cannot be empty")
                continue

            image_path = input("📷 Enter path to face image: ").strip()

            if not Path(image_path).exists():
                print(f"❌ Error: File not found: {image_path}")
                continue

            system.add_person(name, image_path)

        elif choice == '3':
            # List people
            system.list_people()

        elif choice == '4':
            # Exit
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Command line mode
        system = FaceAccessSystem()

        if sys.argv[1] == 'recognize' and len(sys.argv) > 2:
            image_path = sys.argv[2]
            identity, distance, access = system.recognize(image_path)
            sys.exit(0 if access else 1)

        elif sys.argv[1] == 'add' and len(sys.argv) > 3:
            name = sys.argv[2]
            image_path = sys.argv[3]
            success = system.add_person(name, image_path)
            sys.exit(0 if success else 1)

        elif sys.argv[1] == 'list':
            system.list_people()
            sys.exit(0)

        else:
            print("Usage:")
            print("  Interactive mode:  python3 face_access_system.py")
            print("  Recognize:         python3 face_access_system.py recognize <image>")
            print("  Add person:        python3 face_access_system.py add <name> <image>")
            print("  List people:       python3 face_access_system.py list")
            sys.exit(1)
    else:
        # Interactive mode
        main()
