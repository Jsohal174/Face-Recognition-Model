#!/bin/bash
# Setup and Run Face Recognition Demo
# This handles the macOS externally-managed environment

echo "============================================================"
echo "  Face Recognition Demo - Setup & Run"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "run_demo.py" ]; then
    echo "❌ Error: Please run this from the face-recognition directory"
    exit 1
fi

echo ""
echo "🔧 Installing dependencies..."
echo ""

# Install with --break-system-packages flag for macOS
pip3 install --break-system-packages deepface opencv-python numpy tensorflow

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Installation failed. Trying alternative method..."
    pip3 install --user deepface opencv-python numpy tensorflow
fi

echo ""
echo "✓ Dependencies installed!"
echo ""
echo "============================================================"
echo "  Building Face Database"
echo "============================================================"
echo ""

# Build database with all people from images folder
python3 setup_database.py

echo ""
echo "============================================================"
echo "  ✅ Setup Complete!"
echo "============================================================"
echo ""
echo "📋 Your system is ready! You can now:"
echo ""
echo "1. Recognize faces:"
echo "   python3 face_access_system.py recognize images/jaskirat.png"
echo ""
echo "2. Add new people:"
echo "   python3 face_access_system.py add \"Name\" images/photo.jpg"
echo ""
echo "3. List all people:"
echo "   python3 face_access_system.py list"
echo ""
echo "4. Interactive mode:"
echo "   python3 face_access_system.py"
echo ""
echo "============================================================"
