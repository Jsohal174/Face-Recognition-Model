# Face Recognition System

A production-ready face recognition system using FaceNet deep learning model. This system can identify people from face images without requiring ID cards - just show your face!

## Features

- **Face Recognition (1:K Matching)**: Identify a person from a database of K people
- **Face Verification (1:1 Matching)**: Verify if a person matches their claimed identity
- **One-Shot Learning**: Recognizes people from just a single reference image
- **Real-time Processing**: Fast inference using pre-trained FaceNet model
- **Easy Database Management**: Simple CLI tools to add/remove people
- **Visual Results**: Saves annotated images with recognition results

## How It Works

The system uses **FaceNet**, a deep learning model that:
1. Takes a 160×160 face image as input
2. Converts it to a 128-dimensional encoding vector
3. Compares encodings using L2 distance
4. Identifies the person if distance < threshold (0.7)

**Key Concepts**:
- **Encoding**: A 128-D vector representation of a face
- **Distance**: How different two face encodings are (lower = more similar)
- **Threshold**: Maximum distance to consider two faces as the same person

## Architecture

```
FaceNet (Inception-based)
├── Input:  160×160×3 RGB images
├── Layers: 106 layers (Inception blocks)
└── Output: 128-dimensional encodings

Face Recognition Pipeline:
1. Load image → 2. Preprocess → 3. Encode → 4. Compare → 5. Identify
```

## Installation

### 1. Clone or Download

```bash
cd face-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download FaceNet Model (~90 MB)

```bash
python download_facenet.py
```

This will guide you through downloading the pre-trained model files.

**Alternative**: If you have the model from course materials:
- Place `model.json` in `model_data/`
- Place `model.h5` in `model_data/`

## Quick Start

### Step 1: Add People to Database

**Interactive Mode** (Easiest):
```bash
python add_person.py --interactive
```

**Single Person**:
```bash
python add_person.py --name "John Doe" --image images/john.jpg
```

**Multiple People** (from directory):
```bash
python add_person.py --directory images/people/
```

### Step 2: Recognize Faces

**Recognize anyone** (1:K matching):
```bash
python recognize.py --image images/test.jpg
```

**Verify specific person** (1:1 matching):
```bash
python recognize.py --image images/test.jpg --mode verify --name "John Doe"
```

**Show top 5 matches**:
```bash
python recognize.py --image images/test.jpg --top-k 5
```

## Usage Examples

### Example 1: Office Building Access Control

```bash
# Add employees to database
python add_person.py --name "Alice" --image images/alice.jpg
python add_person.py --name "Bob" --image images/bob.jpg

# Recognize employee at entrance (no ID card needed!)
python recognize.py --image camera_capture.jpg
# Output: ✓ Recognized: Alice
#         Distance: 0.45 (threshold: 0.7)
```

### Example 2: Airport Security

```bash
# Verify passport holder matches their photo
python recognize.py --image camera.jpg --mode verify --name "Alice"
# Output: ✓ It's Alice, welcome!
#         Distance: 0.52 (threshold: 0.7)
```

### Example 3: Photo Tagging

```bash
# Find who's in the photo
python recognize.py --image party_photo.jpg --top-k 3
# Output: Top 3 matches:
#         1. ✓ Alice     Distance: 0.45
#         2. ✓ Bob       Distance: 0.62
#         3. ✗ Charlie   Distance: 0.85
```

## Project Structure

```
face-recognition/
├── src/
│   ├── __init__.py
│   ├── face_encoder.py       # Converts images to 128-D encodings
│   ├── face_database.py      # Manages database of face encodings
│   ├── face_verifier.py      # Face verification (1:1)
│   └── face_recognizer.py    # Face recognition (1:K)
│
├── model_data/               # FaceNet model files
│   ├── model.json           # Model architecture (~100 KB)
│   └── model.h5             # Model weights (~90 MB)
│
├── database/                 # Face encodings database
│   └── face_database.json   # Stores 128-D encodings per person
│
├── images/                   # Input images
├── output/                   # Recognition results with visualizations
│
├── add_person.py            # Add people to database
├── recognize.py             # Main recognition script
├── download_facenet.py      # Download model files
├── config.yaml              # Configuration settings
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Technical Details

### FaceNet Model

- **Architecture**: Inception-based CNN
- **Training Method**: Triplet Loss
  - Anchor (A): Your face
  - Positive (P): Another photo of you
  - Negative (N): Someone else's face
  - Goal: Make ||f(A) - f(P)||² + α < ||f(A) - f(N)||²

- **Papers**:
  - FaceNet (Schroff et al., 2015)
  - DeepFace (Taigman et al., 2014)

### Distance Threshold

The threshold (default: 0.7) controls strictness:

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.5 | Very strict | High security (few false positives) |
| 0.7 | Balanced | General use (recommended) |
| 1.0 | Lenient | More matches (more false positives) |

**Typical Distances**:
- Same person: 0.0 - 0.6
- Different people: 0.8 - 1.2

### Face Verification vs Recognition

| Feature | Verification (1:1) | Recognition (1:K) |
|---------|-------------------|-------------------|
| Input | Image + Name | Image only |
| Output | Match/No Match | Identity or Unknown |
| Use Case | Access with ID card | Access without ID |
| Accuracy Req | ~99% | Depends on K |
| Example | Phone unlock | Building entry |

## Advanced Usage

### Custom Threshold

```bash
# More strict (fewer false positives)
python recognize.py --image test.jpg --threshold 0.5

# More lenient (fewer false negatives)
python recognize.py --image test.jpg --threshold 0.9
```

### Batch Processing

```python
from src.face_encoder import FaceEncoder
from src.face_database import FaceDatabase
from src.face_recognizer import FaceRecognizer

# Initialize
encoder = FaceEncoder()
encoder.load_model()
database = FaceDatabase()
recognizer = FaceRecognizer(encoder, database)

# Recognize multiple images
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = recognizer.recognize_batch(images)

for img, (distance, identity) in zip(images, results):
    print(f"{img}: {identity} (distance: {distance:.3f})")
```

### Database Management

**List all people**:
```python
from src.face_database import FaceDatabase

db = FaceDatabase()
db.list_all()
```

**Remove person**:
```python
db.remove_person("John Doe")
db.save()
```

**Clear database**:
```python
db.clear()
```

## Configuration

Edit `config.yaml` to customize:

```yaml
recognition:
  threshold: 0.7        # Recognition threshold
  top_k: 5             # Number of top matches

visualization:
  output_dir: "output" # Where to save results
  colors:
    recognized: [0, 255, 0]     # Green
    not_recognized: [0, 0, 255] # Red
```

## Troubleshooting

### Issue: Model files not found

```
❌ Error: Model files not found in model_data/
```

**Solution**: Run `python download_facenet.py` to download the model, or manually place `model.json` and `model.h5` in `model_data/`.

### Issue: Empty database

```
⚠️  Warning: Database is empty!
```

**Solution**: Add people using `python add_person.py --interactive`

### Issue: Person not recognized

```
❓ Person not recognized (closest: Alice, distance: 0.85)
```

**Solutions**:
1. Increase threshold: `--threshold 0.9`
2. Add more photos of the person to database
3. Ensure good lighting and face is clearly visible
4. Face should be front-facing (not profile view)

### Issue: Too many false positives

**Solution**: Decrease threshold: `--threshold 0.5`

## Performance Tips

1. **Image Quality**: Use clear, well-lit face images
2. **Face Size**: Faces should be prominent in the image
3. **Multiple Photos**: Add 2-3 photos per person (different angles/lighting)
4. **Preprocessing**: Crop images to focus on faces (reduces background noise)
5. **Database Size**: With 100+ people, consider using approximate nearest neighbors

## API Reference

### FaceEncoder

```python
encoder = FaceEncoder(model_dir='model_data')
encoder.load_model()

# Encode a single image
encoding = encoder.img_to_encoding('image.jpg')  # Shape: (1, 128)

# Encode multiple images
encodings = encoder.encode_batch(['img1.jpg', 'img2.jpg'])
```

### FaceDatabase

```python
database = FaceDatabase('database/face_database.json')

# Add person
database.add_person('Alice', encoding, 'alice.jpg')
database.save()

# Get encoding
alice_encoding = database.get_encoding('Alice')

# List all
database.list_all()

# Remove person
database.remove_person('Alice')
```

### FaceRecognizer

```python
recognizer = FaceRecognizer(encoder, database, threshold=0.7)

# Recognize
distance, identity = recognizer.recognize('test.jpg')

# Top K matches
matches = recognizer.recognize_top_k('test.jpg', k=5)

# With confidence score
result = recognizer.recognize_with_confidence('test.jpg')
# Returns: {'identity': 'Alice', 'distance': 0.45,
#           'confidence': 0.85, 'is_recognized': True}
```

### FaceVerifier

```python
verifier = FaceVerifier(encoder, database, threshold=0.7)

# Verify
distance, is_match = verifier.verify('test.jpg', 'Alice')

# Batch verify
results = verifier.verify_batch(image_paths, identities)
```

## Real-World Applications

1. **Access Control**: Office buildings, secure facilities
2. **Time Attendance**: Automatic clock-in/out systems
3. **Customer Recognition**: VIP customer identification in retail
4. **Security**: Surveillance systems, watchlists
5. **Photo Organization**: Automatic photo tagging and grouping
6. **Border Control**: Airport immigration and customs

## Limitations

- Requires one reference photo per person (one-shot learning)
- Performance degrades with very large databases (1000+ people)
- Sensitive to lighting conditions and face angles
- Cannot detect liveness (can be fooled by photos)
- Privacy concerns with biometric data storage

## Future Improvements

- [ ] Add liveness detection to prevent photo spoofing
- [ ] Implement face detection to auto-crop faces from images
- [ ] Support for video stream processing
- [ ] GPU acceleration for faster processing
- [ ] Web interface for easier management
- [ ] Multiple photos per person for better accuracy
- [ ] Export/import database in different formats

## License

MIT License - Feel free to use for educational and commercial purposes.

## References

1. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A unified embedding for face recognition and clustering*. CVPR.

2. Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). *DeepFace: Closing the gap to human-level performance in face verification*. CVPR.

3. Keras FaceNet Implementation: https://github.com/nyoki-mtl/keras-facenet

4. Original FaceNet: https://github.com/davidsandberg/facenet

## Support

For issues and questions:
- Check the Troubleshooting section above
- Review configuration in `config.yaml`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Acknowledgments

- FaceNet model by Google Research
- Keras implementation by nyoki-mtl
- Course materials from Deep Learning Specialization

---

**Ready to try it out?**

```bash
# 1. Download model
python download_facenet.py

# 2. Add yourself
python add_person.py --interactive

# 3. Test recognition
python recognize.py --image images/your_photo.jpg
```

**Enjoy building your face recognition system!** 🎉
