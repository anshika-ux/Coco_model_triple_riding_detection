# Triple Riding Detection System

A comprehensive computer vision system for detecting triple riding violations (three or more people on a motorcycle) using YOLOv8 object detection. The system processes both images and videos, generates detailed reports, and stores results in MongoDB for persistent storage and analysis.

## 🚀 Features

- **Dual Detection Modes**: Process individual images or video files with frame-by-frame analysis
- **Advanced Detection Algorithm**: Intelligent rider-vehicle grouping using IoU and distance metrics
- **Violation Classification**: Automatic identification of triple riding violations with confidence scoring
- **MongoDB Integration**: Persistent storage of detection results with separate collections for images and videos
- **Rich Annotations**: Visual markers, confidence scores, and violation indicators on output images
- **Comprehensive Reporting**: CSV reports with detailed statistics and MongoDB queries
- **Flexible Configuration**: Extensive command-line options for fine-tuning detection parameters

## 📋 Prerequisites

- **Python 3.8+**
- **MongoDB** (local installation or cloud instance)
- **MongoDB Compass** (optional, for database visualization)

### System Dependencies

```bash
# macOS (with Homebrew)
brew install mongodb/brew/mongodb-community

# Ubuntu/Debian
sudo apt-get install mongodb

# Or use Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd triple-riding-detection
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start MongoDB Service
```bash
# macOS with Homebrew
brew services start mongodb/brew/mongodb-community

# Linux systemd
sudo systemctl start mongod

# Or manual start
mongod --dbpath /usr/local/var/mongodb --logpath /usr/local/var/log/mongodb/mongo.log --fork
```

## 📖 Usage

### Image Detection

Process a folder of images and store results in MongoDB:

```bash
python detect_triple_riding.py \
  --images datasets/triple_riding/images \
  --weights yolov8n.pt \
  --mongodb-uri mongodb://localhost:27017/ \
  --db-name triple_riding_db \
  --collection-name detections \
  --conf 0.3 \
  --dist-thresh 0.75 \
  --compact-violation \
  --show-confidence \
  --show-arrow \
  --quiet
```

### Video Detection

Process a video file with frame-by-frame analysis:

```bash
python detect_video_triple_riding.py \
  --file datasets/triple_riding/video/test.mp4 \
  --weights yolov8n.pt \
  --mongodb-uri mongodb://localhost:27017/ \
  --db-name triple_riding_db \
  --collection-name video_detections \
  --conf 0.3 \
  --dist-thresh 0.75 \
  --frame-interval 60 \
  --compact-violation \
  --show-confidence \
  --show-arrow \
  --quiet
```

### Quick Start Scripts

Use the automated setup script:

```bash
# Full setup and run
./scripts/run_all.sh

# Skip installation if already done
./scripts/run_all.sh --no-install

# Pass custom arguments
./scripts/run_all.sh -- --conf 0.5 --images custom/images/
```

## 🗄️ Database Integration

The system uses MongoDB for persistent storage with separate collections:

### Collections Structure

- **`detections`**: Image detection results
- **`video_detections`**: Video frame detection results

### Document Schema

**Image Document:**
```json
{
  "name": "image1.jpg",
  "type": "image",
  "total_vehicles": 2,
  "total_riders": 3,
  "violations": 1,
  "status": "VIOLATION",
  "image_path": "/path/to/image1.jpg",
  "output_path": "/path/to/output/image1.jpg",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "processed_at": "2024-01-15T10:30:00.000Z"
}
```

**Video Frame Document:**
```json
{
  "name": "frame_000060",
  "type": "video_frame",
  "frame_number": 60,
  "total_vehicles": 1,
  "total_riders": 2,
  "violations": 0,
  "status": "COMPLIANT",
  "frame_path": "/path/to/frame_000060.jpg",
  "video_source": "datasets/triple_riding/video/test.mp4",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "processed_at": "2024-01-15T10:30:00.000Z"
}
```

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--conf` | 0.3 | Detection confidence threshold |
| `--iou-thresh` | 0.1 | IoU threshold for rider-vehicle grouping |
| `--dist-thresh` | 0.75 | Distance threshold for rider association |
| `--frame-interval` | 30 | Process every Nth frame (video only) |

## 📊 Reports and Analytics

### CSV Reports
Generated automatically in output directories with columns:
- Image/Video identifier
- Total vehicles detected
- Total riders detected
- Number of violations
- Status (VIOLATION/COMPLIANT)

## 🔧 Development

### Running Tests
```bash
# Test image detection
python detect_triple_riding.py --images datasets/triple_riding/images --quiet

# Test video detection
python detect_video_triple_riding.py --file datasets/triple_riding/video/test.mp4 --quiet
```

### Code Structure
- `detect_triple_riding.py`: Image detection pipeline
- `detect_video_triple_riding.py`: Video detection pipeline
- `database.py`: MongoDB integration utilities
- `scripts/run_all.sh`: Automated setup script

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is provided as-is under the MIT License. See LICENSE file for details.

## 📞 Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check the MongoDB connection and ensure the service is running
- Verify file paths and permissions

---

**Note**: The system uses COCO class indices (person=0, motorbike=3) by default. For custom-trained models with different class mappings, update the class indices in the detection scripts.
