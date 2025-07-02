# Real-Time Object Detection System with User Authentication

A secure web application that performs real-time object detection using your webcam, built with Flask, OpenCV, and YOLOv8. The system includes user authentication, admin dashboard, and advanced detection features.

## Features

- **User Authentication & Management**
  - Secure login and registration system
  - Role-based access control (Admin/Regular users)
  - Session management and tracking
  - Admin dashboard for user management

- **Real-time Object Detection**
  - Webcam-based real-time detection with optimized YOLOv8m model
  - Robust camera initialization with multiple backend fallbacks
  - Frame rate optimization (processes every 2nd frame)
  - Detection confidence threshold optimization (0.3)
  - Person detection filtering with size-based exclusion
  - Real-time detection summary display with FPS and detection count
  - Error handling with graceful fallback frames
  - Detection history tracking and analytics

- **Image & Video Processing**
  - Image upload with object detection
  - Video upload with progress tracking
  - Detection history logging
  - Object speed tracking
  - Zone detection and classification

- **Advanced Features**
  - Real-time heatmap visualization with zone tracking
  - Object speed tracking and classification
  - Zone-based detection analytics
  - Detection history logging with timestamps
  - Performance optimization with frame buffering
  - CUDA error handling with fallback to CPU
  - Frame quality optimization with JPEG encoding

- **Database & Security**
  - SQLite database for user information
  - Secure password hashing with bcrypt
  - Session tracking and management
  - Admin dashboard with user activity monitoring
  - Contact form with email notifications
  - Secure file handling for uploads

## Requirements

- Python 3.8+
- Flask (2.0+)
- Flask-Login
- OpenCV (cv2) >= 4.5.0
- Ultralytics YOLO >= 8.0
- SQLite3
- Web browser with JavaScript enabled
- Webcam with DirectShow or MSMF support (Windows)
- CUDA (optional for GPU acceleration)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd object-detection-using-webcam
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   # For GPU acceleration (optional)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

1. Run the application:
   ```
   python Application.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Register a new account or log in if you already have one.
4. After logging in:
   - Use the navigation menu to access different features
   - Start real-time detection for webcam-based object detection
   - Upload images for static object detection
   - Upload videos for batch processing
   - Admin users can access the admin dashboard

## Project Structure

```
├── Application.py       # Main Flask application
├── models.py            # User authentication models
├── database.db          # SQLite database for user information
├── static/              # Static files (CSS, JS)
├── templates/           # HTML templates
│   ├── base.html        # Base template with common elements
│   ├── index.html       # Main application interface
│   ├── login.html       # User login page
│   ├── register.html    # User registration page
│   ├── admin/           # Admin dashboard templates
│   └── detection.html   # Detection interface
├── uploads/             # Temporary storage for uploaded files
├── outputs/             # Processed images and videos
├── yolov8m.pt           # YOLOv8 medium model weights
└── requirements.txt     # Project dependencies
```

## How It Works

- The application uses Flask as the web framework to serve the interface
- Flask-Login handles user authentication and session management
- User credentials are securely stored in a SQLite database with bcrypt hashing
- OpenCV is used to capture video from the webcam and process images/videos
- YOLOv8 models from Ultralytics perform the object detection
- Real-time detection results are streamed back to the browser with overlays and analytics
- Uploaded videos are processed with progress tracking and ETA
- All detection features are protected behind authentication
- Admin dashboard provides user management, monitoring, and analytics capabilities

## Troubleshooting

- If the webcam doesn't start:
  - The application will try multiple camera backends (DirectShow, DSHOW, MSMF)
  - Falls back to default backend if others fail
  - Check if other applications are using the webcam
  - Ensure you have proper camera permissions
- For performance reasons:
  - Real-time detection processes every 2nd frame
  - Frame size is optimized for performance (1280x720)
  - Object detection confidence threshold is balanced (0.3)
  - Uses frame buffering to reduce latency
- Video processing:
  - Provides progress updates and estimated time remaining
  - Optimized for performance with frame buffering
  - Handles CUDA errors gracefully with CPU fallback
  - Uses efficient JPEG encoding for streaming

## License

MIT License

## Acknowledgments

- YOLOv8 by Ultralytics
- Flask web framework
- OpenCV computer vision library
- SQLite for database management
- Flask-Login for authentication
