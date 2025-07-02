from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for, flash, session
import cv2
from ultralytics import YOLO
import os
import logging
import time
import numpy as np
from datetime import datetime
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import User, ContactMessage, init_db, get_db_connection, get_active_sessions, set_user_as_admin
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv, Bottleneck, C2f, SPPF, Concat, DFL

# Initialize the database
init_db()

# Allow PyTorch to safely unpickle all needed YOLO classes
torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    Conv,
    Bottleneck,
    C2f,
    SPPF,
    Concat,
    DFL
])

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

import json
import torch
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Module, Sequential, ModuleList, ModuleDict
from ultralytics.nn.modules import Conv, Bottleneck, SPPF, C2f, Concat, Detect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize object tracker

# Initialize heatmap
heatmap = np.zeros((480, 640), dtype=np.uint8)

# Initialize analytics
analytics = {
    'total_objects': 0,
    'object_speeds': {},
    'zone_counts': {},
    'detection_history': [],
    'fps': 0,
    'frame_count': 0,
    'start_time': time.time()
}

# Frame buffer handling
def process_frame_buffer(frame, buffer_size=3):
    """Process frame buffer to reduce latency."""
    if len(frame_buffer) >= buffer_size:
        # Use oldest frame to reduce latency
        return frame_buffer.pop(0)
    return frame

# Frame buffer
frame_buffer = []

# Frame rate optimization
def calculate_fps():
    """Calculate frames per second."""
    current_time = time.time()
    time_diff = current_time - analytics['start_time']
    if time_diff > 0:
        analytics['fps'] = analytics['frame_count'] / time_diff
        analytics['frame_count'] = 0
        analytics['start_time'] = current_time
    return analytics['fps']

# Define zone boundaries
def get_zone(x1, y1, x2, y2, img_width, img_height):
    """Determine which zone an object is in."""
    # Calculate object's center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Define zones based on image dimensions
    # Divide image into 3x3 grid
    zone_width = img_width / 3
    zone_height = img_height / 3
    
    # Determine zone
    zone_x = int(center_x // zone_width)
    zone_y = int(center_y // zone_height)
    
    # Return zone as string (e.g., "1-1", "2-2", etc.)
    return f"{zone_x+1}-{zone_y+1}"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-please-change-in-production')

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Admin-only access decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need admin privileges to access this page.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize the database
init_db()

# Load optimized YOLOv8m model with balanced accuracy and performance
try:
    # Import required classes
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import (
        Conv, 
        Bottleneck, 
        SPPF, 
        C2f, 
        Concat, 
        Detect, 
        DFL
    )
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn import Sequential, ModuleList, ModuleDict

    # Create a context manager for loading the model
    with torch.serialization.safe_globals([
        DetectionModel,
        Sequential,
        ModuleList,
        ModuleDict,
        Conv,
        Bottleneck,
        SPPF,
        C2f,
        Concat,
        Detect,
        DFL
    ]):
        # Load the model within the safe context
        model = YOLO('yolov8m.pt')

    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise

# Configure model for high accuracy with smooth performance
model.conf = 0.35  # Lower confidence threshold for better detection
model.iou = 0.50  # Lower IOU threshold for better tracking
model.max_det = 150  # Reasonable max detections
model.agnostic_nms = False
model.classes = None
model.amp = True  # Use AMP for better performance

# Move model to GPU if available and set to half precision
if torch.cuda.is_available():
    model.to('cuda')
    model.model.half()
    logger.info("Model moved to GPU and set to half precision")

# Set model to evaluation mode
model.model.eval()
logger.info("YOLO model loaded and configured successfully")

# Ensure model is loaded correctly
if not hasattr(model, 'model'):
    raise Exception("Failed to load YOLO model properly")

# Ensure uploads and outputs directories exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variables for video detection state
video_processing = False
stop_detection = False
current_progress = 0
processing_start_time = None
estimated_time_remaining = None




# Helper function to create error frames with text
def create_error_frame(error_text):
    # Create a black image with error text
    height, width = 480, 640
    img = np.zeros((height, width, 3), np.uint8)
    
    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)  # White text
    thickness = 2
    
    # Split text into multiple lines if needed
    words = error_text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        text = ' '.join(current_line)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        if text_size[0] > width - 40:  # Leave some margin
            current_line.pop()  # Remove last word
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate y position for centered text
    y_position = height // 2 - ((len(lines) - 1) * 30) // 2
    
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x_position = (width - text_size[0]) // 2
        cv2.putText(img, line, (x_position, y_position), font, font_scale, color, thickness)
        y_position += 30
    
    # Preprocess image efficiently
    img_height, img_width = img.shape[:2]
    
    # Convert to RGB once
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform detection with optimized settings
    results = model(img_rgb, verbose=False, imgsz=800)
    
    # Get detections
    boxes = results[0].boxes
    
    # Skip processing if no detections
    if not boxes:
        return img
    
    # Process detections efficiently
    filtered_boxes = []
    for box in boxes:
        conf = box.conf[0].item()
        if conf < 0.45:
            continue
            
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Basic size filtering
        width = x2 - x1
        height = y2 - y1
        if width < 15 or height < 15:
            continue
            
        filtered_boxes.append((box, width, height))
    
    # If no valid detections after filtering
    if not filtered_boxes:
        return img
    
    # Process filtered detections
    for box, width, height in filtered_boxes:
        # Get detection parameters
        conf = box.conf[0].item()
        class_id = box.cls[0].item()
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Apply basic filtering
        if conf < 0.45:
            continue
            
        # Calculate object size
        width = x2 - x1
        height = y2 - y1
        
        # Basic size filtering
        if width < 15 or height < 15:
            continue
            
        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add label only for high confidence detections
        if conf > 0.6:
            label = f'{model.names[int(class_id)]} {conf:.2f}'
            cv2.putText(img, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Update analytics
    analytics['total_objects'] += len(filtered_boxes)
    
    # Update zone counts
    for box, width, height in filtered_boxes:
        if box.conf[0].item() > 0.45:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            zone = get_zone(x1, y1, x2, y2, img_width, img_height)
            analytics['zone_counts'][zone] = analytics['zone_counts'].get(zone, 0) + 1
            
            # Filter based on object size (ignore very small detections)
            if width < 20 or height < 20:
                continue
                
            # Calculate aspect ratio for better object classification
            aspect_ratio = width / height
            
            # Apply class-specific filtering
            if class_id == 0:  # Person
                if aspect_ratio < 0.2 or aspect_ratio > 2.0:
                    continue
            elif class_id == 2:  # Car
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    continue
            
            # Draw bounding box efficiently
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add label only for high confidence detections
            if conf > 0.6:
                label = f'{model.names[int(class_id)]} {conf:.2f}'
                cv2.putText(img, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update analytics with more detailed information
        analytics['total_objects'] += len(boxes)
        
        # Update zone counts with confidence weighting
        for box in boxes:
            if box.conf[0].item() > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                zone = get_zone(x1, y1, x2, y2, img_width, img_height)
                weight = box.conf[0].item()  # Use confidence as weight
                analytics['zone_counts'][zone] = analytics['zone_counts'].get(zone, 0) + weight
                
                # Track object speed
                if zone in analytics['object_speeds']:
                    analytics['object_speeds'][zone] = (analytics['object_speeds'][zone] + width * height * weight) / 2
                else:
                    analytics['object_speeds'][zone] = width * height * weight
    
    # Convert to JPEG
    _, buffer = cv2.imencode('.jpg', img)
    return buffer

@app.route('/')
@login_required
def index():
    """Render the homepage with analytics."""
    return render_template('index.html', 
                         total_objects=analytics['total_objects'],
                         zone_counts=analytics['zone_counts'])

@app.route('/about')
def about():
    """Render the about page."""
    return render_template("about.html")

@app.route('/services')
def services():
    """Render the services page."""
    return render_template("services.html")

@app.route('/contact')
def contact():
    """Render the contact page."""
    return render_template("contact.html")

# Admin routes
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard showing overview of users and messages."""
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, email, is_admin, created_at FROM users').fetchall()
    conn.close()
    
    messages = ContactMessage.get_all()
    active_sessions = get_active_sessions()
    
    return render_template('admin_dashboard.html', users=users, messages=messages, active_sessions=active_sessions)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Admin page to manage users."""
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, email, is_admin, created_at FROM users').fetchall()
    conn.close()
    
    return render_template('admin_users.html', users=users)

@app.route('/admin/contacts')
@login_required
@admin_required
def admin_contacts():
    """Admin page to view contact form submissions."""
    messages = ContactMessage.get_all()
    return render_template('admin_contacts.html', messages=messages)

@app.route('/admin/sessions')
@login_required
@admin_required
def admin_sessions():
    """Admin page to view and manage active sessions."""
    active_sessions = get_active_sessions()
    return render_template('admin_sessions.html', active_sessions=active_sessions)

@app.route('/admin/users/<int:user_id>/make-admin', methods=['POST'])
@login_required
@admin_required
def admin_make_admin(user_id):
    """Make a user an admin."""
    conn = get_db_connection()
    user = conn.execute('SELECT username FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user:
        flash('User not found.', 'danger')
    else:
        conn.execute('UPDATE users SET is_admin = 1 WHERE id = ?', (user_id,))
        conn.commit()
        flash(f'User {user["username"]} is now an admin.', 'success')
    
    conn.close()
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    """Delete a user."""
    # Don't allow deleting yourself
    if user_id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin_users'))
    
    conn = get_db_connection()
    user = conn.execute('SELECT username FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user:
        flash('User not found.', 'danger')
    else:
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        flash(f'User {user["username"]} has been deleted.', 'success')
    
    conn.close()
    return redirect(url_for('admin_users'))

@app.route('/admin/sessions/<string:session_id>/terminate', methods=['POST'])
@login_required
@admin_required
def admin_terminate_session(session_id):
    """Terminate a user session."""
    # This is a placeholder since we don't have actual session tracking
    # In a real application, you would remove the session from your session store
    flash('Session terminated successfully.', 'success')
    return redirect(url_for('admin_sessions'))

@app.route('/contact/submit', methods=['POST'])
def contact_submit():
    """Handle contact form submission."""
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')
    
    # Validate form data
    if not all([name, email, subject, message]):
        flash('Please fill out all fields.', 'danger')
        return redirect(url_for('contact'))
    
    try:
        # Store the contact message in the database
        from models import ContactMessage
        contact_message = ContactMessage.create(name, email, subject, message)
        
        # Log the submission
        logger.info(f"Contact form submission from {name} ({email}): {subject} - Stored with ID: {contact_message.id}")
        
        # Send email to admin
        admin_email = "sunniraj4511@gmail.com"
        send_contact_email(name, email, subject, message, admin_email)
        
        # Flash a success message
        flash('Thank you for your message! We will get back to you soon.', 'success')
    except Exception as e:
        logger.error(f"Error processing contact form submission: {str(e)}")
        flash('There was an error processing your message. Please try again later.', 'danger')
    
    # Redirect back to the contact page
    return redirect(url_for('contact'))

def send_contact_email(name, email, subject, message, admin_email):
    """Send contact form data to admin email."""
    try:
        # Create email content
        msg = MIMEMultipart()
        msg['From'] = email
        msg['To'] = admin_email
        msg['Subject'] = f"Contact Form: {subject}"
        
        # Email body
        body = f"""New Contact Form Submission
        
Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
This message was sent from the Object Detection System contact form.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server and send email
        # Note: For production, you should use environment variables for these credentials
        # and consider using a service like SendGrid, Mailgun, etc.
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        
        # You would need to set up an app password for your Gmail account
        # and store it securely, not hardcoded like this
        # For demonstration purposes only - in production use environment variables
        app_password = os.environ.get('EMAIL_PASSWORD', '')
        if not app_password:
            logger.warning("Email password not set. Email not sent.")
            return
            
        server.login(admin_email, app_password)
        text = msg.as_string()
        server.sendmail(email, admin_email, text)
        server.quit()
        
        logger.info(f"Contact email sent to {admin_email} from {email}")
    except Exception as e:
        logger.error(f"Error sending contact email: {str(e)}")
        # Don't raise the exception - we still want to store the message in the database
        # even if email sending fails

# This route has been replaced by the new admin_contacts route with proper admin access control

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        # Debug logging
        logger.info(f"Login attempt for username: {username}")
        
        user = User.find_by_username(username)
        
        if user:
            logger.info(f"User found: {user.username}, ID: {user.id}")
            if user.check_password(password):
                try:
                    login_user(user, remember=remember)
                    logger.info(f"Login successful for user: {username}")
                    flash('Login successful!', 'success')
                    next_page = request.args.get('next')
                    return redirect(next_page or url_for('index'))
                except Exception as e:
                    logger.error(f"Error during login_user: {str(e)}")
                    flash('An error occurred during login. Please try again.', 'danger')
            else:
                logger.info(f"Invalid password for user: {username}")
                flash('Invalid username or password', 'danger')
        else:
            logger.info(f"User not found: {username}")
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not username or not email or not password or not confirm_password:
            flash('All fields are required', 'danger')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
            
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return render_template('register.html')
            
        # Check if username or email already exists
        if User.find_by_username(username):
            flash('Username already exists', 'danger')
            return render_template('register.html')
            
        if User.find_by_email(email):
            flash('Email already exists', 'danger')
            return render_template('register.html')
            
        # Create new user
        user = User.create(username, email, password)
        
        if user:
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('An error occurred during registration', 'danger')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Handle user logout."""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/outputs/<filename>')
def output_file(filename):
    """Serve output files with proper MIME types."""
    try:
        # Set the correct MIME type for video files
        mimetype = None
        if filename.endswith('.mp4'):
            mimetype = 'video/mp4'
        return send_from_directory(OUTPUT_FOLDER, filename, mimetype=mimetype)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/start_detection')
@login_required
def start_detection():
    """Start real-time video detection from webcam."""
    def generate_frames():
        global video_processing, stop_detection
        cap = None
        
        try:
            # Print available cameras for debugging
            logger.info("Checking available camera devices...")
            available_cameras = []
            for i in range(10):  # Check indices 0-9
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    ret, _ = temp_cap.read()
                    if ret:
                        available_cameras.append(i)
                        logger.info(f"Camera index {i} is available")
                    temp_cap.release()
            
            if not available_cameras:
                logger.error("No cameras detected on the system")
                error_frame = create_error_frame("No cameras detected on your system. Please connect a webcam and try again.")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + error_frame.tobytes() + b'\r\n')
                return
            
            logger.info(f"Available cameras: {available_cameras}")
            
            # Try multiple approaches to open the camera
            success = False
            
            # First try: DirectShow backend with index 0 (most common webcam)
            try:
                logger.info("Trying camera 0 with DirectShow backend")
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info("Successfully opened camera 0 with DirectShow backend")
                        success = True
                    else:
                        logger.warning("Camera opened but couldn't read frame, releasing")
                        cap.release()
                        cap = None
                else:
                    logger.warning("Failed to open camera with DirectShow backend")
                    if cap is not None:
                        cap.release()
                        cap = None
            except Exception as e:
                logger.error(f"Error with DirectShow backend: {str(e)}")
                if cap is not None:
                    cap.release()
                    cap = None
            
            # Second try: Default backend with index 0
            if not success:
                try:
                    logger.info("Trying camera 0 with default backend")
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            logger.info("Successfully opened camera 0 with default backend")
                            success = True
                        else:
                            logger.warning("Camera opened but couldn't read frame, releasing")
                            cap.release()
                            cap = None
                    else:
                        logger.warning("Failed to open camera with default backend")
                        if cap is not None:
                            cap.release()
                            cap = None
                except Exception as e:
                    logger.error(f"Error with default backend: {str(e)}")
                    if cap is not None:
                        cap.release()
                        cap = None
            
            # Third try: Try other camera indices
            if not success:
                for idx in range(1, 3):  # Try indices 1 and 2
                    try:
                        logger.info(f"Trying camera index {idx}")
                        cap = cv2.VideoCapture(idx)
                        if cap.isOpened():
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                logger.info(f"Successfully opened camera {idx}")
                                success = True
                                break
                            else:
                                logger.warning(f"Camera {idx} opened but couldn't read frame, releasing")
                                cap.release()
                                cap = None
                        else:
                            logger.warning(f"Failed to open camera {idx}")
                            if cap is not None:
                                cap.release()
                                cap = None
                    except Exception as e:
                        logger.error(f"Error with camera {idx}: {str(e)}")
                        if cap is not None:
                            cap.release()
                            cap = None
            
            # If all attempts failed
            if not success or cap is None:
                logger.error("Failed to open any camera after multiple attempts")
                error_frame = create_error_frame("Could not access webcam. Please check your camera connection and permissions.")
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + error_frame.tobytes() + b'\r\n')
                return
                
            # Set webcam properties for better performance
            try:
                # First read a test frame to make sure the camera is working
                test_success, test_frame = cap.read()
                if not test_success or test_frame is None:
                    logger.error("Failed to read initial test frame from webcam")
                    error_frame = create_error_frame("Camera connected but not providing video stream. Try restarting your computer.")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + error_frame.tobytes() + b'\r\n')
                    cap.release()
                    return
                    
                # Set properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set FPS to 30
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size for lower latency
                
                # Check if properties were actually set
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"Webcam properties - Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}")
            except Exception as e:
                logger.error(f"Error setting webcam properties: {str(e)}")
                # Continue anyway, these are just optimizations
            
            logger.info("Webcam opened successfully, starting detection")
            video_processing = True
            stop_detection = False
            frame_count = 0
            last_detected_frame_bytes = None

            while cap.isOpened() and not stop_detection:
                success, frame = cap.read()
                if not success:
                    logger.warning("Failed to read frame from webcam")
                    # Try to read one more time before giving up
                    success, frame = cap.read()
                    if not success:
                        logger.error("Repeatedly failed to read from webcam, stopping")
                        error_frame = create_error_frame("Lost connection to webcam")
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + error_frame.tobytes() + b'\r\n')
                        break

                frame_count += 1
                frame_bytes = None

                # Process every 2nd frame for performance
                if frame_count % 2 == 0:
                    try:
                        # Make sure frame is not None and has valid dimensions
                        if frame is None or frame.size == 0:
                            logger.warning("Empty frame received")
                            continue
                        
                        # Check frame dimensions and orientation
                        h, w = frame.shape[:2]
                        if h == 0 or w == 0:
                            logger.warning(f"Invalid frame dimensions: {w}x{h}")
                            continue
                            
                        # Resize while maintaining aspect ratio
                        target_width = 640
                        aspect_ratio = w / h
                        target_height = int(target_width / aspect_ratio)
                        
                        # Ensure target height is reasonable
                        if target_height > 1000 or target_height < 100:
                            target_height = 480
                            
                        frame_resized = cv2.resize(frame, (target_width, target_height))
                        
                        # Enhanced detection with optimized parameters
                        # Wrap the model inference in a try-except block
                        try:
                            # Optimized settings for faster real-time detection
                            results = model(
                                frame_resized, 
                                conf=0.3,       # Balanced confidence threshold for accuracy and speed
                                iou=0.45,       # Balanced IoU threshold
                                max_det=20,     # Limit detections for better performance
                                verbose=False,  # Disable verbose output for speed
                                augment=False   # Disable augmentation for faster processing
                            )
                            
                            # Simplified and optimized post-processing for better performance
                            # Use YOLO's built-in visualization but with custom filtering for person detection
                            
                            # Get the detection results
                            if len(results[0].boxes) > 0:
                                # Filter out small person detections (likely hands)
                                filtered_indices = []
                                class_counts = {}
                                
                                # Process each detection
                                for i, (box, cls, conf) in enumerate(zip(
                                    results[0].boxes.xyxy.cpu().numpy(),
                                    results[0].boxes.cls.cpu().numpy(),
                                    results[0].boxes.conf.cpu().numpy())):
                                    
                                    cls_name = results[0].names[int(cls)]
                                    
                                    # Only filter person class to avoid hand misclassification
                                    if cls_name == 'person':
                                        # Calculate relative size
                                        width = box[2] - box[0]
                                        height = box[3] - box[1]
                                        box_area = width * height
                                        frame_area = frame_resized.shape[0] * frame_resized.shape[1]
                                        relative_size = box_area / frame_area
                                        
                                        # Keep only if it's a reasonable size
                                        if relative_size > 0.03:
                                            filtered_indices.append(i)
                                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                                    else:
                                        # Keep all other object classes
                                        filtered_indices.append(i)
                                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                                
                                # Use YOLO's built-in plotting for better performance
                                detected_frame = results[0].plot(boxes=filtered_indices)
                                
                                # Add a simple detection summary at the top
                                if class_counts:
                                    summary_text = "Detected: " + ", ".join([f"{count} {name}" for name, count in class_counts.items()])
                                    cv2.rectangle(detected_frame, (0, 0), (detected_frame.shape[1], 30), (0, 0, 0), -1)
                                    cv2.putText(detected_frame, summary_text, (10, 20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            else:
                                # No detections, use the original frame
                                detected_frame = frame_resized.copy()
                            
                            # Detection summary is already added above
                        except Exception as model_error:
                            logger.error(f"Model inference error: {str(model_error)}")
                            # Fallback: just display the frame with a warning message
                            detected_frame = frame_resized.copy()
                            cv2.putText(detected_frame, "Detection model error - showing raw camera feed", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # Add timestamp to show the feed is live
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            cv2.putText(detected_frame, timestamp, 
                                        (10, detected_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Use higher quality JPEG encoding
                        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                        _, buffer = cv2.imencode('.jpg', detected_frame, encode_params)
                        frame_bytes = buffer.tobytes()
                        last_detected_frame_bytes = frame_bytes
                    except Exception as e:
                        logger.error(f"Error processing frame: {str(e)}")
                        if 'CUDA' in str(e):
                            # Special handling for CUDA errors
                            error_frame = create_error_frame("GPU error detected. Try restarting the application.")
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + error_frame.tobytes() + b'\r\n')
                            break
                        continue
                elif last_detected_frame_bytes:
                    frame_bytes = last_detected_frame_bytes

                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            video_processing = False
            logger.info("Webcam detection stopped")

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
@login_required
def stop_detection_route():
    """Stop real-time video detection."""
    global stop_detection
    try:
        stop_detection = True
        logger.info("Detection stop requested")
        return jsonify({'message': 'Detection stopped successfully.', 'status': 'success'})
    except Exception as e:
        logger.error(f"Error stopping detection: {str(e)}")
        return jsonify({'message': f'Error stopping detection: {str(e)}', 'status': 'error'}), 500

@app.route('/detect-image', methods=['POST'])
@login_required
def detect_image():
    """Detect objects in an uploaded image."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Failed to read the image'}), 400

        # Lower confidence threshold to detect smaller objects like pens and glasses
        results = model(image, conf=0.25)
        
        # Log detected objects (without voice feedback)
        try:
            detected_objects = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < 0.40:  # Minimum confidence threshold
                        continue
                    cls_name = r.names[cls_id]
                    if cls_name not in detected_objects:
                        detected_objects.append(cls_name)
            
            # Log what was found in the image
            if detected_objects:
                logger.info(f"Objects detected in image: {detected_objects}")
        except Exception as e:
            logger.error(f"Error logging detected objects: {str(e)}")
        
        detected_image = results[0].plot()

        output_filename = f"detected_{file.filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, detected_image)

        return jsonify({'message': 'Objects detected successfully', 'output_filename': output_filename})
    except Exception as e:
        logger.error(f"Error in image detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-video', methods=['POST'])
@login_required
def detect_video():
    """Detect objects in an uploaded video."""
    global video_processing, stop_detection, current_progress

    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video file'}), 400

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use the same filename for the output
        output_filename = file.filename
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        video_processing = True
        stop_detection = False
        current_progress = 0
        frames_processed = 0
        global processing_start_time, estimated_time_remaining
        processing_start_time = time.time()
        estimated_time_remaining = None

        while cap.isOpened() and not stop_detection:
            ret, frame = cap.read()
            if not ret or stop_detection:
                break

            try:
                original_height, original_width = frame.shape[:2]
                aspect_ratio = original_width / original_height
                target_width = 640
                target_height = int(target_width / aspect_ratio)

                frame_resized = cv2.resize(frame, (target_width, target_height))
                
                # Use the same enhanced detection parameters as real-time detection
                try:
                    # Optimized settings for faster video processing
                    results = model(
                        frame_resized, 
                        conf=0.3,       # Balanced confidence threshold for accuracy and speed
                        iou=0.45,       # Balanced IoU threshold
                        max_det=20,     # Limit detections for better performance
                        verbose=False,  # Disable verbose output for speed
                        augment=False   # Disable augmentation for faster processing
                    )
                    
                    # Simplified and optimized post-processing for better performance
                    # Use YOLO's built-in visualization but with custom filtering for person detection
                    
                    # Get the detection results
                    if len(results[0].boxes) > 0:
                        # Filter out small person detections (likely hands)
                        filtered_indices = []
                        class_counts = {}
                        
                        # Process each detection
                        for i, (box, cls, conf) in enumerate(zip(
                            results[0].boxes.xyxy.cpu().numpy(),
                            results[0].boxes.cls.cpu().numpy(),
                            results[0].boxes.conf.cpu().numpy())):
                            
                            cls_name = results[0].names[int(cls)]
                            
                            # Only filter person class to avoid hand misclassification
                            if cls_name == 'person':
                                # Calculate relative size
                                width = box[2] - box[0]
                                height = box[3] - box[1]
                                box_area = width * height
                                frame_area = frame_resized.shape[0] * frame_resized.shape[1]
                                relative_size = box_area / frame_area
                                
                                # Keep only if it's a reasonable size
                                if relative_size > 0.03:
                                    filtered_indices.append(i)
                                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                            else:
                                # Keep all other object classes
                                filtered_indices.append(i)
                                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        
                        # Use YOLO's built-in plotting for better performance
                        detected_frame_resized = results[0].plot(boxes=filtered_indices)
                        
                        # Add a simple detection summary at the top
                        if class_counts:
                            summary_text = "Detected: " + ", ".join([f"{count} {name}" for name, count in class_counts.items()])
                            cv2.rectangle(detected_frame_resized, (0, 0), (detected_frame_resized.shape[1], 30), (0, 0, 0), -1)
                            cv2.putText(detected_frame_resized, summary_text, (10, 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        # No detections, use the original frame
                        detected_frame_resized = frame_resized.copy()
                    
                    # Class counts are already handled in the filtering code above
                    
                    # Summary is already added in the filtering code above
                except Exception as model_error:
                    logger.error(f"Model inference error in video processing: {str(model_error)}")
                    # Fallback: just display the original frame with a warning message
                    detected_frame_resized = frame_resized.copy()
                    cv2.putText(detected_frame_resized, "Detection model error", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                detected_frame_original_size = cv2.resize(detected_frame_resized, (frame_width, frame_height))
                out.write(detected_frame_original_size)

                frames_processed += 1
                if total_frames > 0:
                    current_progress = int((frames_processed / total_frames) * 100)
                    
                    # Calculate estimated time remaining
                    if frames_processed > 10:  # Wait for a few frames to get a stable estimate
                        elapsed_time = time.time() - processing_start_time
                        frames_per_second = frames_processed / elapsed_time if elapsed_time > 0 else 0
                        remaining_frames = total_frames - frames_processed
                        if frames_per_second > 0:
                            estimated_time_remaining = remaining_frames / frames_per_second
                        else:
                            estimated_time_remaining = None
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                continue

        cap.release()
        out.release()
        video_processing = False
        if not stop_detection:
            current_progress = 100

        return jsonify({
            'message': 'Video processing finished' if not stop_detection else 'Video processing stopped',
            'output_filename': output_filename,
            'stopped_early': stop_detection
        })
    except Exception as e:
        logger.error(f"Error in video detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_progress')
@login_required
def video_progress():
    """Get current video processing progress."""
    global current_progress, video_processing, estimated_time_remaining
    
    response_data = {
        'progress': current_progress, 
        'processing': video_processing
    }
    
    # Add estimated time remaining if available
    if estimated_time_remaining is not None:
        response_data['eta_seconds'] = int(estimated_time_remaining)
        
        # Format time for display
        if estimated_time_remaining > 60:
            minutes = int(estimated_time_remaining // 60)
            seconds = int(estimated_time_remaining % 60)
            response_data['eta_formatted'] = f"{minutes}m {seconds}s"
        else:
            response_data['eta_formatted'] = f"{int(estimated_time_remaining)}s"
    
    return jsonify(response_data)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
