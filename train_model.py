from ultralytics import YOLO
import os
import yaml
from datetime import datetime

# Create directories for training
os.makedirs('data/images/train', exist_ok=True)
os.makedirs('data/images/val', exist_ok=True)
os.makedirs('data/labels/train', exist_ok=True)
os.makedirs('data/labels/val', exist_ok=True)

# Create dataset configuration file
def create_dataset_yaml():
    data = {
        'path': 'data',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 80,  # Number of classes (COCO has 80 classes)
        'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                 'hair drier', 'toothbrush']
    }

    with open('data/dataset.yaml', 'w') as f:
        yaml.dump(data, f)

# Create training configuration file
def create_training_config():
    config = {
        'model': 'yolov8m.yaml',  # Using nano model architecture
        'data': 'data/dataset.yaml',
        'epochs': 100,  # Number of training epochs
        'batch': 16,  # Batch size
        'imgsz': 640,  # Image size
        'device': '0',  # Use GPU if available
        'name': 'yolov8_custom',  # Name of the experiment
        'project': 'runs/train',  # Directory for saving results
        'exist_ok': False,  # Allow overwriting existing results
        'pretrained': False,  # Train from scratch
        'patience': 50,  # Patience for early stopping
        'save_period': 5,  # Save model every 5 epochs
        'optimizer': 'AdamW',  # Optimizer
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3.0,  # Warmup epochs (fractions ok)
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'fl_gamma': 0.0,  # Focal loss gamma (efficientDet default gamma=1.5)
        'label_smoothing': 0.0,  # Label smoothing
        'nbs': 64,  # Nominal batch size
        'auto': True,  # Auto-adjust lr scheduler and batch size
        'evolve': False,  # Evolve hyperparameters
        'cache': False,  # Cache images in RAM (fast)
        'image_weights': False,  # Use weighted image selection for training
        'quad': False,  # Use quadruplet augmentations
        'linear_lr': False,  # Linear LR
        'label_downsample_ratio': 1.0,  # Label downsampling ratio (2.0 is default)
        'warmup_cos': False,  # Cosine LR warmup
        'reproduce': False,  # Reproduce YOLOv5 results
        'close_mosaic': 10,  # Disable mosaic augmentation for final 10 epochs
        'save_json': True,  # Save COCO JSON
        'save_hybrid': False,  # Save hybrid pseudo labels
        'save_period': -1,  # Save checkpoint every x epochs (-1 for no saving)
        'seed': 0,  # Global training seed
        'deterministic': False,  # Deterministic training
        'single_cls': False,  # Train multi-class data as single-class
        'optimizer': 'AdamW',  # Optimizer (Adam, AdamW, SGD, RAdam, or RMSProp)
        'sync_bn': False,  # Use SyncBatchNorm, only available in DDP mode
        'recompute_loss': False,  # Use computed matching-matrix instead of gIoU matrix
        'reinit': False,  # Reinitialize optimizer states
        'ema': True,  # Use EMA
        'ema_decay': 0.9999,  # EMA decay rate
        'ema_tau': 2000,  # Epochs to wait before updating EMA weights
        'upload_dataset': False,  # Upload dataset to HUB
        'bbox_interval': -1,  # Set bounding-box image logging interval for W&B
        'artifact_alias': 'latest'  # Version of dataset artifact to use
    }

    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)

def train_model():
    # Create dataset configuration
    create_dataset_yaml()
    create_training_config()
    
    # Initialize a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')  # Using nano architecture
    
    # Train the model
    results = model.train(
        data='data/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8_custom',
        pretrained=False,  # Train from scratch
        device='0'  # Use GPU if available
    )
    
    print("\nTraining completed!")
    print(f"Best validation metrics:")
    print(f"mAP@0.5: {results.metrics['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95: {results.metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {results.metrics['metrics/precision(B)']:.4f}")
    print(f"Recall: {results.metrics['metrics/recall(B)']:.4f}")

if __name__ == '__main__':
    train_model()
