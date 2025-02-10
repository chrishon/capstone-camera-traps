import cv2
import numpy as np
import torch

def extract_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (64, 64)))  # Resize to reduce complexity
    cap.release()
    return frames

def prepare_train_data(frames, sequence_length: int = 5):
    sequence_length = 5  # 3-5 frames as input, adjust as needed
    X, y = [], []

    for i in range(len(frames) - sequence_length):
        X.append(frames[i:i + sequence_length])  # Input sequence
        y.append(frames[i + sequence_length])    # Next frame to predict

    # Convert to numpy arrays and normalize
    X = np.array(X) / 255.0
    y = np.array(y) / 255.0

    X = X.reshape((X.shape[0],X.shape[1],X.shape[-1],X.shape[2],X.shape[3]))
    y = y.reshape((y.shape[0],y.shape[-1],y.shape[1],y.shape[2]))

    return X,y

def prepare_train_data_diffusion(frames, sequence_length=5):
    X, y = [], []
    
    for i in range(len(frames) - sequence_length):
        X.append(frames[i:i + sequence_length])
        y.append(frames[i + sequence_length])
    
    # Convert to numpy arrays and normalize to [-1, 1]
    X = np.array(X) / 255.0 * 2 - 1
    y = np.array(y) / 255.0 * 2 - 1
    
    # Reshape X to (N, seq_len, C, H, W) then stack channels
    X = np.transpose(X, (0, 1, 4, 2, 3))  # (N, seq_len, C, H, W)
    X_condition = X.reshape(X.shape[0], -1, X.shape[3], X.shape[4])  # (N, seq_len*C, H, W)
    
    # Reshape y to (N, C, H, W)
    y = np.transpose(y, (0, 3, 1, 2))
    
    return X_condition, y