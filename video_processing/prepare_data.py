import cv2
import numpy as np
import torch

def extract_frames(video_path: str, size: tuple[int, ...] = (64, 64)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frames.append(cv2.resize(frame, dsize=size))
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
    
    # Convert to numpy arrays and normalize
    X = np.array(X, dtype=np.float32) / 255.0 * 2 - 1  # [-1, 1]
    y = np.array(y, dtype=np.float32) / 255.0 * 2 - 1
    
    # Proper channel ordering for PyTorch
    X = np.transpose(X, (0, 1, 4, 2, 3))  # (N, seq_len, C, H, W)
    X_condition = X.reshape(X.shape[0], -1, X.shape[3], X.shape[4])  # (N, seq_len*C, H, W)
    
    y = np.transpose(y, (0, 3, 1, 2))  # (N, C, H, W)
    
    return X_condition, y