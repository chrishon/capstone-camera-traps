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

def prepare_diffusion_data(frames, noise_scheduler, num_train_timesteps=1000):
    # Convert frames to PyTorch tensors (B, H, W, C) → (B, C, H, W)
    frames = torch.tensor(np.array(frames)).float().permute(0, 3, 1, 2) / 255.0
    
    # Sample random timesteps and noise
    batch_size = len(frames)
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,))
    noise = torch.randn_like(frames)
    
    # Add noise to frames (forward diffusion process)
    noisy_frames = noise_scheduler.add_noise(frames, noise, timesteps)
    
    return noisy_frames, noise  # Model learns to predict noise from noisy_frames