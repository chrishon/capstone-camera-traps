import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def predict_multiple_samples(model, condition, scheduler, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        # Create multiple noise instances
        sample_shape = (num_samples, 3, 64, 64)
        samples = torch.randn(sample_shape, device=device)
        
        # Repeat condition for all samples
        condition = condition.unsqueeze(0).repeat(num_samples, 1, 1, 1).to(device)
        
        # Denoising loop
        for t in scheduler.timesteps:
            model_input = torch.cat([samples, condition], dim=1)
            noise_pred = model(model_input, t).sample
            samples = scheduler.step(noise_pred, t, samples).prev_sample
        
        # Process all samples
        predicted_frames = []
        for sample in samples:
            frame = sample.cpu().numpy()
            frame = (frame + 1) / 2  # Denormalize
            frame = np.transpose(frame, (1, 2, 0))  # CHW to HWC
            predicted_frames.append(frame)
            
        return predicted_frames

def frame_analysis(predicted, ground_truth):
    """Compare frames using multiple metrics"""
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy().transpose(1, 2, 0)
    
    results = {}
    
    # Per-pixel absolute difference
    abs_diff = np.abs(predicted - ground_truth)
    results['mean_absolute_error'] = np.mean(abs_diff)
    results['max_absolute_error'] = np.max(abs_diff)
    
    # Structural Similarity (SSIM)
    results['ssim'] = ssim(predicted, ground_truth, 
                          multichannel=True, 
                          data_range=1.0,
                          channel_axis=2)
    
    # Patch-based differences
    patch_size = 16
    patches_pred = extract_patches(predicted, patch_size)
    patches_gt = extract_patches(ground_truth, patch_size)
    results['patch_diff'] = np.mean(np.abs(patches_pred - patches_gt), axis=(1,2,3))
    
    return results

def extract_patches(image, patch_size):
    """Extract image patches"""
    h, w, c = image.shape
    patches = []
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            if patch.shape[:2] == (patch_size, patch_size):
                patches.append(patch)
                
    return np.array(patches)

import matplotlib.pyplot as plt

def visualize_comparison(predicted, ground_truth, patch_size=16):
   
    pred_np = predicted.cpu().numpy().transpose(1, 2, 0) if isinstance(predicted, torch.Tensor) else predicted
    gt_np = ground_truth.cpu().numpy().transpose(1, 2, 0) if isinstance(ground_truth, torch.Tensor) else ground_truth
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original vs Predicted
    axes[0,0].imshow(gt_np)
    axes[0,0].set_title("Ground Truth")
    axes[0,1].imshow(pred_np)
    axes[0,1].set_title("Predicted")
    
    # Absolute difference heatmap
    diff = np.abs(gt_np - pred_np)
    im = axes[1,0].imshow(diff.mean(axis=-1), cmap='hot', vmin=0, vmax=0.5)
    plt.colorbar(im, ax=axes[1,0])
    axes[1,0].set_title("Absolute Difference Heatmap")
    
    # Patch-based analysis
    patches_diff = []
    for i in range(0, gt_np.shape[0], patch_size):
        for j in range(0, gt_np.shape[1], patch_size):
            patch_diff = diff[i:i+patch_size, j:j+patch_size].mean()
            patches_diff.append(patch_diff)
    
    patch_grid = np.zeros((gt_np.shape[0]//patch_size, 
                         gt_np.shape[1]//patch_size))
    patch_grid = patch_grid.reshape(-1)
    patch_grid[:len(patches_diff)] = patches_diff
    patch_grid = patch_grid.reshape((gt_np.shape[0]//patch_size,
                                   gt_np.shape[1]//patch_size))
    
    im = axes[1,1].imshow(patch_grid, cmap='hot', vmin=0, vmax=0.5)
    plt.colorbar(im, ax=axes[1,1])
    axes[1,1].set_title(f"Patch Average Differences (size {patch_size}x{patch_size})")
    
    plt.tight_layout()
    plt.show()

def temporal_analysis(frames, window_size=5):
    """Analyze temporal consistency across frames"""
    metrics = {
        'flow_magnitude': [],
        'intensity_change': [],
        'temporal_ssim': []
    }
    
    for i in range(len(frames)-1):
        # Optical flow magnitude (requires previous frame)
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        metrics['flow_magnitude'].append(np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean())
        
        # Intensity change
        metrics['intensity_change'].append(np.mean(np.abs(frames[i+1] - frames[i])))
        
        # Temporal SSIM
        metrics['temporal_ssim'].append(ssim(frames[i], frames[i+1], 
                                        multichannel=True, 
                                        data_range=1.0))
    
    # Moving averages
    for k in metrics:
        metrics[k] = moving_average(metrics[k], window_size)
        
    return metrics

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

