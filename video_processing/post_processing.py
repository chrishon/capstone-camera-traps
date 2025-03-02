import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def predict_multiple_samples(model, condition, scheduler, device, num_samples=5, size = (64,64)):
    model.eval()
    with torch.no_grad():
        # Create multiple noise instances
        h = size[0]
        w = size[1]
        sample_shape = (num_samples, 3, h, w)
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

def multi_sample_frame_analysis(predictions, ground_truth):
    """Compare multiple predictions to a single ground truth"""
    metrics = {
        'mae': [],
        'ssim': [],
        'patch_diff': []
    }
    
    for pred in predictions:
        results = frame_analysis(pred, ground_truth)
        metrics['mae'].append(results['mean_absolute_error'])
        metrics['ssim'].append(results['ssim'])
        metrics['patch_diff'].append(results['patch_diff'])
    
    # Calculate statistics
    return {
        'mae_mean': np.mean(metrics['mae']),
        'mae_std': np.std(metrics['mae']),
        'ssim_mean': np.mean(metrics['ssim']),
        'ssim_std': np.std(metrics['ssim']),
        'patch_diff_dist': np.array(metrics['patch_diff'])
    }

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

def visualize_multiple_comparisons(predictions, ground_truth, patch_size=16):
    fig = plt.figure(figsize=(24, 18))  # Increased figure size
    num_samples = len(predictions)
    
    # Create grid: 3 rows (images, diffs, histograms), num_samples+1 columns
    gs = fig.add_gridspec(3, num_samples + 1)
    
    # Precompute histogram data
    gt_vals = ground_truth.flatten()
    gt_hist, gt_bins = np.histogram(gt_vals, bins=50, range=(0, 1))
    bin_centers = (gt_bins[:-1] + gt_bins[1:]) / 2
    
    # Calculate max count for consistent y-axis scaling
    max_count = gt_hist.max()
    for pred in predictions:
        pred_hist, _ = np.histogram(pred.flatten(), bins=gt_bins, range=(0, 1))
        max_count = max(max_count, pred_hist.max())

    # Ground truth column
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_gt.imshow(ground_truth)
    ax_gt.set_title("Ground Truth")
    
    # Ground truth histogram
    ax_gt_hist = fig.add_subplot(gs[2, 0])
    ax_gt_hist.plot(bin_centers, gt_hist, color='blue', label='Ground Truth')
    ax_gt_hist.set_ylim(0, max_count * 1.1)
    ax_gt_hist.set_title("Pixel Value Distribution")
    ax_gt_hist.legend()

    # Average difference
    ax_avg = fig.add_subplot(gs[1, 0])
    avg_diff = np.mean([np.abs(p - ground_truth) for p in predictions], axis=0).mean(axis=-1)
    im_avg = ax_avg.imshow(avg_diff, cmap='hot', vmin=0, vmax=0.5)
    ax_avg.set_title("Average Diff")
    plt.colorbar(im_avg, ax=ax_avg)

    # Sample columns
    for i, pred in enumerate(predictions):
        # Prediction image
        ax_pred = fig.add_subplot(gs[0, i+1])
        ax_pred.imshow(pred)
        ax_pred.set_title(f"Sample {i+1}")
        
        # Difference map
        ax_diff = fig.add_subplot(gs[1, i+1])
        diff = np.abs(pred - ground_truth).mean(axis=-1)
        im_diff = ax_diff.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        ax_diff.set_title(f"Diff {i+1}")
        plt.colorbar(im_diff, ax=ax_diff)
        
        # Histogram
        ax_hist = fig.add_subplot(gs[2, i+1])
        pred_hist, _ = np.histogram(pred.flatten(), bins=gt_bins, range=(0, 1))
        ax_hist.plot(bin_centers, gt_hist, color='blue', label='Ground Truth')
        ax_hist.plot(bin_centers, pred_hist, color='orange', label='Predicted')
        ax_hist.set_ylim(0, max_count * 1.1)
        ax_hist.set_title(f"Sample {i+1} Distribution")
        ax_hist.legend()

    plt.tight_layout()
    plt.show()

def temporal_analysis(frames, window_size=5,patch_size=5):
    """Analyze temporal consistency across frames"""
    metrics = {
        'flow_magnitude': [],
        'intensity_change': [],
        'temporal_ssim': [],
        'patch_mean_change': [],
        'patch_std_change': []
    }
    
    for i in range(len(frames)-1):
        # Convert frames to uint8 for optical flow calculation
        frame_current = (frames[i] * 255).astype(np.uint8)
        frame_next = (frames[i+1] * 255).astype(np.uint8)
        
        # Optical flow magnitude
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(frame_current, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frame_next, cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        metrics['flow_magnitude'].append(np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean())
        
        # Intensity change (using original float values)
        metrics['intensity_change'].append(np.mean(np.abs(frames[i+1] - frames[i])))
        
        # Structural similarity with explicit channel handling
        metrics['temporal_ssim'].append(
            ssim(frames[i], frames[i+1],
                 data_range=1.0,
                 channel_axis=2,  # Explicit channel axis
                 win_size=7 if min(frames[i].shape[:2]) >= 7 else 3)
        )
    # New patch analysis
        patches_curr = extract_patches(frames[i], patch_size)
        patches_next = extract_patches(frames[i+1], patch_size)
        
        mean_diffs = []
        std_diffs = []
        
        for p_curr, p_next in zip(patches_curr, patches_next):
            # Calculate patch statistics
            mean_curr, std_curr = np.mean(p_curr), np.std(p_curr)
            mean_next, std_next = np.mean(p_next), np.std(p_next)
            
            mean_diffs.append(np.abs(mean_curr - mean_next))
            std_diffs.append(np.abs(std_curr - std_next))
        
        metrics['patch_mean_change'].append(np.mean(mean_diffs))
        metrics['patch_std_change'].append(np.mean(std_diffs))
    # Moving averages
    for k in metrics:
        if len(metrics[k]) > 0:
            metrics[k] = moving_average(metrics[k], window_size)
        
    return metrics

def multi_sample_temporal_analysis(predictions_sequence, ground_truth_sequence):
    """Analyze temporal consistency across multiple predicted sequences"""
    metrics = {
        'flow_consistency': [],
        'intensity_variance': []
    }
    
    for sample_preds in predictions_sequence:
        # Calculate temporal metrics for this sample
        temp_metrics = temporal_analysis(sample_preds)
        
        # Compare to ground truth temporal metrics
        gt_temp_metrics = temporal_analysis(ground_truth_sequence)
        
        # Calculate consistency metrics
        metrics['flow_consistency'].append(
            np.mean(np.abs(np.array(temp_metrics['flow_magnitude']) - 
                          np.array(gt_temp_metrics['flow_magnitude'])))
        )
        
        metrics['intensity_variance'].append(
            np.var(temp_metrics['intensity_change']))
    
    return {
        'flow_consistency_mean': np.mean(metrics['flow_consistency']),
        'intensity_variance_mean': np.mean(metrics['intensity_variance']),
        'per_sample_metrics': metrics
    }

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

