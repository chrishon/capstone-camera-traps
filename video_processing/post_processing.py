import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

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

