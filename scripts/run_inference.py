"""
Main Inference Script for Drone Perception.
Processes video files and visualizes the results of our deep perception models.
"""

import cv2
import torch
import numpy as np
import time
import argparse
import sys
from pathlib import Path

# Enhance path to allow imports from src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.optical_flow import FlowEstimator
from src.depth_estimation import DepthEstimator

def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    """
    Convert dense optical flow to a BGR image using HSV color representation.
    """
    # Flow shape is (2, H, W), we need (H, W, 2)
    flow = flow.transpose(1, 2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def preprocess_for_flow(frame_rgb: np.ndarray) -> torch.Tensor:
    """
    Prepares an RGB numpy image for the flow model.
    Expected output: torch.Tensor of shape (1, C, H, W)
    """
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

def main(video_path: str, realtime: bool):
    print("Loading deep CV models...")
    depth_estimator = DepthEstimator()
    flow_estimator = FlowEstimator()

    print(f"Opening video feed from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        print("Please ensure the file exists and is a valid video format.")
        return

    ret, prev_frame_bgr = cap.read()
    if not ret:
        print("Error: Failed to read the first frame from the video.")
        return

    prev_frame_rgb = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle 0 or NaN
        fps = 30.0

    print("Starting inference loop. Press 'q' or 'ESC' to exit.")
    while True:
        start_time = time.time()
        
        ret, frame_bgr = cap.read()
        if not ret:
            print("End of video reached.")
            break
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Depth Estimation (MiDaS)
        depth_tensor = depth_estimator.estimate(frame_rgb)
        depth_np = depth_tensor.cpu().numpy()[0] # Returns shape (H, W)
        
        # Normalize depth map for visualization 
        depth_norm = cv2.normalize(depth_np, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        
        # 2. Optical Flow Estimation (RAFT)
        t1 = preprocess_for_flow(prev_frame_rgb)
        t2 = preprocess_for_flow(frame_rgb)
        
        flow_tensor = flow_estimator.estimate(t1, t2)
        flow_np = flow_tensor.cpu().numpy()[0] # Returns shape (2, H, W)
        
        flow_bgr = flow_to_bgr(flow_np)
        
        # 3. Safety Mask Calculation
        # Heuristic: pixels that are close (high depth) AND moving fast (high flow magnitude)
        flow_mag, _ = cv2.cartToPolar(flow_np[0], flow_np[1])
        
        depth_metric = cv2.normalize(depth_np, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        flow_metric = cv2.normalize(flow_mag, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        
        # Define thresholds: > 0.7 indicates proximity (MiDaS higher = closer), > 0.4 indicates strong motion
        close_mask = depth_metric > 0.7
        moving_mask = flow_metric > 0.4
        
        safety_mask_bool = np.logical_and(close_mask, moving_mask)
        
        # Render safety mask overlay
        safety_mask_vis = np.zeros_like(frame_bgr)
        safety_mask_vis[safety_mask_bool] = [0, 0, 255] # Red overlay for warning zones
        
        # Bottom-Right output fuses the original image with the highlighted alert pixels
        safety_overlay = cv2.addWeighted(frame_bgr, 0.65, safety_mask_vis, 0.8, 0)
        
        # 4. Rendering the 2x2 Grid Layout
        h, w = frame_bgr.shape[:2]
        
        # Ensure identical sizes for vstack and hstack
        depth_colored = cv2.resize(depth_colored, (w, h))
        flow_bgr = cv2.resize(flow_bgr, (w, h))
        
        # Top row: Raw Input | Optical Flow Vector Field
        top_row = np.hstack((frame_bgr, flow_bgr))
        # Bottom row: Contextual Depth Map | Fused Safety Mask
        bottom_row = np.hstack((depth_colored, safety_overlay))
        
        # Assemble Final View
        combined_grid = np.vstack((top_row, bottom_row))
        
        # Scale grid proportionally if vertical resolution is too large for comfortable viewing
        target_height = 800
        if combined_grid.shape[0] > target_height:
            scale = target_height / combined_grid.shape[0]
            combined_grid = cv2.resize(combined_grid, (int(combined_grid.shape[1] * scale), target_height))
            
        cv2.imshow("Drone Perception Engine [Real-Time]", combined_grid)
        
        # Carry over the current state for the next differential calculation
        prev_frame_rgb = frame_rgb.copy()
        
        # 5. Handle timing and conditional Frame Skipping (Real-time Mode)
        processing_time = time.time() - start_time
        
        if realtime:
            # Drop frames to keep streaming loop in sync with real-time video pace
            frames_to_skip = int(processing_time * fps)
            if frames_to_skip > 0:
                for _ in range(frames_to_skip):
                    cap.grab()
        
        # Standard exit check
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate visual perception primitives on a video feed.")
    parser.add_argument("--video", type=str, default="data/sample_drone_video.mp4", 
                        help="Path to the drone mp4 video file. Default reads from /data.")
    parser.add_argument("--realtime", action="store_true", 
                        help="Enable dropping frames to maintain real-time latency.")
    args = parser.parse_args()
    
    # Resolve default data directory dynamically
    video_target = args.video
    if video_target == "data/sample_drone_video.mp4":
        root = Path(__file__).resolve().parent.parent
        video_target = str(root / "data" / "sample_drone_video.mp4")
        
    main(video_target, args.realtime)
