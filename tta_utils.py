"""
Test Time Augmentation (TTA) utilities for DDCM-Net
Implements paper-compliant TTA with 448×448 patches, 100-pixel stride, and flipping transformations
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time


class TTATransforms:
    """Test Time Augmentation transformations"""
    
    @staticmethod
    def apply_tta_transforms(patch):
        """
        Apply TTA transformations: original, horizontal flip, vertical flip, both flips
        
        Args:
            patch: Input tensor of shape [1, 3, H, W]
        
        Returns:
            List of (name, transformed_patch) tuples
        """
        transforms = []
        
        # Original
        transforms.append(('original', patch))
        
        # Horizontal flip (mirroring)
        transforms.append(('hflip', torch.flip(patch, [3])))
        
        # Vertical flip
        transforms.append(('vflip', torch.flip(patch, [2])))
        
        # Both flips
        transforms.append(('hvflip', torch.flip(patch, [2, 3])))
        
        return transforms
    
    @staticmethod
    def reverse_tta_transforms(prediction, transform_name):
        """
        Reverse the TTA transformation on the prediction
        
        Args:
            prediction: Model output tensor [1, num_classes, H, W]
            transform_name: Name of the transformation to reverse
        
        Returns:
            Reversed prediction tensor
        """
        if transform_name == 'original':
            return prediction
        elif transform_name == 'hflip':
            return torch.flip(prediction, [3])
        elif transform_name == 'vflip':
            return torch.flip(prediction, [2])
        elif transform_name == 'hvflip':
            return torch.flip(prediction, [2, 3])
        else:
            return prediction


class TTAPredictor:
    """Test Time Augmentation predictor for DDCM-Net models"""
    
    def __init__(self, model, device='auto', num_classes=6):
        """
        Initialize TTA predictor
        
        Args:
            model: Trained DDCM-Net model
            device: Device to run inference on
            num_classes: Number of classes for segmentation
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.num_classes = num_classes
        self.transforms = TTATransforms()
        
    def predict_with_sliding_window_tta(self, image, patch_size=448, stride=100):
        """
        Apply TTA with sliding windows as described in the paper
        
        Args:
            image: Input image tensor [1, 3, H, W] or [3, H, W]
            patch_size: Size of sliding window patches (default: 448)
            stride: Stride for sliding window (default: 100)
        
        Returns:
            Final averaged prediction [1, num_classes, H, W]
        """
        self.model.eval()
        
        # Ensure input is 4D [1, 3, H, W]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        batch_size, channels, height, width = image.shape
        
        # Move to device
        image = image.to(self.device)
        
        # Initialize output canvas and count map
        prediction_canvas = torch.zeros(batch_size, self.num_classes, height, width, device=self.device)
        count_canvas = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        print(f"Applying TTA with {patch_size}×{patch_size} patches, stride={stride}")
        
        # Start timing
        start_time = time.time()
        
        # Calculate positions for sliding windows (ensuring full coverage)
        y_positions = []
        x_positions = []
        
        # Generate y positions
        for y in range(0, height - patch_size + 1, stride):
            y_positions.append(y)
        # Ensure we include the bottom edge
        if y_positions and y_positions[-1] + patch_size < height:
            y_positions.append(height - patch_size)
        elif not y_positions:
            y_positions.append(0)
        
        # Generate x positions
        for x in range(0, width - patch_size + 1, stride):
            x_positions.append(x)
        # Ensure we include the right edge
        if x_positions and x_positions[-1] + patch_size < width:
            x_positions.append(width - patch_size)
        elif not x_positions:
            x_positions.append(0)
        
        total_patches = len(y_positions) * len(x_positions)
        print(f"Processing {total_patches} patches ({len(y_positions)}×{len(x_positions)})")
        
        patch_count = 0
        
        # Sliding window extraction with complete coverage
        for y in y_positions:
            for x in x_positions:
                patch_count += 1
                
                # Extract patch (handle edge cases)
                y_end = min(y + patch_size, height)
                x_end = min(x + patch_size, width)
                patch = image[:, :, y:y_end, x:x_end]
                
                # Pad patch if needed (for edge cases)
                if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
                    pad_h = patch_size - patch.shape[2]
                    pad_w = patch_size - patch.shape[3]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Apply TTA transformations
                transforms = self.transforms.apply_tta_transforms(patch)
                
                patch_predictions = []
                
                # Process each transformation
                for transform_name, transformed_patch in transforms:
                    with torch.no_grad():
                        # Get model prediction
                        pred = self.model(transformed_patch)
                        
                        # Reverse the transformation on prediction
                        pred_reversed = self.transforms.reverse_tta_transforms(pred, transform_name)
                        patch_predictions.append(pred_reversed)
                
                # Average predictions from all TTA transformations
                avg_patch_pred = torch.stack(patch_predictions).mean(dim=0)
                
                # Remove padding if it was added
                if avg_patch_pred.shape[2] > (y_end - y) or avg_patch_pred.shape[3] > (x_end - x):
                    avg_patch_pred = avg_patch_pred[:, :, :(y_end - y), :(x_end - x)]
                
                # Add to prediction canvas
                prediction_canvas[:, :, y:y_end, x:x_end] += avg_patch_pred
                count_canvas[:, :, y:y_end, x:x_end] += 1
                
                # Progress update
                if patch_count % 50 == 0 or patch_count == total_patches:
                    print(f"Processed {patch_count}/{total_patches} patches")
        
        # Average overlapping predictions (with safety check for division by zero)
        epsilon = 1e-8
        final_prediction = prediction_canvas / (count_canvas + epsilon)
        
        # Verify complete coverage
        min_count = count_canvas.min().item()
        max_count = count_canvas.max().item()
        print(f"Coverage verification: min_count={min_count}, max_count={max_count}")
        
        if min_count == 0:
            print("Warning: Some pixels were not covered by any patches!")
            uncovered_mask = (count_canvas[0, 0] == 0).cpu().numpy()
            uncovered_pixels = uncovered_mask.sum()
            print(f"   Uncovered pixels: {uncovered_pixels}/{count_canvas.numel()}")
        
        # End timing
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"TTA inference completed in {inference_time:.2f} seconds!")
        
        return final_prediction, inference_time
    
    def predict_regular(self, image, patch_size=448, stride=100):
        """
        Regular prediction using sliding window without TTA
        
        Args:
            image: Input image tensor [1, 3, H, W] or [3, H, W]
            patch_size: Size of sliding window patches (default: 448)
            stride: Stride for sliding window (default: 100)
        
        Returns:
            Prediction tensor [1, num_classes, H, W]
        """
        self.model.eval()
        
        # Ensure input is 4D [1, 3, H, W]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        batch_size, channels, height, width = image.shape
        
        # Move to device
        image = image.to(self.device)
        
        # Start timing
        start_time = time.time()
        
        # For small images, use direct prediction
        if height <= patch_size and width <= patch_size:
            with torch.no_grad():
                prediction = self.model(image)
                # End timing
                end_time = time.time()
                inference_time = end_time - start_time
                return prediction, inference_time
        
        # Initialize output canvas and count map for sliding window
        prediction_canvas = torch.zeros(batch_size, self.num_classes, height, width, device=self.device)
        count_canvas = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        # Calculate positions for sliding windows (ensuring full coverage)
        y_positions = []
        x_positions = []
        
        # Generate y positions
        for y in range(0, height - patch_size + 1, stride):
            y_positions.append(y)
        # Ensure we include the bottom edge
        if y_positions and y_positions[-1] + patch_size < height:
            y_positions.append(height - patch_size)
        elif not y_positions:
            y_positions.append(0)
        
        # Generate x positions
        for x in range(0, width - patch_size + 1, stride):
            x_positions.append(x)
        # Ensure we include the right edge
        if x_positions and x_positions[-1] + patch_size < width:
            x_positions.append(width - patch_size)
        elif not x_positions:
            x_positions.append(0)
        
        # Sliding window extraction and prediction
        for y in y_positions:
            for x in x_positions:
                # Extract patch (handle edge cases)
                y_end = min(y + patch_size, height)
                x_end = min(x + patch_size, width)
                patch = image[:, :, y:y_end, x:x_end]
                
                # Pad patch if needed (for edge cases)
                if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
                    pad_h = patch_size - patch.shape[2]
                    pad_w = patch_size - patch.shape[3]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Get model prediction (no TTA transformations)
                with torch.no_grad():
                    pred = self.model(patch)
                
                # Remove padding if it was added
                if pred.shape[2] > (y_end - y) or pred.shape[3] > (x_end - x):
                    pred = pred[:, :, :(y_end - y), :(x_end - x)]
                
                # Add to prediction canvas
                prediction_canvas[:, :, y:y_end, x:x_end] += pred
                count_canvas[:, :, y:y_end, x:x_end] += 1
        
        # Average overlapping predictions (with safety check for division by zero)
        epsilon = 1e-8
        final_prediction = prediction_canvas / (count_canvas + epsilon)
        
        # End timing
        end_time = time.time()
        inference_time = end_time - start_time
        
        return final_prediction, inference_time


class TTAEvaluator:
    """Comprehensive TTA evaluation for model comparison"""
    
    def __init__(self, class_names=None):
        """
        Initialize TTA evaluator
        
        Args:
            class_names: List of class names for visualization
        """
        self.class_names = class_names or [
            'Impervious surfaces', 'Building', 'Low vegetation', 
            'Tree', 'Car', 'Clutter/background'
        ]
    
    def visualize_results(self, original_image, ground_truth, tta_prediction, 
                         regular_prediction=None, model_name="Model"):
        """
        Visualize TTA results compared to ground truth and regular prediction
        
        Args:
            original_image: Original image tensor [1, 3, H, W]
            ground_truth: Ground truth tensor [H, W] 
            tta_prediction: TTA prediction tensor [1, num_classes, H, W]
            regular_prediction: Optional regular prediction for comparison
            model_name: Name of the model for titles
        """
        # Convert predictions to class labels
        tta_pred_labels = torch.argmax(tta_prediction, dim=1)[0].cpu()
        
        if original_image.dim() == 4:
            img = original_image[0]
        else:
            img = original_image
            
        img = img.permute(1, 2, 0).cpu().numpy()
        
        # Create visualization
        if regular_prediction is not None:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            reg_pred_labels = torch.argmax(regular_prediction, dim=1)[0].cpu()
            
            axes[3].imshow(reg_pred_labels, cmap='tab10', vmin=0, vmax=5)
            axes[3].set_title(f'{model_name} Regular')
            axes[3].axis('off')
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(ground_truth, cmap='tab10', vmin=0, vmax=5)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # TTA prediction
        axes[2].imshow(tta_pred_labels, cmap='tab10', vmin=0, vmax=5)
        axes[2].set_title(f'{model_name} TTA')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_metrics(self, prediction, ground_truth, num_classes=6):
        """
        Calculate accuracy and IoU metrics
        
        Args:
            prediction: Prediction labels tensor
            ground_truth: Ground truth tensor
            num_classes: Number of classes
        
        Returns:
            Tuple of (accuracy, mean_iou, class_ious)
        """
        # Overall accuracy
        accuracy = (prediction == ground_truth).float().mean().item()
        
        # Calculate IoU for each class
        class_ious = []
        for c in range(num_classes):
            pred_c = (prediction == c)
            target_c = (ground_truth == c)
            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            if union > 0:
                class_ious.append((intersection / union).item())
            else:
                class_ious.append(0.0)
        
        mean_iou = np.mean(class_ious)
        return accuracy, mean_iou, class_ious
    
    def evaluate_model_with_tta(self, model, test_loader, model_name="Model", 
                               max_images=None, visualize=True, patch_size=448, stride=100, crop_large_images=True):
        """
        Comprehensive evaluation of a model with TTA
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            model_name: Name of the model for reporting
            max_images: Maximum number of images to process
            visualize: Whether to show visualizations
            patch_size: TTA patch size
            stride: TTA stride
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"TTA Evaluation: {model_name}")
        print(f"{'='*60}")
        
        # Create TTA predictor
        tta_predictor = TTAPredictor(model, num_classes=len(self.class_names))
        
        # Get test dataset
        test_dataset = test_loader.dataset
        available_images = test_dataset.get_available_images()
        
        if max_images is not None and max_images < len(available_images):
            available_images = available_images[:max_images]
            print(f"Processing {max_images} images for demonstration")
        
        # Initialize metrics storage
        all_tta_accuracies = []
        all_reg_accuracies = []
        all_tta_ious = []
        all_reg_ious = []
        all_class_tta_accuracies = [[] for _ in range(len(self.class_names))]
        all_class_reg_accuracies = [[] for _ in range(len(self.class_names))]
        all_tta_times = []
        all_reg_times = []
        
        # Process each image
        for img_idx, img_name in enumerate(available_images):
            print(f"\nProcessing {img_idx + 1}/{len(available_images)}: {img_name}")
            
            try:
                # Load full-resolution image and label
                full_image, full_label = test_dataset.get_full_image(img_idx)

                _, _, height, width = full_image.shape
                if crop_large_images and (height > 1500 or width > 1500):
                    print(f"  Cropping large image ({height}×{width}) to 1500×1500")
                    crop_h = min(1500, height)
                    crop_w = min(1500, width)
                    start_h = (height - crop_h) // 2
                    start_w = (width - crop_w) // 2
                    
                    full_image = full_image[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
                    full_label = full_label[start_h:start_h+crop_h, start_w:start_w+crop_w]
                
                # Apply TTA
                tta_prediction, tta_time = tta_predictor.predict_with_sliding_window_tta(
                    full_image, patch_size=patch_size, stride=stride
                )
                
                # Regular prediction for comparison
                regular_prediction, reg_time = tta_predictor.predict_regular(
                    full_image, patch_size=patch_size, stride=stride
                )
                
                # Visualize if enabled
                if visualize:
                    self.visualize_results(
                        full_image, full_label, tta_prediction, 
                        regular_prediction, model_name
                    )
                
                # Calculate metrics
                tta_pred_labels = torch.argmax(tta_prediction, dim=1)[0].cpu()
                reg_pred_labels = torch.argmax(regular_prediction, dim=1)[0].cpu()
                
                tta_acc, tta_miou, tta_class_ious = self.calculate_metrics(
                    tta_pred_labels, full_label, len(self.class_names)
                )
                reg_acc, reg_miou, reg_class_ious = self.calculate_metrics(
                    reg_pred_labels, full_label, len(self.class_names)
                )
                
                # Store results
                all_tta_accuracies.append(tta_acc)
                all_reg_accuracies.append(reg_acc)
                all_tta_ious.append(tta_miou)
                all_reg_ious.append(reg_miou)
                all_tta_times.append(tta_time)
                all_reg_times.append(reg_time)
                
                # Store class-wise results
                for class_id in range(len(self.class_names)):
                    mask = (full_label == class_id)
                    if mask.sum() > 0:
                        tta_class_acc = (tta_pred_labels[mask] == class_id).float().mean().item()
                        reg_class_acc = (reg_pred_labels[mask] == class_id).float().mean().item()
                        
                        all_class_tta_accuracies[class_id].append(tta_class_acc)
                        all_class_reg_accuracies[class_id].append(reg_class_acc)
                
                print(f"  Regular: Acc={reg_acc:.3f}, mIoU={reg_miou:.3f}, Time={reg_time:.2f}s")
                print(f"  TTA: Acc={tta_acc:.3f}, mIoU={tta_miou:.3f}, Time={tta_time:.2f}s")
                print(f"  Improvement: Acc={tta_acc - reg_acc:+.3f}, mIoU={tta_miou - reg_miou:+.3f}")
                print(f"  Time Overhead: {tta_time - reg_time:+.2f}s ({tta_time/reg_time:.1f}x slower)" if reg_time > 0 else "")
                
            except Exception as e:
                print(f"  Error processing {img_name}: {e}")
                continue
        
        # Calculate per-class accuracy averages
        avg_class_tta_accuracies = []
        avg_class_reg_accuracies = []
        
        for class_id in range(len(self.class_names)):
            if all_class_tta_accuracies[class_id]:  # Only if we have data for this class
                avg_class_tta_accuracies.append(np.mean(all_class_tta_accuracies[class_id]))
                avg_class_reg_accuracies.append(np.mean(all_class_reg_accuracies[class_id]))
            else:
                avg_class_tta_accuracies.append(0.0)
                avg_class_reg_accuracies.append(0.0)
        
        # Aggregate results
        if all_tta_accuracies:
            results = {
                'model_name': model_name,
                'num_images': len(all_tta_accuracies),
                'avg_tta_acc': np.mean(all_tta_accuracies),
                'avg_reg_acc': np.mean(all_reg_accuracies),
                'avg_tta_miou': np.mean(all_tta_ious),
                'avg_reg_miou': np.mean(all_reg_ious),
                'acc_improvement': np.mean(all_tta_accuracies) - np.mean(all_reg_accuracies),
                'miou_improvement': np.mean(all_tta_ious) - np.mean(all_reg_ious),
                'all_tta_accuracies': all_tta_accuracies,
                'all_reg_accuracies': all_reg_accuracies,
                'all_tta_ious': all_tta_ious,
                'all_reg_ious': all_reg_ious,
                'avg_class_tta_accuracies': avg_class_tta_accuracies,
                'avg_class_reg_accuracies': avg_class_reg_accuracies,
                'all_class_tta_accuracies': all_class_tta_accuracies,
                'all_class_reg_accuracies': all_class_reg_accuracies,
                'avg_tta_time': np.mean(all_tta_times),
                'avg_reg_time': np.mean(all_reg_times),
                'total_tta_time': np.sum(all_tta_times),
                'total_reg_time': np.sum(all_reg_times),
                'all_tta_times': all_tta_times,
                'all_reg_times': all_reg_times,
                'avg_time_overhead': np.mean(all_tta_times) - np.mean(all_reg_times),
                'avg_speedup_factor': np.mean(all_tta_times) / np.mean(all_reg_times) if np.mean(all_reg_times) > 0 else 0
            }
            
            # Print summary
            print(f"\n{'='*40}")
            print(f"SUMMARY - {model_name}")
            print(f"{'='*40}")
            print(f"Images processed: {results['num_images']}")
            print(f"Average Regular Accuracy: {results['avg_reg_acc']:.4f}")
            print(f"Average TTA Accuracy: {results['avg_tta_acc']:.4f}")
            print(f"Accuracy Improvement: {results['acc_improvement']:+.4f}")
            print(f"Average Regular mIoU: {results['avg_reg_miou']:.4f}")
            print(f"Average TTA mIoU: {results['avg_tta_miou']:.4f}")
            print(f"mIoU Improvement: {results['miou_improvement']:+.4f}")
            
            # Print timing statistics
            print(f"\nTiming Statistics:")
            print(f"Average Regular Time: {results['avg_reg_time']:.2f}s per image")
            print(f"Average TTA Time: {results['avg_tta_time']:.2f}s per image")
            print(f"Time Overhead: {results['avg_time_overhead']:+.2f}s ({results['avg_speedup_factor']:.1f}x slower)")
            print(f"Total Processing Time: Regular={results['total_reg_time']:.1f}s, TTA={results['total_tta_time']:.1f}s")
            
            # Print per-class accuracy summary
            print(f"\nPer-Class Accuracy (Durchschnitt):")
            for class_id, class_name in enumerate(self.class_names):
                if avg_class_reg_accuracies[class_id] > 0:  # Only show classes with data
                    reg_acc = avg_class_reg_accuracies[class_id]
                    tta_acc = avg_class_tta_accuracies[class_id]
                    improvement = tta_acc - reg_acc
                    print(f"  {class_id}: {class_name:<20} Reg: {reg_acc:.3f} -> TTA: {tta_acc:.3f} ({improvement:+.3f})")
            
            return results
        else:
            print("No images were successfully processed.")
            return None
    
    def compare_models_with_tta(self, models_dict, test_loader, max_images=None, 
                               visualize=False, patch_size=448, stride=100, crop_large_images=True):
        """
        Compare multiple models using TTA
        
        Args:
            models_dict: Dictionary of {'model_name': model} pairs
            test_loader: Test data loader
            max_images: Maximum number of images to process
            visualize: Whether to show visualizations
            patch_size: TTA patch size
            stride: TTA stride
        
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*80}")
        print(f"MULTI-MODEL TTA COMPARISON")
        print(f"{'='*80}")
        
        all_results = {}
        
        # Evaluate each model
        for model_name, model in models_dict.items():
            results = self.evaluate_model_with_tta(
                model, test_loader, model_name, max_images, 
                visualize, patch_size, stride, crop_large_images
            )
            if results:
                all_results[model_name] = results
        
        # Print comparison summary
        if len(all_results) > 1:
            print(f"\n{'='*80}")
            print(f"MODEL COMPARISON SUMMARY")
            print(f"{'='*80}")
            
            print(f"{'Model':<20} {'Reg Acc':<8} {'TTA Acc':<8} {'Acc Δ':<8} {'Reg mIoU':<8} {'TTA mIoU':<8} {'mIoU Δ':<8} {'Reg Time':<10} {'TTA Time':<10} {'Speedup':<8}")
            print("-" * 120)
            
            for model_name, results in all_results.items():
                print(f"{model_name:<20} {results['avg_reg_acc']:<8.3f} {results['avg_tta_acc']:<8.3f} "
                      f"{results['acc_improvement']:<8.3f} {results['avg_reg_miou']:<8.3f} "
                      f"{results['avg_tta_miou']:<8.3f} {results['miou_improvement']:<8.3f} "
                      f"{results['avg_reg_time']:<10.2f} {results['avg_tta_time']:<10.2f} {results['avg_speedup_factor']:<8.1f}")
        
        return all_results


def load_model_for_tta(model_path, variant='base', num_classes=6, backbone='resnet50', device='auto'):
    """
    Utility function to load a model for TTA evaluation
    
    Args:
        model_path: Path to the saved model
        variant: Model variant ('base' or 'enhanced')
        num_classes: Number of classes
        backbone: Backbone architecture
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    from model import create_model, create_trainer
    
    # Create model
    model = create_model(variant=variant, num_classes=num_classes, backbone=backbone)
    
    # Create trainer to load weights
    trainer = create_trainer(model, device=device)
    
    # Load weights if model file exists
    if os.path.exists(model_path):
        trainer.load_model(model_path)
        print(f"Loaded model from: {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
    
    return model