import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate decay scheduler"""
    
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]


class DCBlock(nn.Module):
    """Dilated CNN-stack (DC) block with dense connections"""
    
    def __init__(self, in_channels, out_channels, dilation_rate=1, kernel_size=3, groups=1):
        super(DCBlock, self).__init__()
        
        padding = dilation_rate * (kernel_size - 1) // 2
        
        self.dilated_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation_rate, padding=padding, groups=groups, bias=False
        )
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        return torch.cat([out, x], dim=1)


class DDCMModule(nn.Module):
    """Dense Dilated Convolutions Merging Module"""
    
    def __init__(self, in_channels, base_channels=32, dilation_rates=[1, 2, 3, 5, 7, 9], groups=1):
        super(DDCMModule, self).__init__()
        
        self.dc_blocks = nn.ModuleList()
        current_in_channels = in_channels
        
        for dilation_rate in dilation_rates:
            block = DCBlock(current_in_channels, base_channels, dilation_rate, groups=groups)
            self.dc_blocks.append(block)
            current_in_channels += base_channels
        
        # Merging layer
        final_channels = in_channels + base_channels * len(dilation_rates)
        self.merge_conv = nn.Conv2d(final_channels, base_channels, kernel_size=1, bias=False)
        self.merge_bn = nn.BatchNorm2d(base_channels)
        self.merge_prelu = nn.PReLU()
    
    def forward(self, x):
        current_input = x
        for dc_block in self.dc_blocks:
            current_input = dc_block(current_input)
        
        out = self.merge_conv(current_input)
        out = self.merge_bn(out)
        out = self.merge_prelu(out)
        return out


class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction"""
    
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        else:
            backbone = models.resnet101(pretrained=pretrained)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.output_channels = 1024
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class DDCMNet(nn.Module):
    """Complete DDCM-Net for land cover classification"""
    
    def __init__(self, num_classes=6, backbone_name='resnet50', pretrained=True):
        super(DDCMNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Low-level encoder
        self.low_level_encoder = DDCMModule(
            in_channels=3, base_channels=3, 
            dilation_rates=[1, 2, 3, 5, 7, 9]
        )
        
        # Backbone
        self.backbone = ResNetBackbone(backbone_name, pretrained)
        
        # High-level decoders
        self.high_level_decoder1 = DDCMModule(
            in_channels=1024, base_channels=36,
            dilation_rates=[1, 2, 3, 4]
        )
        
        self.high_level_decoder2 = DDCMModule(
            in_channels=36, base_channels=18,
            dilation_rates=[1]
        )
        
        # Fusion and classification
        self.fusion_conv = nn.Conv2d(21, 64, kernel_size=3, padding=1)  # 3 + 18 = 21
        self.fusion_bn = nn.BatchNorm2d(64)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for newly added layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not any(backbone_module is m for backbone_module in self.backbone.modules()):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Low-level features path: Input -> DDCM -> 0.5 pool -> half resolution
        low_features = self.low_level_encoder(x)
        # Apply 0.5 pooling as specified in the paper
        low_features = F.interpolate(
            low_features, scale_factor=0.5, 
            mode='bilinear', align_corners=False
        )
        
        # High-level features path: Input -> Backbone -> DDCM1 -> 4x up -> DDCM2 -> 2x up -> half resolution
        high_features = self.backbone(x)
        high_decoded1 = self.high_level_decoder1(high_features)
        # Apply 4x upsampling as specified in the paper (32 -> 128)
        high_decoded1 = F.interpolate(
            high_decoded1, scale_factor=4, 
            mode='bilinear', align_corners=False
        )
        high_decoded2 = self.high_level_decoder2(high_decoded1)
        # Apply 2x upsampling to match low-level features (128 -> 256)
        high_decoded2 = F.interpolate(
            high_decoded2, scale_factor=2, 
            mode='bilinear', align_corners=False
        )
        
        # Both paths should be at half resolution for fusion
        fused = torch.cat([low_features, high_decoded2], dim=1)
        
        # Final prediction
        x = self.fusion_conv(fused)
        x = self.fusion_bn(x)
        x = self.fusion_relu(x)
        x = self.classifier(x)
        
        # Upsample back to original input size
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)


class DDCMTrainer:
    """Training wrapper for DDCM-Net with visualization"""
    
    def __init__(self, model, device='auto', class_names=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = model.to(self.device)
        
        self.class_names = class_names or [
            'Impervious', 'Building', 'Low_veg', 'Tree', 'Car', 'Clutter'
        ]
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_miou': [], 'val_miou': []
        }
    
    def compute_class_weights(self, dataloader, method='median_frequency'):
        """Compute class weights for balancing"""
        if method == 'median_frequency':
            return self._compute_median_frequency_weights(dataloader)
        else:
            return None
    
    def _compute_median_frequency_weights(self, dataloader):
        """Compute median frequency balancing weights"""
        print("Computing median frequency balancing weights...")
        class_counts = torch.zeros(self.model.num_classes)
        total_pixels = 0
        
        for _, targets in tqdm(dataloader, desc="Computing class frequencies"):
            targets = targets.to(self.device)
            for class_id in range(self.model.num_classes):
                class_counts[class_id] += (targets == class_id).sum().item()
            total_pixels += targets.numel()
        
        # Calculate frequencies
        frequencies = class_counts / total_pixels
        
        # Median frequency balancing: weight = median_freq / class_freq
        median_freq = torch.median(frequencies)
        weights = median_freq / (frequencies + 1e-8)  # Add small epsilon to avoid division by zero
        
        print(f"Class frequencies: {frequencies.numpy()}")
        print(f"Median frequency: {median_freq:.6f}")
        print(f"Class weights: {weights.numpy()}")
        
        return weights
    
    def compute_metrics(self, outputs, targets):
        """Compute accuracy and mIoU"""
        predictions = torch.argmax(outputs, dim=1)
        
        # Accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        
        # mIoU
        ious = []
        for class_id in range(self.model.num_classes):
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
            
            if target_mask.sum() == 0:  # No ground truth for this class
                if pred_mask.sum() == 0:  # No prediction either
                    ious.append(1.0)
                else:
                    ious.append(0.0)
            else:
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()
                ious.append((intersection / union).item())
        
        miou = np.mean(ious)
        return accuracy.item(), miou
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_miou = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Metrics
            acc, miou = self.compute_metrics(outputs, targets)
            
            total_loss += loss.item()
            total_acc += acc
            total_miou += miou
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.3f}',
                'mIoU': f'{miou:.3f}'
            })
        
        return total_loss/num_batches, total_acc/num_batches, total_miou/num_batches
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_miou = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                # Metrics
                acc, miou = self.compute_metrics(outputs, targets)
                
                total_loss += loss.item()
                total_acc += acc
                total_miou += miou
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{acc:.3f}',
                    'mIoU': f'{miou:.3f}'
                })
        
        return total_loss/num_batches, total_acc/num_batches, total_miou/num_batches
    
    def fit(self, train_loader, val_loader, epochs=50, lr=6.01e-5, weight_decay=2e-5, 
            class_weights=None, use_mfb=True, lr_scheduler='step'):
        """
        Train the model using best practices from the DDCM-Net paper:
        - Adam optimizer with AMSGrad
        - Weight decay 2e-5 applied only to weights (not biases/batch-norm)
        - Learning rate 8.5e-5/√2 ≈ 6.01e-5 for weights, 2x for biases
        - StepLR schedule: 0.85 decay every 15 epochs (default)
        - Alternative: Polynomial decay with power 0.9
        - Cross-entropy loss with median frequency balancing (MFB)
        
        Args:
            lr_scheduler: 'step' for StepLR (default) or 'poly' for polynomial decay
        """
        # Compute median frequency balancing weights if requested and not provided
        if use_mfb and class_weights is None:
            class_weights = self.compute_class_weights(train_loader, method='median_frequency')
        
        # Setup parameter groups with different weight decay and learning rates
        weight_params = []
        bias_params = []
        bn_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name:
                bias_params.append(param)
            elif 'bn' in name or 'norm' in name:
                bn_params.append(param)
            else:
                weight_params.append(param)
        
        # Parameter groups: weights with weight decay, biases with 2x LR, batch-norm without weight decay
        param_groups = [
            {'params': weight_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': bias_params, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': bn_params, 'lr': lr, 'weight_decay': 0.0}
        ]
        
        # Setup loss function with class weights if provided
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(param_groups, amsgrad=True)
        
        # Setup learning rate scheduler
        if lr_scheduler == 'poly':
            max_iter = epochs * len(train_loader)
            scheduler = PolynomialLR(optimizer, max_iter, power=0.9)
        else:  # default: step
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.85)
        
        best_miou = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc, train_miou = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_miou = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_miou'].append(train_miou)
            self.history['val_miou'].append(val_miou)
            
            # Print metrics
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, mIoU: {train_miou:.3f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, mIoU: {val_miou:.3f}")
            
            # Save best model
            if val_miou > best_miou:
                best_miou = val_miou
                self.save_model('best_model.pth')
                print(f"New best model saved! mIoU: {best_miou:.3f}")
        
        print(f"\nTraining completed! Best mIoU: {best_miou:.3f}")
        return self.history
    
    def plot_training_history(self, figsize=(15, 5)):
        """Plot training history"""
        if not self.history['train_loss']:
            print("No training history to plot")
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=2)
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # mIoU
        axes[2].plot(epochs, self.history['train_miou'], 'b-', label='Train', linewidth=2)
        axes[2].plot(epochs, self.history['val_miou'], 'r-', label='Validation', linewidth=2)
        axes[2].set_title('Mean IoU')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('mIoU')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, images):
        """Make predictions on images"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()
            
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            return predictions.cpu().numpy(), outputs.cpu().numpy()
    
    def visualize_predictions(self, dataloader, num_samples=4, figsize=(20, 5)):
        """Visualize predictions"""
        self.model.eval()
        
        colors = ['white', 'blue', 'cyan', 'green', 'yellow', 'red']
        
        fig, axes = plt.subplots(3, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        sample_count = 0
        with torch.no_grad():
            for images, targets in dataloader:
                if sample_count >= num_samples:
                    break
                
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                for i in range(min(images.shape[0], num_samples - sample_count)):
                    # Get single sample
                    img = images[i].cpu()
                    target = targets[i].cpu()
                    pred = predictions[i].cpu()
                    
                    # Denormalize image
                    mean = torch.tensor([0.485, 0.456, 0.406])
                    std = torch.tensor([0.229, 0.224, 0.225])
                    img = img * std[:, None, None] + mean[:, None, None]
                    img = torch.clamp(img, 0, 1)
                    
                    col = sample_count
                    
                    # Original image
                    axes[0, col].imshow(img.permute(1, 2, 0))
                    axes[0, col].set_title('Original')
                    axes[0, col].axis('off')
                    
                    # Ground truth
                    axes[1, col].imshow(target, cmap='tab10', vmin=0, vmax=5)
                    axes[1, col].set_title('Ground Truth')
                    axes[1, col].axis('off')
                    
                    # Prediction
                    axes[2, col].imshow(pred, cmap='tab10', vmin=0, vmax=5)
                    axes[2, col].set_title('Prediction')
                    axes[2, col].axis('off')
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'class_names': self.class_names
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.class_names = checkpoint.get('class_names', self.class_names)


def create_model(variant='base', num_classes=6, backbone='resnet50', pretrained=True):
    """Create DDCM-Net model"""
    return DDCMNet(num_classes, backbone, pretrained)


def create_trainer(model, device='auto', class_names=None):
    """Create DDCM trainer"""
    return DDCMTrainer(model, device, class_names)