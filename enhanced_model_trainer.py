import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0, ResNet50_Weights, EfficientNet_B0_Weights
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch.nn.functional as F

class EnhancedCandlestickDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, use_technical_features=True):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.use_technical_features = use_technical_features
        
        # Enhanced technical feature columns
        self.technical_features = [
            'rsi', 'macd', 'bb_position', 'stoch_k', 'volume_ratio',
            'volatility', 'distance_to_support', 'distance_to_resistance',
            'vpt', 'bb_squeeze', 'rsi_momentum', 'macd_momentum',
            'support_strength', 'resistance_strength', 'volatility_regime', 'trend_regime'
        ]
        
        # Prepare technical features if available
        if use_technical_features and all(col in self.labels_df.columns for col in self.technical_features):
            # Handle NaN values
            self.labels_df[self.technical_features] = self.labels_df[self.technical_features].fillna(0)
            
            # Normalize technical features
            self.scaler = StandardScaler()
            self.technical_data = self.scaler.fit_transform(
                self.labels_df[self.technical_features].values
            )
        else:
            self.use_technical_features = False
            self.technical_data = None
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        if self.use_technical_features and self.technical_data is not None:
            technical_features = torch.FloatTensor(self.technical_data[idx])
            return image, technical_features, label
        else:
            return image, label

class AdaptiveFusion(nn.Module):
    """Adaptive fusion mechanism for combining visual and technical features"""
    def __init__(self, visual_dim, technical_dim):
        super(AdaptiveFusion, self).__init__()
        self.visual_gate = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 4),
            nn.ReLU(),
            nn.Linear(visual_dim // 4, 1),
            nn.Sigmoid()
        )
        self.technical_gate = nn.Sequential(
            nn.Linear(technical_dim, technical_dim // 2),
            nn.ReLU(),
            nn.Linear(technical_dim // 2, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(visual_dim + technical_dim, visual_dim + technical_dim)
        
    def forward(self, visual_feats, technical_feats):
        v_gate = self.visual_gate(visual_feats)
        t_gate = self.technical_gate(technical_feats)
        
        gated_visual = visual_feats * v_gate
        gated_technical = technical_feats * t_gate
        
        combined = torch.cat([gated_visual, gated_technical], dim=1)
        return self.fusion(combined)

class MultiHeadAttention(nn.Module):
    """Multi-head attention for feature enhancement"""
    def __init__(self, d_model, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, combined_features):
        # Reshape for attention
        x = combined_features.unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.squeeze(1)
        
        # Residual connection and layer norm
        return self.layer_norm(combined_features + attn_output)

class EnhancedMultiModalCNN(nn.Module):
    def __init__(self, num_classes=2, num_technical_features=16, use_technical=True, backbone='resnet50'):
        super(EnhancedMultiModalCNN, self).__init__()
        
        self.use_technical = use_technical
        
        # Image feature extractor with updated weights parameter
        if backbone == 'resnet50':
            self.image_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            image_features = self.image_backbone.fc.in_features
            self.image_backbone.fc = nn.Identity()  # Remove final layer
        elif backbone == 'efficientnet':
            self.image_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            image_features = self.image_backbone.classifier[1].in_features
            self.image_backbone.classifier = nn.Identity()
        
        # Enhanced image feature processing
        self.image_processor = nn.Sequential(
            nn.Linear(image_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Enhanced technical feature processing
        if use_technical:
            self.technical_processor = nn.Sequential(
                nn.Linear(num_technical_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            
            # Adaptive fusion
            self.adaptive_fusion = AdaptiveFusion(256, 32)
            combined_features = 256 + 32
            
            # Multi-head attention
            self.attention = MultiHeadAttention(combined_features, num_heads=4)
        else:
            combined_features = 256
        
        # Enhanced final classifier with more layers
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, image, technical_features=None):
        # Extract image features
        image_feats = self.image_backbone(image)
        image_feats = self.image_processor(image_feats)
        
        if self.use_technical and technical_features is not None:
            # Extract technical features
            tech_feats = self.technical_processor(technical_features)
            
            # Adaptive fusion
            combined = self.adaptive_fusion(image_feats, tech_feats)
            
            # Apply attention
            combined = self.attention(combined)
            
            # Final classification
            output = self.classifier(combined)
        else:
            output = self.classifier(image_feats)
        
        return output

class TradingFocalLoss(nn.Module):
    """Enhanced Focal Loss with class weights and trading cost consideration"""
    def __init__(self, alpha=None, gamma=2, reduction='mean', transaction_cost=0.001):
        super(TradingFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.transaction_cost = transaction_cost
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            ce_loss = ce_loss * at
        
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        # Add trading cost penalty for frequent trading
        pred_classes = torch.argmax(inputs, dim=1)
        if len(pred_classes) > 1:
            trade_changes = (pred_classes[1:] != pred_classes[:-1]).float()
            cost_penalty = trade_changes.mean() * self.transaction_cost
            focal_loss = focal_loss + cost_penalty
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedCandlestickTrainer:
    def __init__(self, data_dir="data", model_save_path="enhanced_model"):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(model_save_path, exist_ok=True)
        
        print(f"Using device: {self.device}")
        
    def get_enhanced_transforms(self):
        """Get enhanced transforms with financial-specific augmentations"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),  # Reduced for financial data
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
        
    def prepare_data(self, batch_size=32, test_size=0.2, use_technical=True):
        """Prepare enhanced data loaders with optimized settings"""
        
        train_transform, val_transform = self.get_enhanced_transforms()
        
        # Read labels and split - use binary classification data
        labels_file = os.path.join(self.data_dir, "binary_labels.csv")
        chart_dir = os.path.join(self.data_dir, "binary_charts")
        
        labels_df = pd.read_csv(labels_file)
        
        # Stratified split to maintain class distribution
        train_df, val_df = train_test_split(
            labels_df, 
            test_size=test_size, 
            stratify=labels_df['label'], 
            random_state=42
        )
        
        # Save split data
        train_df.to_csv(os.path.join(self.data_dir, "enhanced_train_labels.csv"), index=False)
        val_df.to_csv(os.path.join(self.data_dir, "enhanced_val_labels.csv"), index=False)
        
        # Create datasets
        train_dataset = EnhancedCandlestickDataset(
            os.path.join(self.data_dir, "enhanced_train_labels.csv"),
            chart_dir,
            transform=train_transform,
            use_technical_features=use_technical
        )
        
        val_dataset = EnhancedCandlestickDataset(
            os.path.join(self.data_dir, "enhanced_val_labels.csv"),
            chart_dir,
            transform=val_transform,
            use_technical_features=use_technical
        )
        
        # Optimized data loaders for CPU/GPU (Windows compatibility)
        num_workers = 0  # Set to 0 for Windows compatibility
        pin_memory = self.device.type == 'cuda'
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Using technical features: {use_technical}")
        
        # Print class distribution
        train_dist = Counter(train_df['label'])
        print(f"Training label distribution: {train_dist}")
        
        return train_loader, val_loader, train_dataset.use_technical_features
    
    def get_class_weights(self, train_loader):
        """Calculate class weights for handling imbalance"""
        all_labels = []
        for batch in train_loader:
            if len(batch) == 3:  # With technical features
                _, _, labels = batch
            else:  # Without technical features
                _, labels = batch
            all_labels.extend(labels.numpy())
        
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(all_labels), 
            y=all_labels
        )
        
        return torch.FloatTensor(class_weights)
    
    def train_model(self, num_epochs=50, learning_rate=0.001, batch_size=32, 
                   use_technical=True, backbone='resnet50', use_focal_loss=True):
        """Enhanced training with all optimizations"""
        
        # Prepare data
        train_loader, val_loader, technical_available = self.prepare_data(
            batch_size, use_technical=use_technical
        )
        
        # Initialize enhanced model
        model = EnhancedMultiModalCNN(
            num_classes=2,
            num_technical_features=16,  # Updated for enhanced features
            use_technical=technical_available,
            backbone=backbone
        )
        model = model.to(self.device)
        
        # Enhanced loss with class weights
        if use_focal_loss:
            class_weights = self.get_class_weights(train_loader).to(self.device)
            criterion = TradingFocalLoss(alpha=class_weights, gamma=2, transaction_cost=0.001)
        else:
            class_weights = self.get_class_weights(train_loader).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Enhanced optimizer with different learning rates for different parts
        image_params = list(model.image_backbone.parameters()) + list(model.image_processor.parameters())
        other_params = [p for p in model.parameters() if not any(p is ip for ip in image_params)]
        
        optimizer = optim.AdamW([
            {'params': image_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained backbone
            {'params': other_params, 'lr': learning_rate}
        ], weight_decay=1e-4)
        
        # Enhanced learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[learning_rate * 0.5, learning_rate * 5],  # Different max rates
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_f1_scores = []
        best_val_f1 = 0.0
        best_val_acc = 0.0
        patience_counter = 0
        patience = 15  # Increased patience
        
        print("Starting enhanced training...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                if technical_available:
                    data, technical_features, target = batch_data
                    technical_features = technical_features.to(self.device)
                else:
                    data, target = batch_data
                    technical_features = None
                
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data, technical_features)
                loss = criterion(output, target)
                loss.backward()
                
                # Enhanced gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 10 == 0:
                    current_lr = scheduler.get_last_lr()
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, LRs: {[f"{lr:.6f}" for lr in current_lr]}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_predictions = []
            all_val_targets = []
            
            with torch.no_grad():
                for batch_data in val_loader:
                    if technical_available:
                        data, technical_features, target = batch_data
                        technical_features = technical_features.to(self.device)
                    else:
                        data, target = batch_data
                        technical_features = None
                    
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data, technical_features)
                    val_loss += criterion(output, target).item()
                    
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    all_val_predictions.extend(predicted.cpu().numpy())
                    all_val_targets.extend(target.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            val_f1 = f1_score(all_val_targets, all_val_predictions, average='weighted')
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print(f'  Val F1 Score: {val_f1:.4f}')
            print('-' * 50)
            
            # Save best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_accuracy
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'val_f1_score': val_f1,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'val_f1_scores': val_f1_scores,
                    'use_technical': technical_available,
                    'backbone': backbone,
                    'class_weights': class_weights.cpu()
                }, os.path.join(self.model_save_path, 'best_enhanced_model.pth'))
                
                print(f"New best model saved! Val F1: {val_f1:.4f}, Val Acc: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Save final model
        torch.save(model.state_dict(), os.path.join(self.model_save_path, 'final_enhanced_model.pth'))
        
        # Plot training history
        self.plot_enhanced_training_history(train_losses, val_losses, val_accuracies, val_f1_scores)
        
        # Evaluate model
        trading_metrics = self.evaluate_enhanced_model(model, val_loader, technical_available)
        
        print(f"\nBest validation F1 score: {best_val_f1:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return model, trading_metrics
    
    def calculate_trading_metrics(self, predictions, targets, returns=None):
        """Calculate trading-specific performance metrics"""
        metrics = {}
        
        # Basic classification metrics
        accuracy = np.mean(predictions == targets)
        f1 = f1_score(targets, predictions, average='weighted')
        
        metrics['accuracy'] = accuracy
        metrics['f1_score'] = f1
        
        # Trading-specific metrics if returns are available
        if returns is not None:
            # Convert predictions to positions (0: short, 1: long, 2: neutral)
            positions = np.where(predictions == 0, -1, np.where(predictions == 1, 1, 0))
            
            # Calculate portfolio returns
            portfolio_returns = positions[:-1] * returns[1:]  # Shift returns by 1
            
            if len(portfolio_returns) > 0:
                total_return = np.sum(portfolio_returns)
                annual_return = total_return * 252 / len(portfolio_returns)  # Annualized
                volatility = np.std(portfolio_returns) * np.sqrt(252)
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                
                # Maximum Drawdown
                cumulative_returns = np.cumsum(portfolio_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = cumulative_returns - running_max
                max_drawdown = np.min(drawdown)
                
                # Win rate
                win_rate = np.mean(portfolio_returns > 0)
                
                # Profit factor
                profits = portfolio_returns[portfolio_returns > 0]
                losses = portfolio_returns[portfolio_returns < 0]
                profit_factor = np.sum(profits) / abs(np.sum(losses)) if len(losses) > 0 else np.inf
                
                metrics.update({
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor
                })
        
        return metrics
    
    def plot_enhanced_training_history(self, train_losses, val_losses, val_accuracies, val_f1_scores):
        """Plot comprehensive training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
        ax1.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot F1 score
        ax3.plot(val_f1_scores, label='Validation F1 Score', color='purple', linewidth=2)
        ax3.set_title('Model F1 Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot comprehensive overview
        ax4.plot(train_losses, label='Training Loss', alpha=0.7, color='blue')
        ax4.plot(val_losses, label='Validation Loss', alpha=0.7, color='red')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(val_accuracies, label='Val Accuracy', color='green', alpha=0.8)
        ax4_twin.plot([x*100 for x in val_f1_scores], label='Val F1*100', color='purple', alpha=0.8)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='blue')
        ax4_twin.set_ylabel('Accuracy/F1*100', color='green')
        ax4.set_title('Training Overview', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'enhanced_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Enhanced training history saved to {self.model_save_path}/enhanced_training_history.png")
        plt.close()
    
    def evaluate_enhanced_model(self, model, val_loader, technical_available):
        """Comprehensive model evaluation with trading metrics"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if technical_available:
                    data, technical_features, target = batch_data
                    technical_features = technical_features.to(self.device)
                else:
                    data, target = batch_data
                    technical_features = None
                
                data, target = data.to(self.device), target.to(self.device)
                output = model(data, technical_features)
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Classification report
        class_names = ['Sell', 'Buy']
        report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=class_names,
            digits=4
        )
        print("\nDetailed Classification Report:")
        print("=" * 50)
        print(report)
        
        # Enhanced confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Enhanced Model Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'enhanced_confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Enhanced confusion matrix saved to {self.model_save_path}/enhanced_confusion_matrix.png")
        plt.close()
        
        # Calculate trading metrics
        trading_metrics = self.calculate_trading_metrics(
            np.array(all_predictions), 
            np.array(all_targets)
        )
        
        # Per-class metrics
        print("\nPer-Class Detailed Metrics:")
        print("=" * 50)
        for i, class_name in enumerate(class_names):
            class_mask = np.array(all_targets) == i
            if np.sum(class_mask) > 0:
                class_predictions = np.array(all_predictions)[class_mask]
                class_accuracy = np.mean(class_predictions == i) * 100
                class_count = np.sum(class_mask)
                precision = cm[i, i] / np.sum(cm[:, i]) * 100 if np.sum(cm[:, i]) > 0 else 0
                recall = cm[i, i] / np.sum(cm[i, :]) * 100 if np.sum(cm[i, :]) > 0 else 0
                print(f"{class_name:>4}: Acc={class_accuracy:5.2f}%, Prec={precision:5.2f}%, "
                      f"Rec={recall:5.2f}%, Count={class_count:4d}")
        
        # Overall metrics
        overall_accuracy = trading_metrics['accuracy'] * 100
        overall_f1 = trading_metrics['f1_score']
        
        print(f"\nOverall Enhanced Metrics:")
        print("=" * 30)
        print(f"Accuracy: {overall_accuracy:.2f}%")
        print(f"Weighted F1 Score: {overall_f1:.4f}")
        
        return trading_metrics
    
    def cross_validate(self, k_folds=5, num_epochs=30, learning_rate=0.001, 
                      batch_size=32, use_technical=True):
        """Enhanced k-fold cross validation"""
        print(f"Starting {k_folds}-fold cross validation...")
        
        # Read labels - use binary classification data
        labels_file = os.path.join(self.data_dir, "binary_labels.csv")
        chart_dir = os.path.join(self.data_dir, "binary_charts")
        labels_df = pd.read_csv(labels_file)
        
        # Initialize cross validation
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(labels_df, labels_df['label'])):
            print(f"\n{'='*20} Fold {fold+1}/{k_folds} {'='*20}")
            
            # Create fold datasets
            train_fold = labels_df.iloc[train_idx]
            val_fold = labels_df.iloc[val_idx]
            
            # Save fold data
            train_fold.to_csv(os.path.join(self.data_dir, f"fold_{fold+1}_train.csv"), index=False)
            val_fold.to_csv(os.path.join(self.data_dir, f"fold_{fold+1}_val.csv"), index=False)
            
            # Create data transforms
            train_transform, val_transform = self.get_enhanced_transforms()
            
            # Create datasets
            train_dataset = EnhancedCandlestickDataset(
                os.path.join(self.data_dir, f"fold_{fold+1}_train.csv"),
                chart_dir, train_transform, use_technical
            )
            val_dataset = EnhancedCandlestickDataset(
                os.path.join(self.data_dir, f"fold_{fold+1}_val.csv"),
                chart_dir, val_transform, use_technical
            )
            
            # Create data loaders (Windows compatibility)
            num_workers = 0  # Set to 0 for Windows compatibility
            pin_memory = self.device.type == 'cuda'
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, pin_memory=pin_memory
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=pin_memory
            )
            
            # Initialize model
            model = EnhancedMultiModalCNN(
                num_classes=2,
                num_technical_features=16,
                use_technical=train_dataset.use_technical_features
            ).to(self.device)
            
            # Calculate class weights for this fold
            fold_labels = [train_dataset[i][-1] for i in range(len(train_dataset))]
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(fold_labels), y=fold_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            
            # Train model for this fold
            criterion = TradingFocalLoss(alpha=class_weights, gamma=2)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            
            # Training loop (simplified for CV)
            best_val_metrics = {'accuracy': 0, 'f1_score': 0}
            for epoch in range(num_epochs):
                model.train()
                for batch_data in train_loader:
                    if train_dataset.use_technical_features:
                        data, technical_features, target = batch_data
                        technical_features = technical_features.to(self.device)
                    else:
                        data, target = batch_data
                        technical_features = None
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data, technical_features)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Validation
                model.eval()
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        if train_dataset.use_technical_features:
                            data, technical_features, target = batch_data
                            technical_features = technical_features.to(self.device)
                        else:
                            data, target = batch_data
                            technical_features = None
                        
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data, technical_features)
                        _, predicted = torch.max(output, 1)
                        
                        val_predictions.extend(predicted.cpu().numpy())
                        val_targets.extend(target.cpu().numpy())
                
                # Calculate metrics
                fold_metrics = self.calculate_trading_metrics(
                    np.array(val_predictions), 
                    np.array(val_targets)
                )
                
                if fold_metrics['f1_score'] > best_val_metrics['f1_score']:
                    best_val_metrics = fold_metrics
            
            fold_results.append(best_val_metrics)
            
            print(f"Fold {fold+1} - Best Accuracy: {best_val_metrics['accuracy']*100:.2f}%, "
                  f"Best F1: {best_val_metrics['f1_score']:.4f}")
        
        # Print cross-validation results
        accuracies = [result['accuracy']*100 for result in fold_results]
        f1_scores = [result['f1_score'] for result in fold_results]
        
        print(f"\n{'='*20} Cross Validation Results {'='*20}")
        print(f"Mean Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in accuracies]}")
        print(f"Individual fold F1 scores: {[f'{f1:.4f}' for f1 in f1_scores]}")
        
        return fold_results

def load_enhanced_model_for_inference(model_path, device, use_technical=True):
    """Load enhanced trained model for inference"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = EnhancedMultiModalCNN(
        num_classes=2,
        num_technical_features=16,  # Updated for enhanced features
        use_technical=checkpoint.get('use_technical', use_technical),
        backbone=checkpoint.get('backbone', 'resnet50')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

class TradingBacktester:
    """Enhanced backtesting system for trading performance evaluation"""
    
    def __init__(self, model, device, technical_scaler=None):
        self.model = model
        self.device = device
        self.technical_scaler = technical_scaler
        
    def backtest_strategy(self, test_data, initial_capital=10000, transaction_cost=0.001):
        """Run comprehensive backtest"""
        
        portfolio_value = [initial_capital]
        positions = []
        trades = []
        
        for i in range(len(test_data)):
            # Get model prediction
            prediction = self.get_prediction(test_data.iloc[i])
            positions.append(prediction)
            
            # Calculate returns (simplified)
            if i > 0:
                prev_price = test_data.iloc[i-1]['Close']
                curr_price = test_data.iloc[i]['Close']
                
                if positions[i-1] == 1:  # Buy position
                    returns = (curr_price - prev_price) / prev_price
                elif positions[i-1] == 0:  # Sell position
                    returns = (prev_price - curr_price) / prev_price
                else:  # Hold position
                    returns = 0
                
                # Apply transaction costs if position changed
                if i > 1 and positions[i-1] != positions[i-2]:
                    returns -= transaction_cost
                
                new_value = portfolio_value[-1] * (1 + returns)
                portfolio_value.append(new_value)
        
        # Calculate performance metrics
        total_return = (portfolio_value[-1] - initial_capital) / initial_capital
        returns_series = np.diff(portfolio_value) / portfolio_value[:-1]
        
        metrics = {
            'total_return': total_return,
            'annual_return': total_return * 252 / len(test_data),
            'volatility': np.std(returns_series) * np.sqrt(252),
            'sharpe_ratio': np.mean(returns_series) / np.std(returns_series) * np.sqrt(252) if np.std(returns_series) > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(portfolio_value),
            'portfolio_values': portfolio_value,
            'positions': positions
        }
        
        return metrics
    
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
    
    def get_prediction(self, data_row):
        """Get model prediction for a single data point"""
        # This would need to be implemented based on your specific data format
        # Placeholder implementation
        return 2  # Hold

if __name__ == "__main__":
    print("Enhanced Candlestick Model Trainer v2.0")
    print("=" * 50)
    
    # Initialize trainer
    trainer = EnhancedCandlestickTrainer()
    
    # Train the enhanced model
    print("Training enhanced model with all optimizations...")
    model, trading_metrics = trainer.train_model(
        num_epochs=60,
        learning_rate=0.0005,
        batch_size=16,
        use_technical=True,
        backbone='resnet50',
        use_focal_loss=True
    )
    
    # Print trading metrics
    print("\nTrading Performance Metrics:")
    print("=" * 30)
    for metric, value in trading_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Optionally run cross-validation
    print("\n" + "="*50)
    print("Running Enhanced Cross Validation...")
    print("="*50)
    cv_results = trainer.cross_validate(
        k_folds=3, num_epochs=25, use_technical=True, batch_size=16
    )
    
    print("\n" + "="*50)
    print("ENHANCED MODEL TRAINING COMPLETE!")
    print("="*50)
    print(f"Models saved in: {trainer.model_save_path}")
    print("Files generated:")
    print("- best_enhanced_model.pth (best model)")
    print("- final_enhanced_model.pth (final model)")  
    print("- enhanced_training_history.png (training plots)")
    print("- enhanced_confusion_matrix.png (evaluation results)")