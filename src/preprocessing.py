import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class DataPreprocessor:
    """
    Handles all data preprocessing tasks
    """
    def __init__(self, real_path, fake_path, img_size=(128, 128)):
        """
        Initialize preprocessor
        
        Args:
            real_path: Path to real images folder
            fake_path: Path to fake images folder
            img_size: Target size for all images (width, height)
        """
        self.real_path = real_path
        self.fake_path = fake_path
        self.img_size = img_size
        self.images = None
        self.labels = None
        
    def load_images(self, max_images=None):
        """
        Load and preprocess all images
        
        Args:
            max_images: Maximum number of images per class (None = all)
        
        Returns:
            images: numpy array of images
            labels: numpy array of labels (0=real, 1=fake)
        """
        images = []
        labels = []
        
        print("=" * 60)
        print("LOADING IMAGES")
        print("=" * 60)
        
        # Load Real Images (label = 0)
        print("\n📁 Loading REAL images...")
        real_files = [f for f in os.listdir(self.real_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_images:
            real_files = real_files[:max_images]
        
        for i, filename in enumerate(real_files):
            img_path = os.path.join(self.real_path, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Resize image
                img = cv2.resize(img, self.img_size)
                # Normalize to 0-1 range
                img = img / 255.0
                
                images.append(img)
                labels.append(0)  # Real = 0
                
                # Progress indicator
                if (i + 1) % 500 == 0:
                    print(f"   ✓ Loaded {i + 1}/{len(real_files)} real images")
        
        print(f"   ✅ Total real images loaded: {len([l for l in labels if l == 0])}")
        
        # Load Fake Images (label = 1)
        print("\n📁 Loading FAKE images...")
        fake_files = [f for f in os.listdir(self.fake_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_images:
            fake_files = fake_files[:max_images]
        
        for i, filename in enumerate(fake_files):
            img_path = os.path.join(self.fake_path, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Resize image
                img = cv2.resize(img, self.img_size)
                # Normalize to 0-1 range
                img = img / 255.0
                
                images.append(img)
                labels.append(1)  # Fake = 1
                
                # Progress indicator
                if (i + 1) % 500 == 0:
                    print(f"   ✓ Loaded {i + 1}/{len(fake_files)} fake images")
        
        print(f"   ✅ Total fake images loaded: {len([l for l in labels if l == 1])}")
        
        # Convert to numpy arrays
        self.images = np.array(images, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)
        
        print("\n" + "=" * 60)
        print(f"✅ LOADING COMPLETE!")
        print("=" * 60)
        print(f"Total images: {len(self.images)}")
        print(f"Image shape: {self.images[0].shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Memory usage: {self.images.nbytes / (1024**2):.2f} MB")
        
        return self.images, self.labels
    
    def split_data(self, images=None, labels=None, test_size=0.2, val_size=0.1):
        """
        Split data into training, validation, and testing sets
        
        Args:
            images: Image data (uses self.images if None)
            labels: Label data (uses self.labels if None)
            test_size: Proportion of data for testing (0.2 = 20%)
            val_size: Proportion of training data for validation (0.1 = 10%)
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if images is None:
            images = self.images
        if labels is None:
            labels = self.labels
        
        print("\n" + "=" * 60)
        print("SPLITTING DATA")
        print("=" * 60)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, 
            test_size=test_size, 
            random_state=42,
            stratify=labels  # Maintain class balance
        )
        
        # Second split: separate validation set from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"\n📊 Dataset Distribution:")
        print(f"   Training set:   {len(X_train)} images ({len(X_train)/len(images)*100:.1f}%)")
        print(f"   Validation set: {len(X_val)} images ({len(X_val)/len(images)*100:.1f}%)")
        print(f"   Test set:       {len(X_test)} images ({len(X_test)/len(images)*100:.1f}%)")
        
        print(f"\n⚖️  Class Balance:")
        print(f"   Training   - Real: {sum(y_train==0)}, Fake: {sum(y_train==1)}")
        print(f"   Validation - Real: {sum(y_val==0)}, Fake: {sum(y_val==1)}")
        print(f"   Test       - Real: {sum(y_test==0)}, Fake: {sum(y_test==1)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def visualize_samples(self, images, labels, num_samples=10, save_path='preprocessing_samples.png'):
        """
        Visualize preprocessed images
        
        Args:
            images: Array of images
            labels: Array of labels
            num_samples: Number of samples to display
            save_path: Where to save the visualization
        """
        print(f"\n📊 Creating visualization...")
        
        # Randomly select samples
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Preprocessed Images (Resized & Normalized)', 
                     fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(indices):
            row = i // 5
            col = i % 5
            
            # Display image
            axes[row, col].imshow(images[idx])
            
            # Set title based on label
            if labels[idx] == 0:
                axes[row, col].set_title('REAL', color='green', fontweight='bold')
            else:
                axes[row, col].set_title('FAKE', color='red', fontweight='bold')
            
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualization saved as '{save_path}'")
        plt.show()
    
    def get_statistics(self, images, labels):
        """
        Get dataset statistics
        
        Args:
            images: Array of images
            labels: Array of labels
        """
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        print(f"\n📐 Image Properties:")
        print(f"   Shape: {images.shape}")
        print(f"   Data type: {images.dtype}")
        print(f"   Min value: {images.min():.4f}")
        print(f"   Max value: {images.max():.4f}")
        print(f"   Mean value: {images.mean():.4f}")
        
        print(f"\n🏷️  Label Properties:")
        print(f"   Shape: {labels.shape}")
        print(f"   Unique labels: {np.unique(labels)}")
        print(f"   Real images (0): {sum(labels == 0)}")
        print(f"   Fake images (1): {sum(labels == 1)}")
        
        print(f"\n💾 Memory:")
        print(f"   Images: {images.nbytes / (1024**2):.2f} MB")
        print(f"   Labels: {labels.nbytes / 1024:.2f} KB")