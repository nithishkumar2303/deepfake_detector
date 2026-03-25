from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, 
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

class DeepfakeDetector:
    """
    CNN Model for Deepfake Detection
    """
    def __init__(self, input_shape=(128, 128, 3)):
        """
        Initialize the model
        
        Args:
            input_shape: Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the CNN architecture
        
        Architecture:
        - 4 Convolutional blocks (Conv -> BatchNorm -> MaxPool)
        - Flatten layer
        - 2 Dense layers with Dropout
        - Output layer (sigmoid for binary classification)
        """
        print("\n" + "=" * 60)
        print("BUILDING CNN MODEL")
        print("=" * 60)
        
        model = Sequential([
            # Input Layer
            Input(shape=self.input_shape),
            
            # ============ CONVOLUTIONAL BLOCK 1 ============
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # ============ CONVOLUTIONAL BLOCK 2 ============
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # ============ CONVOLUTIONAL BLOCK 3 ============
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # ============ CONVOLUTIONAL BLOCK 4 ============
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # ============ FLATTEN ============
            Flatten(),
            
            # ============ DENSE LAYERS ============
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # ============ OUTPUT LAYER ============
            Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
        ])
        
        self.model = model
        
        print("\n✅ Model architecture created!")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not built yet! Call build_model() first.")
        
        print("\n" + "=" * 60)
        print("COMPILING MODEL")
        print("=" * 60)
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n✅ Model compiled!")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: Binary Crossentropy")
        print(f"   Metrics: Accuracy")
    
    def summary(self):
        """
        Display model architecture
        """
        if self.model is None:
            raise ValueError("Model not built yet! Call build_model() first.")
        
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60 + "\n")
        
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        print("\n" + "=" * 60)
        print(f"📊 Total Parameters: {total_params:,}")
        print("=" * 60)
        
    def get_callbacks(self, patience=5):
        """
        Create training callbacks
        
        Args:
            patience: Number of epochs to wait before early stopping
        
        Returns:
            List of callbacks
        """
        callbacks = [
            # Stop training if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Save the best model
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when stuck
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built yet! Call build_model() first.")
        
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print("\n" + "=" * 60)
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test images
            y_test: Test labels
        
        Returns:
            Test loss and accuracy
        """
        if self.model is None:
            raise ValueError("Model not built yet!")
        
        print("\n" + "=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print("=" * 60)
        
        return test_loss, test_accuracy
    
    def plot_history(self, save_path='models/training_history.png'):
        """
        Plot training history
        
        Args:
            save_path: Where to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available!")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Training history plot saved to '{save_path}'")
        plt.show()
    
    def save_model(self, filepath='models/deepfake_detector.h5'):
        """
        Save the trained model
        
        Args:
            filepath: Where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save(filepath)
        print(f"\n✅ Model saved to '{filepath}'")
    
    def predict(self, image):
        """
        Predict if an image is real or fake
        
        Args:
            image: Input image (should be preprocessed)
        
        Returns:
            Prediction (0 = real, 1 = fake)
        """
        import numpy as np
        
        if self.model is None:
            raise ValueError("Model not built yet!")
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)[0][0]
        
        return prediction