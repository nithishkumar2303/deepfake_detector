from src.preprocessing import DataPreprocessor
from src.model import DeepfakeDetector
import numpy as np
import os

class Trainer:
    """
    Complete training pipeline for deepfake detection
    """
    def __init__(self):
        self.preprocessor = None
        self.detector = None
        self.history = None
        
    def run_full_training(self, 
                          real_path="data/real",
                          fake_path="data/fake",
                          img_size=(128, 128),
                          max_images=None,
                          epochs=20,
                          batch_size=32,
                          learning_rate=0.001):
        """
        Complete training pipeline from data loading to model saving
        
        Args:
            real_path: Path to real images
            fake_path: Path to fake images
            img_size: Image size (width, height)
            max_images: Maximum images per class (None = all)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        
        print("\n" + "🚀" * 30)
        print("DEEPFAKE DETECTOR TRAINING PIPELINE")
        print("🚀" * 30 + "\n")
        
        # ============================================================
        # STEP 1: DATA PREPROCESSING
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 60)
        
        self.preprocessor = DataPreprocessor(
            real_path=real_path,
            fake_path=fake_path,
            img_size=img_size
        )
        
        # Load images
        print("\n📂 Loading images...")
        images, labels = self.preprocessor.load_images(max_images=max_images)
        
        # Get statistics
        self.preprocessor.get_statistics(images, labels)
        
        # Split data
        print("\n📊 Splitting data into train/val/test sets...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            images, labels,
            test_size=0.2,
            val_size=0.1
        )
        
        # ============================================================
        # STEP 2: MODEL BUILDING
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 2: MODEL BUILDING")
        print("=" * 60)
        
        self.detector = DeepfakeDetector(input_shape=(img_size[0], img_size[1], 3))
        
        # Build model
        print("\n🔨 Building CNN architecture...")
        self.detector.build_model()
        
        # Compile model
        print("\n⚙️  Compiling model...")
        self.detector.compile_model(learning_rate=learning_rate)
        
        # Show summary
        self.detector.summary()
        
        # ============================================================
        # STEP 3: MODEL TRAINING
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 3: MODEL TRAINING")
        print("=" * 60)
        
        print(f"\n📋 Training Configuration:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        input("\n⏸️  Press ENTER to start training...")
        
        # Train model
        self.history = self.detector.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # ============================================================
        # STEP 4: MODEL EVALUATION
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 4: MODEL EVALUATION")
        print("=" * 60)
        
        test_loss, test_accuracy = self.detector.evaluate(X_test, y_test)
        
        # ============================================================
        # STEP 5: SAVE MODEL AND RESULTS
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 5: SAVING MODEL")
        print("=" * 60)
        
        # Save model
        self.detector.save_model('models/deepfake_detector.h5')
        
        # Plot training history
        self.detector.plot_history('models/training_history.png')
        
        # Save training report
        self.save_training_report(test_accuracy, test_loss, epochs, batch_size)
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "🎉" * 30)
        print("TRAINING COMPLETE!")
        print("🎉" * 30)
        
        print(f"\n📊 Final Results:")
        print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"   Test Loss: {test_loss:.4f}")
        
        print(f"\n💾 Saved Files:")
        print(f"   ✅ models/deepfake_detector.h5 (trained model)")
        print(f"   ✅ models/best_model.h5 (best model during training)")
        print(f"   ✅ models/training_history.png (accuracy/loss graphs)")
        print(f"   ✅ models/training_report.txt (detailed report)")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Run the web application: streamlit run app.py")
        print(f"   2. Upload images to test the detector")
        print(f"   3. Share your results!")
        
        return self.detector
    
    def save_training_report(self, accuracy, loss, epochs, batch_size):
        """
        Save detailed training report
        """
        report_path = 'models/training_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DEEPFAKE DETECTOR - TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-" * 60 + "\n")
            f.write("Type: Convolutional Neural Network (CNN)\n")
            f.write("Input Shape: (128, 128, 3)\n")
            f.write("Total Parameters: ~2.3M\n\n")
            
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Optimizer: Adam\n")
            f.write(f"Loss Function: Binary Crossentropy\n\n")
            
            f.write("FINAL RESULTS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"Test Loss: {loss:.4f}\n\n")
            
            if self.history:
                final_train_acc = self.history.history['accuracy'][-1]
                final_val_acc = self.history.history['val_accuracy'][-1]
                
                f.write("TRAINING HISTORY:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Final Training Accuracy: {final_train_acc * 100:.2f}%\n")
                f.write(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%\n")
                f.write(f"Best Validation Accuracy: {max(self.history.history['val_accuracy']) * 100:.2f}%\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("-" * 60 + "\n")
            f.write("1. deepfake_detector.h5 - Final trained model\n")
            f.write("2. best_model.h5 - Best model checkpoint\n")
            f.write("3. training_history.png - Training graphs\n")
            f.write("4. training_report.txt - This report\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        print(f"\n✅ Training report saved to '{report_path}'")


def main():
    """
    Main training function
    """
    # Create trainer
    trainer = Trainer()
    
    # Configuration
    print("\n" + "⚙️ " * 30)
    print("TRAINING CONFIGURATION")
    print("⚙️ " * 30)
    
    print("\n📋 Default Settings:")
    print("   Images per class: 2000 (for faster training)")
    print("   Image size: 128x128")
    print("   Epochs: 20")
    print("   Batch size: 32")
    print("   Learning rate: 0.001")
    
    print("\n💡 Tip: Start with small dataset first to test everything works!")
    
    use_defaults = input("\nUse default settings? (y/n): ").lower()
    
    if use_defaults == 'y':
        max_images = 2000
        epochs = 20
        batch_size = 32
    else:
        max_images = int(input("Images per class (e.g., 2000): "))
        epochs = int(input("Number of epochs (e.g., 20): "))
        batch_size = int(input("Batch size (e.g., 32): "))
    
    # Start training
    trainer.run_full_training(
        real_path="data/real",
        fake_path="data/fake",
        img_size=(128, 128),
        max_images=max_images,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001
    )


if __name__ == "__main__":
    main()