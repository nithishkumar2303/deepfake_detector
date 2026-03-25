from src.preprocessing import DataPreprocessor
import numpy as np

def main():
    """
    Test preprocessing pipeline
    """
    print("\n" + "🔧" * 30)
    print("TESTING DATA PREPROCESSING")
    print("🔧" * 30 + "\n")
    
    # Initialize preprocessor
    print("Step 1: Initializing preprocessor...")
    preprocessor = DataPreprocessor(
        real_path="data/real",
        fake_path="data/fake",
        img_size=(128, 128)
    )
    print("✅ Preprocessor initialized\n")
    
    # Load images (using subset for faster testing)
    print("Step 2: Loading images...")
    print("(Using 2000 images of each type for testing)\n")
    
    images, labels = preprocessor.load_images(max_images=2000)
    
    # Get statistics
    preprocessor.get_statistics(images, labels)
    
    # Split data
    print("\nStep 3: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        images, labels,
        test_size=0.2,
        val_size=0.1
    )
    
    # Visualize samples
    print("\nStep 4: Visualizing samples...")
    preprocessor.visualize_samples(X_train, y_train, num_samples=10)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING TEST COMPLETE!")
    print("=" * 60)
    print("\n📦 Data Ready for Training:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_val shape:   {X_val.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_val shape:   {y_val.shape}")
    print(f"   y_test shape:  {y_test.shape}")
    
    print("\n💡 Next Step: Build and train the CNN model!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    data = main()