from src.preprocessing_transfer import TransferDataPreprocessor
from src.model_transfer import TransferDeepfakeDetector
import matplotlib.pyplot as plt

def plot_history(history, save_path="models/transfer_training_history.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Training history saved to {save_path}")
    plt.show()

def main():
    print("\n" + "🚀" * 30)
    print("TRAINING REBUILT MODEL WITH MOBILENETV2")
    print("🚀" * 30 + "\n")

    preprocessor = TransferDataPreprocessor(
        real_path="cropped_data/real",
        fake_path="cropped_data/fake",
        img_size=(224, 224)
    )

    images, labels = preprocessor.load_images()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(images, labels)

    detector = TransferDeepfakeDetector(input_shape=(224, 224, 3))
    detector.build_model()
    detector.compile_model(learning_rate=0.001)
    detector.summary()

    input("\nPress ENTER to start training...")

    history = detector.train(
        X_train, y_train,
        X_val, y_val,
        epochs=15,
        batch_size=32
    )

    detector.evaluate(X_test, y_test)
    detector.save_model("models/deepfake_transfer_model.keras")
    plot_history(history)

if __name__ == "__main__":
    main()