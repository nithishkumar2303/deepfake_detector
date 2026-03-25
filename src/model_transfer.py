from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class TransferDeepfakeDetector:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def build_model(self):
        print("=" * 60)
        print("BUILDING TRANSFER LEARNING MODEL")
        print("=" * 60)

        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet"
        )

        base_model.trainable = False

        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation="sigmoid")(x)

        self.model = Model(inputs, outputs)

        print("✅ Transfer model built successfully")
        return self.model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        print("✅ Model compiled")

    def summary(self):
        self.model.summary()

    def get_callbacks(self):
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                "models/best_transfer_model.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

    def train(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        return self.history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return loss, accuracy

    def save_model(self, path="models/deepfake_transfer_model.keras"):
        self.model.save(path)
        print(f"✅ Model saved to {path}")