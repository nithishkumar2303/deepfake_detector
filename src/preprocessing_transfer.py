import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class TransferDataPreprocessor:
    def __init__(self, real_path, fake_path, img_size=(224, 224)):
        self.real_path = real_path
        self.fake_path = fake_path
        self.img_size = img_size

    def load_images(self, max_images=None):
        images = []
        labels = []

        print("=" * 60)
        print("LOADING CROPPED FACE IMAGES")
        print("=" * 60)

        # real = 0
        real_files = [
            f for f in os.listdir(self.real_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if max_images:
            real_files = real_files[:max_images]

        print(f"\nLoading REAL images from {self.real_path} ...")
        for i, filename in enumerate(real_files, 1):
            img_path = os.path.join(self.real_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, self.img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype("float32") / 255.0
                images.append(img)
                labels.append(0)

            if i % 200 == 0 or i == len(real_files):
                print(f"Loaded {i}/{len(real_files)} real images")

        # fake = 1
        fake_files = [
            f for f in os.listdir(self.fake_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if max_images:
            fake_files = fake_files[:max_images]

        print(f"\nLoading FAKE images from {self.fake_path} ...")
        for i, filename in enumerate(fake_files, 1):
            img_path = os.path.join(self.fake_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, self.img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype("float32") / 255.0
                images.append(img)
                labels.append(1)

            if i % 200 == 0 or i == len(fake_files):
                print(f"Loaded {i}/{len(fake_files)} fake images")

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print("\n" + "=" * 60)
        print("LOADING COMPLETE")
        print("=" * 60)
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Real count: {(labels == 0).sum()}")
        print(f"Fake count: {(labels == 1).sum()}")

        return images, labels

    def split_data(self, images, labels, test_size=0.2, val_size=0.1):
        print("\n" + "=" * 60)
        print("SPLITTING DATA")
        print("=" * 60)

        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_size,
            random_state=42,
            stratify=labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=42,
            stratify=y_temp
        )

        print(f"Train: {len(X_train)}")
        print(f"Validation: {len(X_val)}")
        print(f"Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test