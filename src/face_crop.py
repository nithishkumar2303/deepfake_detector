import os
import cv2

class FaceCropper:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_and_crop_face(self, image, padding=20):
        """
        Detect the largest face and crop it.
        Returns cropped face or None if no face found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return None

        # pick largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        cropped_face = image[y1:y2, x1:x2]
        return cropped_face

    def process_folder(self, input_folder, output_folder, max_images=None):
        """
        Crop faces from all images in a folder and save them.
        """
        os.makedirs(output_folder, exist_ok=True)

        files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if max_images:
            files = files[:max_images]

        total = len(files)
        saved = 0
        skipped = 0

        print(f"\nProcessing folder: {input_folder}")
        print(f"Total images found: {total}")

        for i, filename in enumerate(files, 1):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                skipped += 1
                continue

            face = self.detect_and_crop_face(image)

            if face is not None:
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, face)
                saved += 1
            else:
                skipped += 1

            if i % 200 == 0 or i == total:
                print(f"Processed {i}/{total} | Saved: {saved} | Skipped: {skipped}")

        print(f"\nDone: {input_folder}")
        print(f"Saved faces: {saved}")
        print(f"Skipped: {skipped}")

        return saved, skipped