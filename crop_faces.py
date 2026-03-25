from src.face_crop import FaceCropper

def main():
    cropper = FaceCropper()

    # change max_images if you want
    max_images = 2000

    print("=" * 60)
    print("CROPPING REAL IMAGES")
    print("=" * 60)
    cropper.process_folder(
        input_folder="data/real",
        output_folder="cropped_data/real",
        max_images=max_images
    )

    print("\n" + "=" * 60)
    print("CROPPING FAKE IMAGES")
    print("=" * 60)
    cropper.process_folder(
        input_folder="data/fake",
        output_folder="cropped_data/fake",
        max_images=max_images
    )

    print("\n" + "=" * 60)
    print("FACE CROPPING COMPLETE")
    print("=" * 60)
    print("Cropped images saved in:")
    print(" - cropped_data/real")
    print(" - cropped_data/fake")

if __name__ == "__main__":
    main()