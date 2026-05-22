import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

# =====================================================
# INITIALIZE FACENET
# =====================================================

print("Loading FaceNet model...")

# Face detector
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40
)

# Face embedding model
resnet = InceptionResnetV1(
    pretrained='vggface2'
).eval()

print("Model loaded successfully!")

# =====================================================
# LOAD KNOWN FACES
# =====================================================

DATASET_DIR = "face_dataset"

known_face_embeddings = []
known_face_names = []

print("\nLoading known faces...")

for person_name in os.listdir(DATASET_DIR):

    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    embeddings = []

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        try:
            # Open image
            img = Image.open(image_path).convert("RGB")

            # Detect and align face
            face = mtcnn(img)

            if face is not None:

                # Generate embedding
                embedding = resnet(
                    face.unsqueeze(0)
                ).detach()

                embeddings.append(embedding)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Average embeddings for better accuracy
    if len(embeddings) > 0:

        avg_embedding = torch.mean(
            torch.stack(embeddings),
            dim=0
        )

        known_face_embeddings.append(avg_embedding)
        known_face_names.append(person_name)

        print(f"Loaded {person_name}")

print("\nKnown people:")
print(known_face_names)

# =====================================================
# START WEBCAM
# =====================================================

cap = cv2.VideoCapture(0)

print("\nStarting webcam...")
print("Press Q to quit")

# =====================================================
# RECOGNITION LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL image
    pil_image = Image.fromarray(rgb_frame)

    # Detect faces
    boxes, _ = mtcnn.detect(pil_image)

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = box.astype(int)

            # Crop face
            face_crop = pil_image.crop(
                (x1, y1, x2, y2)
            )

            # Get aligned face
            face_tensor = mtcnn(face_crop)

            if face_tensor is not None:

                # Generate embedding
                embedding = resnet(
                    face_tensor.unsqueeze(0)
                ).detach()

                # Compare with known faces
                min_distance = 999
                identity = "Unknown"

                for i, known_embedding in enumerate(
                    known_face_embeddings
                ):

                    distance = torch.dist(
                        embedding,
                        known_embedding
                    ).item()

                    if distance < min_distance:
                        min_distance = distance
                        identity = known_face_names[i]

                # Threshold
                if min_distance > 0.9:
                    identity = "Unknown"

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label text
                label = f"{identity} ({min_distance:.2f})"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    # Show webcam
    cv2.imshow(
        "Face Recognition Attendance System",
        frame
    )

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================================
# CLEANUP
# =====================================================

cap.release()
cv2.destroyAllWindows()

print("\nProgram closed.")