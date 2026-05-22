import streamlit as st
from PIL import Image
import os
from datetime import datetime

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Face Data Collector",
    page_icon="📸",
    layout="centered"
)

st.title("📸 AI Face Data Collector")
st.write(
    "Capture face images using your webcam and save them with a person's name."
)

# -----------------------------
# Create Dataset Folder
# -----------------------------
DATASET_DIR = "face_dataset"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# -----------------------------
# Name Input
# -----------------------------
person_name = st.text_input(
    "Enter Person Name",
    placeholder="Example: Rahul"
)

# -----------------------------
# Camera Input
# -----------------------------
picture = st.camera_input("Take a Picture")

# -----------------------------
# Save Image
# -----------------------------
if picture is not None:

    if person_name.strip() == "":
        st.warning("Please enter a person's name before saving.")
    else:

        # Create person folder
        person_folder = os.path.join(DATASET_DIR, person_name)

        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name}_{timestamp}.jpg"

        filepath = os.path.join(person_folder, filename)

        # Save image
        image = Image.open(picture)
        image.save(filepath)

        st.success(f"Image saved successfully!")

        st.image(image, caption=f"Saved for {person_name}")

        st.write("Saved Path:")
        st.code(filepath)

# -----------------------------
# Show Dataset Summary
# -----------------------------
st.divider()

st.subheader("📂 Dataset Summary")

people = os.listdir(DATASET_DIR)

if len(people) == 0:
    st.info("No face data collected yet.")
else:
    for person in people:

        person_folder = os.path.join(DATASET_DIR, person)

        if os.path.isdir(person_folder):
            image_count = len(os.listdir(person_folder))

            st.write(f"👤 {person} → {image_count} images")