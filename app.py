import streamlit as st
from model import PigReIDModel
from reid_engine import PigDatabase, identify_pig
from database import FarmDatabaseManager
from utils import load_image, create_directory
import zipfile
import os
import shutil

st.title("Smart Pig ID System")

st.sidebar.title("Farm Management")
mode = st.sidebar.radio("Mode", ["Use Existing Farm", "Initialize New Farm"])
farm_name = st.sidebar.text_input("Farm Name")

manager = FarmDatabaseManager(base_path="farms")

# -----------------------------
# MODE 1: Use Existing Farm
# -----------------------------
if mode == "Use Existing Farm":

    if not farm_name:
        st.warning("Please enter a farm name.")

    elif farm_name not in manager.list_farms():
        st.error(f"Farm '{farm_name}' does not exist!")

    else:
        st.success(f"Loading farm '{farm_name}'...")

        if "model" not in st.session_state:
            st.session_state["model"] = PigReIDModel(
                checkpoint_path="checkpoints/best_model.pth"
            )

        if "farm_db" not in st.session_state:
            st.session_state["farm_db"] = PigDatabase(farm_name=farm_name)

        model = st.session_state["model"]
        farm_db = st.session_state["farm_db"]

        st.success("Model and farm database loaded.")

        uploaded_file = st.file_uploader(
            "Upload Pig Ear Image",
            type=["jpg", "png"]
        )
    
        if uploaded_file:

            image = load_image(uploaded_file)
            checkpoint = fr"farms\{farm_name}\checkpoints\best_model.pth"
            classes_json = fr"farms\{farm_name}\classes.json"
    
            label, confidence = identify_pig(image,
                checkpoint,
                classes_json,
        
            )

            if label:
                st.success(
                    f"Pig identified: {label} (Confidence: {confidence:.2f})"
                )

            else:
                st.warning(
                    f"Unknown pig (Confidence: {confidence:.2f})"
                )

                new_label = st.text_input(
                    "Enter new Pig ID to register:"
                )

                if new_label:

                    embedding = model.get_embedding(image)

                    farm_db.add_pig(embedding, new_label)
                    farm_db.save_database()

                    pig_folder = os.path.join(
                        "farms",
                        farm_name,
                        new_label
                    )

                    create_directory(pig_folder)

                    img_count = len(os.listdir(pig_folder)) + 1

                    img_path = os.path.join(
                        pig_folder,
                        f"{new_label}_{img_count}.jpg"
                    )

                    image.save(img_path)

                    st.success(
                        f"New pig '{new_label}' registered successfully!"
                    )

# -----------------------------
# MODE 2: Initialize New Farm
# -----------------------------
if mode == "Initialize New Farm":

    if not farm_name:
        st.warning("Please enter a farm name.")

    else:

        # Create farm if it doesn't exist
        if farm_name not in manager.list_farms():
            manager.create_farm(farm_name)

        # Initialize session variables
        if "farm_db" not in st.session_state:
            st.session_state["farm_db"] = PigDatabase(
                farm_name=farm_name
            )

        if "model" not in st.session_state:
            st.session_state["model"] = PigReIDModel(
                checkpoint_path="checkpoints/base_model.ckpt"
            )

        if "continue_adding" not in st.session_state:
            st.session_state["continue_adding"] = True

        if "capture_index" not in st.session_state:
            st.session_state["capture_index"] = 0

        farm_db = st.session_state["farm_db"]
        model = st.session_state["model"]

        init_option = st.radio(
            "Choose Initialization Method:",
            [
                "Take Pig Ear Photos & Enter IDs",
                "Upload Existing Dataset (ZIP)"
            ]
        )

        # -----------------------------
        # OPTION 1: Capture Pig Photos
        # -----------------------------
        if init_option == "Take Pig Ear Photos & Enter IDs":

            st.info(
                "Capture pig ear images and assign Pig IDs one at a time."
            )

            capture_index = st.session_state["capture_index"]

            # Pig counter
            try:
                pig_count = len(set(farm_db.list_pigs()))
            except:
                pig_count = 0

            st.metric("🐷 Pigs Registered", pig_count)

            if not st.session_state["continue_adding"]:
                st.success(
                    f"Farm '{farm_name}' initialization complete!"
                )
                st.stop()

            pig_id = st.text_input(
                "Enter Pig ID",
                key=f"pig_id_{capture_index}"
            )

            uploaded_img = st.camera_input(
                "Capture Pig Ear Photo",
                key=f"camera_input_{capture_index}"
            )

            if st.button("Save Pig Image"):

                if pig_id and uploaded_img:

                    image = load_image(uploaded_img)

                    embedding = model.get_embedding(image)

                    farm_db.add_pig(embedding, pig_id)
                    farm_db.save_database()

                    pig_folder = os.path.join(
                        "farms",
                        farm_name,
                        pig_id
                    )

                    create_directory(pig_folder)

                    img_count = len(os.listdir(pig_folder)) + 1

                    img_path = os.path.join(
                        pig_folder,
                        f"{pig_id}_{img_count}.jpg"
                    )

                    image.save(img_path)

                    st.success(
                        f"Pig '{pig_id}' registered successfully!"
                    )

                else:
                    st.warning(
                        "Please enter Pig ID and capture image."
                    )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📷 Capture Another Picture"):

                    st.session_state["capture_index"] += 1

                    st.rerun()

            with col2:
                if st.button("✅ Finish Farm Setup"):

                    st.session_state["continue_adding"] = False

                    st.success(
                        f"Farm '{farm_name}' initialization complete!"
                    )

                    st.stop()

        # -----------------------------
        # OPTION 2: Upload ZIP Dataset
        # -----------------------------
        elif init_option == "Upload Existing Dataset (ZIP)":

            st.info(
                "Upload a ZIP file containing pig images in subfolders named by Pig ID."
            )

            zip_file = st.file_uploader(
                "Upload ZIP",
                type="zip"
            )

            if zip_file:

                temp_dir = f"temp_{farm_name}"

                create_directory(temp_dir)

                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                st.success("ZIP extracted successfully.")

                for pig_id_folder in os.listdir(temp_dir):

                    pig_path = os.path.join(
                        temp_dir,
                        pig_id_folder
                    )

                    if os.path.isdir(pig_path):

                        for img_file in os.listdir(pig_path):

                            img_path = os.path.join(
                                pig_path,
                                img_file
                            )

                            image = load_image(img_path)

                            if image:

                                embedding = model.get_embedding(image)

                                farm_db.add_pig(
                                    embedding,
                                    pig_id_folder
                                )

                                pig_folder = os.path.join(
                                    "farms",
                                    farm_name,
                                    pig_id_folder
                                )

                                create_directory(pig_folder)

                                img_count = len(
                                    os.listdir(pig_folder)
                                ) + 1

                                img_path_save = os.path.join(
                                    pig_folder,
                                    f"{pig_id_folder}_{img_count}.jpg"
                                )

                                image.save(img_path_save)

                farm_db.save_database()

                st.success(
                    f"Farm '{farm_name}' initialized from uploaded dataset!"
                )

                shutil.rmtree(temp_dir)