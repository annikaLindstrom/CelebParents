import streamlit as st

st.title("**Celeb Parents**")
st.subheader('**Find my look-a-likes!**')

import pandas as pd
import time
import faiss
from deepface import DeepFace
from pathlib import Path
parent_path = Path(__file__).parent.parent
data_path = Path(__file__).parent.parent / 'data'

# %% Test search first image
index_male = faiss.read_index('data/full/faiss_flat_ip_male.index')
index_female = faiss.read_index('data/full/faiss_flat_ip_female.index')
df = pd.read_csv('data/full/info.csv')
df_male = df[df['gender'] == 1].reset_index(drop=True)
df_female = df[df['gender'] == 0].reset_index(drop=True)

# %%
import cv2
import ast
import numpy as np
import urllib.request
from sklearn.preprocessing import normalize

# Header for image choice
st.header("Upload your photo or capture one with your webcam")
st.markdown("Note: Sometimes the photo will not match the celebrity name. This is an error in the IMDB dataset. The dataset has multiple images for each actor, and some photos are wrongly assigned. For instance, Sacha Baron Cohen has a female actress assigned to his folder. We leave the work of fixing this issue as a future development.")

# Initialize session state variables if not already set
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

if 'captured_image' not in st.session_state:
    st.session_state['captured_image'] = None

# Choice for the user to upload or capture image
choice = st.radio("Choose your method:", ("Upload an Image", "Capture with Webcam"))

if choice == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['captured_image'] = None  # Reset the other option

elif choice == "Capture with Webcam":
    captured_image = st.camera_input("Take a picture with your webcam")
    if captured_image is not None:
        st.session_state['captured_image'] = captured_image
        st.session_state['uploaded_file'] = None  # Reset the other option

# Process the image if either is available
if st.session_state['uploaded_file'] or st.session_state['captured_image']:
    if st.session_state['uploaded_file']:
        bytes_data = st.session_state['uploaded_file'].getvalue()
    else:
        bytes_data = st.session_state['captured_image'].getvalue()

    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if img is not None:
        # Convert the color from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Display the uploaded or captured image
        st.image(img, caption='Uploaded/Captured Image', use_column_width=True)
        # More processing code can follow here...
    # Save the processed image temporarily to pass to DeepFace
    cv2.imwrite('temp_image.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Running DeepFace analysis
    try:
        t0 = time.time()
        yourself_representation = DeepFace.represent(
            img_path='temp_image.jpg',
            detector_backend='opencv',
        )
        normalized_user_embeddings = normalize([yourself_representation[0]["embedding"]], axis=1, norm='l2')
        t1 = time.time()
        st.write(f"Processed your image in {t1 - t0:.2f} seconds")

        # Perform MALE search with the image representation
        index_male = faiss.read_index("data/full/faiss_flat_ip_male.index")
        match_male = index_male.search(np.asarray(normalized_user_embeddings), k=3)
        matching_rows_male = match_male[1][0, :]

        # Perform FEMALE search with the image representation
        index_female = faiss.read_index("data/full/faiss_flat_ip_female.index")
        match_female = index_female.search(np.asarray(normalized_user_embeddings), k=3)
        matching_rows_female = match_female[1][0, :]

        # Display matches side by side
        for i in range(3):
            col1, col2 = st.columns(2)

            with col1:
                instance_male = df_male.iloc[matching_rows_male.tolist()[i]]
                name_list_male = ast.literal_eval(instance_male['name'])  # Convert the string representation of list to actual list
                name_male = name_list_male[0] if name_list_male else "Unknown"  # Extract the first item
                distance_male = match_male[0][0, i]
                full_path_string_male = instance_male['full_path']
                full_path_male = ast.literal_eval(full_path_string_male)
                clean_path_male = full_path_male[0]
                # img_path = "imdb_crop/" + clean_path
                img_url_male = "https://storage.googleapis.com/imdb_cropped_images/imdb_crop/" + clean_path_male
                #st.write(f"Debug: Male name: {name_male}, URL: {img_url_male}")  # Debug print

                # Use urllib to get the image from the URL
                resp_male = urllib.request.urlopen(img_url_male)
                image_male = np.asarray(bytearray(resp_male.read()), dtype="uint8")
                celeb_img_male = cv2.imdecode(image_male, cv2.IMREAD_COLOR)  # Decode the image data
                # celeb_img = cv2.imread(img_path)
                celeb_img_male = cv2.cvtColor(celeb_img_male, cv2.COLOR_BGR2RGB)
                celeb_img_male = cv2.resize(celeb_img_male, (192, 192))
                st.write(f"{i + 1}. {name_male} (Distance: {distance_male:.2f})")
                st.image(celeb_img_male, use_column_width=True)

            with col2:
                instance_female = df_female.iloc[matching_rows_female.tolist()[i]]
                name_list_female = ast.literal_eval(instance_female['name'])  # Convert the string representation of list to actual list
                name_female = name_list_female[0] if name_list_female else "Unknown"  # Extract the first item
                distance_female = match_female[0][0, i]
                full_path_string_female = instance_female['full_path']
                full_path_female = ast.literal_eval(full_path_string_female)
                clean_path_female = full_path_female[0]
                # img_path = "imdb_crop/" + clean_path
                img_url_female = "https://storage.googleapis.com/imdb_cropped_images/imdb_crop/" + clean_path_female
                #st.write(f"Debug: Female name: {name_female}, URL: {img_url_female}")  # Debug print

                # Use urllib to get the image from the URL
                resp_female = urllib.request.urlopen(img_url_female)
                image_female = np.asarray(bytearray(resp_female.read()), dtype="uint8")
                celeb_img_female = cv2.imdecode(image_female, cv2.IMREAD_COLOR)  # Decode the image data
                # celeb_img = cv2.imread(img_path)
                celeb_img_female = cv2.cvtColor(celeb_img_female, cv2.COLOR_BGR2RGB)
                celeb_img_female = cv2.resize(celeb_img_female, (192, 192))
                st.write(f"{i + 1}. {name_female} (Distance: {distance_female:.2f})")
                st.image(celeb_img_female, use_column_width=True)

    except Exception as e:
        st.write("Failed to process the image:", e)

