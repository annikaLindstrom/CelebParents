import streamlit as st

# Press Release

st.markdown(
    '''
Press Release
FOR IMMEDIATE RELEASE

[Durham, 04/2024] — A team of talented students at Duke University has unveiled a cutting-edge new application that is sure to catch the eye of pop culture enthusiasts and technology aficionados alike: the "CelebParentMatch," a celebrity look-alike generator. Developed as part of a class project, this app uses advanced machine learning techniques to analyze user-submitted photos and identify which celebrity the user most resembles.

**About CelebParentMatch**
CelebParentMatch harnesses the power of the extensive cropped IMDB-Faces dataset, which contains about 90,000 images of various celebrities, to offer users a fun and engaging way to connect with the world of stardom. By applying facial recognition algorithms (DeepFace, specifically), CelebParentMatch compares user facial features with those of celebrities to find a match.

**Features of CelebParentMatch**
User-Friendly Interface: Easy upload and instant results.
Privacy-Focused: Your photos are your business. We forget them faster than a celebrity forgets an ex.

CelebParentMatch is not just a source of entertainment. It has educational value in demonstrating the capabilities of artificial intelligence and machine learning, especially in the field of computer vision.
'''
)

st.subheader('**Technical Details**')

st.markdown('''
First, we create the embeddings for each image in the IMDB-Faces dataset using DeepFace. More specifically, first we clean the dataset by removing pictures that don’t have faces in them, have more than one face or are of low quality (meaning a face score smaller than 3). Then, we extract the embeddings (a vector representation of the features) in each image using the VGG-Face algorithm as implemented in DeepFace. VGG-Face is a Very-Deep 16 layer regular convolutional neural network, first proposed in Parkhi et al. (2015), that reduces images to a 4096 vector of features. Its architecture is described in Figure 1.

After creating the embeddings, we use Faiss to create a search index that matches the embeddings extracted from each image with the original dataset. Faiss allows us to efficiently search for the maximal cosine similarity between a given input image and the IMDB-Faces dataset images, and it lets us relate this maximal element to its original format.

Therefore, the algorithm works as follows: Given an input image, first we use DeepFace to extract a 4096 vector of features (the same process done with the IMDB-Faces dataset). Then, we search for the male and female embedding in the dataset that has the largest cosine similarity with the input image. IMDB-Faces has images pre-labeled as either female or male, which facilitates this step. Finally, it returns the pair of images (male and female) that has the largest similarity with the input.

''')

# Acknowledgements
st.subheader('**Acknowledgements**')
with st.expander("**Acknowledgements**"):
    st.markdown('''
        CelebParentMatch was brought into existence through the efforts of our team.

        **Annika (amh227)** laid the foundation with our starter code which processed the images and created embeddings, all while juggling the joys and jitters of welcoming a new baby into her life.

        **Fatima (fa83)** then extended that code to incorporate the [DeepFace library](https://github.com/serengil/deepface/tree/master). With the indispensable aids of the internet and Stack Overflow, and frequent recourse to the 'phone a friend' lifeline, she developed the code that calculates the distances between user images and celebrities using [FAISS](https://ai.meta.com/tools/faiss/).  Much like managing a 4-month-old, she handled unexpected bugs with grace and a sense of humor.

        **Rafael (rpa9)** had the idea for the project, wrote the technical details and failed miserably in making it operational. Thanks Fatima and Annika for making it happen!

        **Fatima (fa83)** also crafted the press release and FAQ, and put together the streamlit.

           ''')

st.header('**FAQs**')

with st.expander("**What is CelebParentMatch?**"):
    st.markdown('''
        CelebParentMatch is your digital mirror into the world of fame, using facial recognition to match your photo with a celebrity who shares your star-worthy features. Just upload a picture, and see which celebrity you might replace in your next life.
    ''')

with st.expander("**How does CelebParentMatch work?**"):
    st.markdown('''
        Simply upload your photo through our app, and our algorithm will analyze your facial features using the IMDB dataset to find the celebrity whose features most closely resemble yours. To do this, we first create a vector of features (i.e., embeddings) of the faces in the IMDB dataset. We use the DeepFace library for that. Next, in order to find look-a-likes, we have to find faces that are close matches to that of the user. To do this, we use FAISS, an advanced library for efficient similarity search and clustering of dense vectors. After extracting the embeddings of your photo using the DeepFace library, these embeddings are then compared against the pre-processed embeddings of celebrity faces from the IMDB dataset stored in our FAISS index. FAISS efficiently searches through these vectors to find the closest matches based on the cosine similarity of the embeddings. This process not only ensures fast results but also maintains high accuracy in identifying the celebrity that most resembles you. Once the closest celebrity match is found, the result is displayed to you instantly on our app.
        ''')

with st.expander("**Why am I getting photos of the same gender? I thought you were meant to show me look alikes of both genders**"):
    st.markdown('''
The IMDB dataset unfortunately has some mislabeled images. Some photos, let's say of a woman, may be attributed to a male celebrity. So, there is definitely room for improvement. First things first, the data needs to be fixed.
    ''')

with st.expander("**Is my photo secure with CelebParentMatch?**"):
    st.markdown('''
    Yes, privacy is our top priority. Photos are processed in real time and are not stored on our servers.
    ''')

with st.expander("**Can I use any photo?**"):
    st.markdown('''
For best results, use a clear, front-facing photo where your face is not obscured by objects or shadows.
    ''')

with st.expander("**Is CelebParentMatch free to use?**"):
    st.markdown('''
Yes, CelebParentMatch is currently available for free as part of our educational project showcasing the power of machine learning.
    ''')

with st.expander("**HELP! The webcam option doesn’t work for me!**"):
    st.markdown('''
If you run the app using Docker, the webcam version will not work. If you run the app locally, however, it will.
    ''')

with st.expander("**Who created CelebParentMatch?**"):
    st.markdown('''
CelebParentMatch was created by STA 561 students at Duke University as part of a class project focused on machine learning.
    ''')



st.header('**Find my look-a-likes!**')

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

