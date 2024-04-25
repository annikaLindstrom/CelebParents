import streamlit as st


# Press Release

st.markdown(
        '''
Press Release  
FOR IMMEDIATE RELEASE

[Durham, 04/2024] — A team of talented students at Duke University has unveiled a cutting-edge new application that is sure to catch the eye of pop culture enthusiasts and technology aficionados alike: the "CelebMatch," a celebrity look-alike generator. Developed as part of a class project, this app uses advanced machine learning techniques to analyze user-submitted photos and identify which celebrity the user most resembles. 

**About CelebMatch**  
CelebMatch harnesses the power of the extensive cropped IMDB-Faces dataset, which contains about 90,000 images of various celebrities, to offer users a fun and engaging way to connect with the world of stardom. By applying facial recognition algorithms (DeepFace, specifically), CelebMatch compares user facial features with those of celebrities to find a match.

**Features of CelebMatch**  
User-Friendly Interface: Easy upload and instant results.  
Privacy-Focused: Your photos are your business. We forget them faster than a celebrity forgets an ex.

CelebMatch is not just a source of entertainment. It has educational value in demonstrating the capabilities of artificial intelligence and machine learning, especially in the field of computer vision.
'''
    )

# Acknowledgements
st.subheader('**Acknowledgements**')
with st.expander("**Acknowledgements**"):
    st.markdown('''
        CelebMatch was brought into existence through the efforts of our team.  
        
        **Annika** laid the foundation with a clear, efficient starter code that processed the images and created embeddings, all while juggling the joys and jitters of welcoming a new baby into her life. 
        
        **Fatima** then extended that code to incorporate the [DeepFace library](https://github.com/serengil/deepface/tree/master). With the indispensable aids of the internet and Stack Overflow, and frequent recourse to the 'phone a friend' lifeline, she developed the code that calculates the distances between user images and celebrities using [FAISS](https://ai.meta.com/tools/faiss/).  Much like managing a 4-month-old, she handled unexpected bugs with grace and a sense of humor.
        
        **Rafael** .........

        **Fatima** also crafted the press release and FAQ. 
        
        **Fatima** put together the streamlit.
    ''')

st.header('**FAQs**')

with st.expander("**What is CelebMatch?**"):
    st.markdown('''
        CelebMatch is your digital mirror into the world of fame, using facial recognition to match your photo with a celebrity who shares your star-worthy features. Just upload a picture, and see which celebrity you might replace in your next life.
    ''')

with st.expander("**How does CelebMatch work?**"):
    st.markdown('''
        Simply upload your photo through our app, and our algorithm will analyze your facial features using the IMDB dataset to find the celebrity whose features most closely resemble yours. To do this, we first create a vector of features (i.e., embeddings) of the faces in the IMDB dataset. We use the DeepFace library for that. Next, in order to find look-a-likes, we have to find faces that are close matches to that of the user. To do this, we use FAISS, an advanced library for efficient similarity search and clustering of dense vectors. After extracting the embeddings of your photo using the DeepFace library, these embeddings are then compared against the pre-processed embeddings of celebrity faces from the IMDB dataset stored in our FAISS index. FAISS efficiently searches through these vectors to find the closest matches based on the cosine similarity of the embeddings. This process not only ensures fast results but also maintains high accuracy in identifying the celebrity that most resembles you. Once the closest celebrity match is found, the result is displayed to you instantly on our app.
        ''')

with st.expander("**Is my photo secure with CelebMatch?**"):
    st.markdown('''
    Yes, privacy is our top priority. Photos are processed in real time and are not stored on our servers.
    ''')

with st.expander("**Can I use any photo?**"):
    st.markdown('''
For best results, use a clear, front-facing photo where your face is not obscured by objects or shadows.
    ''')


with st.expander("**Is CelebMatch free to use?**"):
    st.markdown('''
Yes, CelebMatch is currently available for free as part of our educational project showcasing the power of machine learning.    
    ''')

with st.expander("**Where can I access CelebMatch?**"):
    st.markdown('''

CelebMatch is available online at [website URL]. You can access it from both desktop and mobile devices.
    ''')

with st.expander("**Who created CelebMatch?**"):
    st.markdown('''
CelebMatch was created by STA 561 students at Duke University as part of a class project focused on machine learning. 
    ''')

st.header('**Find my look-a-like!**')

import pandas as pd
import time
import faiss
from deepface import DeepFace

#%% Test search first image
index = faiss.read_index("data/full/faiss_flat_ip.index")
df = pd.read_csv("data/full/info.csv")

#%%
import cv2
import ast
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
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
        normalized_user_embeddings = normalize([yourself_representation[0]["embedding"]],axis=1, norm='l2')
        t1 = time.time()
        st.write(f"Processed in {t1 - t0:.2f} seconds")

        # Perform search with the image representation
        index = faiss.read_index("data/full/faiss_flat_ip.index")
        match = index.search(np.asarray(normalized_user_embeddings), k=3)
        matching_rows = match[1][0, :]

        # Display matches
        for i in range(3):
            instance = df.iloc[matching_rows.tolist()[i]]
            name_list = ast.literal_eval(instance['name'])  # Convert the string representation of list to actual list
            name = name_list[0] if name_list else "Unknown"  # Extract the first item
            distance = match[0][0, i]
            full_path_string = instance['full_path']
            full_path = ast.literal_eval(full_path_string)
            clean_path = full_path[0]
            #img_path = "imdb_crop/" + clean_path
            img_url = "https://storage.googleapis.com/imdb_cropped_images/imdb_crop/" + clean_path

            # Use urllib to get the image from the URL
            resp = urllib.request.urlopen(img_url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            celeb_img = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Decode the image data
            #celeb_img = cv2.imread(img_path)
            celeb_img = cv2.cvtColor(celeb_img, cv2.COLOR_BGR2RGB)
            celeb_img = cv2.resize(celeb_img, (192, 192))
            st.write(f"{i + 1}. {name} (Distance: {distance:.2f})")
            st.image(celeb_img, use_column_width=False)

    except Exception as e:
        st.write("Failed to process the image:", e)
