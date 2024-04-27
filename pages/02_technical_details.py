import streamlit as st

st.header('**Technical Details**')

st.markdown('''
First, we create the embeddings for each image in the IMDB-Faces dataset using DeepFace. More specifically, first we clean the dataset by removing pictures that don’t have faces in them, have more than one face or are of low quality (meaning a face score smaller than 3). Then, we extract the embeddings (a vector representation of the features) in each image using the VGG-Face algorithm as implemented in DeepFace. VGG-Face is a Very-Deep 16 layer regular convolutional neural network, first proposed in Parkhi et al. (2015), that reduces images to a 4096 vector of features. Its architecture is described in Figure 1.

After creating the embeddings, we use Faiss to create a search index that matches the embeddings extracted from each image with the original dataset. Faiss allows us to efficiently search for the maximal cosine similarity between a given input image and the IMDB-Faces dataset images, and it lets us relate this maximal element to its original format.

Therefore, the algorithm works as follows: Given an input image, first we use DeepFace to extract a 4096 vector of features (the same process done with the IMDB-Faces dataset). Then, we search for the male and female embedding in the dataset that has the largest cosine similarity with the input image. IMDB-Faces has images pre-labeled as either female or male, which facilitates this step. Finally, it returns the pair of images (male and female) that has the largest similarity with the input.

''')
st.image('figure01.jpg', caption='Figure 1: VGG-Face Architecture')
