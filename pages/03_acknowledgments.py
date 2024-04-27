import streamlit as st
st.header('**Acknowledgements**')

st.markdown('''
CelebParentMatch was brought into existence through the efforts of our team.

**Annika (amh227)** laid the foundation with our starter code which processed the images and created embeddings, all while juggling the joys and jitters of welcoming a new baby into her life.

**Fatima (fa83)** then extended that code to incorporate the [DeepFace library](https://github.com/serengil/deepface/tree/master). With the indispensable aids of the internet and Stack Overflow, and frequent recourse to the 'phone a friend' lifeline, she developed the code that calculates the distances between user images and celebrities using [FAISS](https://ai.meta.com/tools/faiss/).  

**Rafael (rpa9)** had the idea for the project, wrote the technical details and failed miserably in making it operational. Thanks Fatima and Annika for making it happen!

**Fatima (fa83)** and **Rafael (rpa9)** also crafted the press release and FAQ.

**Fatima (fa83)** and put together the streamlit app, scripts, and dockerfile.

''')