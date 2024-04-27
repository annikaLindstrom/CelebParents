import streamlit as st

st.header('**FAQs**')

with st.expander("**What is CelebParentMatch?**"):
    st.markdown('''
        CelebParentMatch is your digital mirror into the world of fame, using facial recognition to match your photo with a celebrity who shares your star-worthy features. Just upload a picture, and see which celebrity pair you might have descended from.
    ''')

with st.expander("**How does CelebParentMatch work?**"):
    st.markdown('''
        Simply upload your photo through our app, and our algorithm will analyze your facial features using the IMDB dataset to find the celebrities whose features most closely resemble yours. To do this, we first create a vector of features (i.e., embeddings) of the faces in the IMDB dataset. We use the DeepFace library for that. Next, in order to find look-a-likes, we have to find faces that are close matches to that of the user. To do this, we use FAISS, an advanced library for efficient similarity search and clustering of dense vectors. After extracting the embeddings of your photo using the DeepFace library, these embeddings are then compared against the pre-processed embeddings of celebrity faces from the IMDB dataset stored in our FAISS index. FAISS efficiently searches through these vectors to find the closest matches based on the cosine similarity of the embeddings. This process not only ensures fast results but also maintains high accuracy in identifying the celebrity that most resembles you. Once the closest celebrity matches are found, the result is displayed to you instantly on our app.
        ''')

with st.expander("**Why am I getting photos of the same gender? I thought you were meant to show me look alikes of both genders**"):
    st.markdown('''
Unfortunately, our IMDB dataset has multiple images for each actor, and some photos are wrongly assigned. For instance, Sacha Baron Cohen has a female actress assigned to his folder. We leave the work of fixing this issue as a future development.
''')

with st.expander("**Why does the app assume two gender-specific parents (male and female)?**"):
    st.markdown('''
We recognize that the structure of families can be diverse, and not all are represented by the traditional model of one male and one female parent. The current design of our CelebParentMatch app uses a dataset that categorizes celebrities into male and female groups, which limits our ability to directly match beyond these binary categories. We're aware this does not encompass all family structures and gender identities.

In future updates, we hope to incorporate a broader perspective that better reflects the diversity of family compositions and embraces a more inclusive approach to gender identity. Our goal is to ensure our app can celebrate all forms of family and identity, aligning more closely with real-world diversity.
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

