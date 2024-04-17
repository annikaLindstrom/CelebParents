import scipy
import pandas as pd
import numpy as np
import cv2
import time
import face_recognition


def findFaceRepresentation(img):
    return face_recognition.face_encodings(img)


def getImage(image_path):
    return cv2.imread("imdb_crop/" + image_path[0])


# Load Data from the .mat into a pandas dataframe
mat = scipy.io.loadmat('imdb_crop/imdb.mat')
columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score",
           "celeb_names", "celeb_id"]
instances = mat['imdb'][0][0][0].shape[1]
df = pd.DataFrame(index=range(0, instances), columns=columns)

for i in mat:
    if i == "imdb":
        current_array = mat[i][0][0]
        for j in range(len(current_array)):
            df[columns[j]] = pd.DataFrame(current_array[j][0])

# remove pictures that don't have face
df = df[df['face_score'] != -np.inf]

# remove pictures with more than one face
df = df[df['second_face_score'].isna()]

# only take high quality photos
df = df[df['face_score'] >= 3]


t0 = time.time()
selectedData = df.iloc[0:90000]
selectedData['image'] = selectedData['full_path'].apply(getImage)

print(time.time()-t0)

selectedData['embed'] = selectedData['image'].apply(findFaceRepresentation)

# save file
selectedData.to_csv('celeb_images.csv', index=False)

