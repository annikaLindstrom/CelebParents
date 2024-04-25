
"""
This script finds the celebrities with shortest distances from a test image.
"""
#%%  load libraries
import pandas as pd
import time
import faiss
from deepface import DeepFace

#%% Test search first image
index = faiss.read_index("./data/full/faiss_flat_ip.index")
df = pd.read_csv("./data/full/info.csv")

#%% get test image
from sklearn.preprocessing import normalize
t0 = time.time()
yourself_representation = DeepFace.represent(
    img_path="./test_faces/angelina.jpeg",
    detector_backend="opencv",
)

normalized_user_embeddings = normalize([yourself_representation[0]["embedding"]],axis=1, norm='l2')

t1 = time.time()
print(f"Finished in {t1-t0} seconds")

# %% search index
import numpy as np
t0 = time.time()
match = index.search(np.asarray(normalized_user_embeddings), k=3)
matching_rows = match[1][0, :]
t1 = time.time()

print(f"Finished in {t1-t0} seconds")

# %%
for i in range(3):
    print("Number {i + 1}:")
    print(df.iloc[matching_rows.tolist()[i]].transpose())

#%%
import cv2
import matplotlib.pyplot as plt
import ast

for i in range(3):
   instance = df.iloc[matching_rows.tolist()[i]]
   name = instance['name']
   distance = match[0][0,i]
   full_path_string = instance['full_path']
   full_path = ast.literal_eval(full_path_string)
   clean_path = full_path[0]
   #img = cv2.imread("./imdb_crop/" + clean_path)
   img = cv2.imread("https://storage.googleapis.com/imdb_cropped_images/imdb_crop/" + clean_path)
   print(i,".",name," (",distance,")")
   plt.axis('off')
   plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
   plt.show()

#%%
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import ast

for i in range(3):
    instance = df.iloc[matching_rows.tolist()[i]]
    name = instance['name']
    distance = match[0][0, i]
    full_path_string = instance['full_path']
    full_path = ast.literal_eval(full_path_string)
    clean_path = full_path[0]

    # Build the full URL to the image
    img_url = "https://storage.googleapis.com/imdb_cropped_images/imdb_crop/" + clean_path

    # Use urllib to get the image from the URL
    resp = urllib.request.urlopen(img_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Decode the image data

    # Check if the image was loaded successfully
    if img is not None:
        print(i, ".", name, " (", distance, ")")
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for displaying
        plt.show()
    else:
        print(f"Failed to load image from {img_url}")