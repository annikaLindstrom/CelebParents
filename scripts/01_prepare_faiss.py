
"""
This script will load the pre-calculated embeddings dataframe
that were created using the data_processing_using_tutorial.ipynb
and store it in two separate files in disk:

1. embeddings.npy: is a Nx4096 numpy-array with all the embeddings
2. info.csv: information about each image such as the name of the
   celebrity, the path to the image etc.

In the next step (file: 02_build_and_save_search_index.py), we will use
the embeddings created here to build a search-index, and the info.csv to
fetch data about the matching images.
"""

import pandas as pd
import numpy as np
from pathlib import Path
parent_path = Path(__file__).parent.parent
data_path = Path(__file__).parent.parent / 'data'

print(f"finished loading libraries")

#%% Configure source and destination paths
sample_file_path = parent_path/"df_with_embeddings.pkl"
embeddings_destination_path = data_path / "full" / "embeddings.npy"
info_destination_path = data_path / "full" / "info.csv"
info_male_destination_path = data_path / "full" / "info_male.csv"
info_female_destination_path = data_path / "full" / "info_female.csv"

embeddings_male_destination_path = data_path / "full" /"embeddings_male.npy"
embeddings_female_destination_path = data_path / "full" /"embeddings_female.npy"

#%% Load all the data
df = pd.read_pickle(sample_file_path).reset_index()

print(f"finished loading dataframe")

#%% Convert the embeddings to a
embeddings = np.asarray([x[0]["embedding"] for x in df["face_vector_raw"].tolist()])
print(f"finished pulling embeddings out of dataframe")

#%% Write the most important columns to disk
df[
    ["dob", "gender", "name", "celeb_names", "celeb_id", "full_path", "photo_taken"]
].to_csv(info_destination_path, index=False)

np.save(embeddings_destination_path, embeddings)

print(f"finished saving data")

#%% Create Female embeddings
df_female = df[df['gender'] == 0].reset_index(drop=True)
df_female.to_csv(info_female_destination_path, index=False)
embeddings_female = np.asarray([x[0]["embedding"] for x in df_female["face_vector_raw"].tolist()])
np.save(embeddings_female_destination_path, embeddings_female)

print(f"finished saving female info and embeddings")

#%% Create Male embeddings
df_male = df[df['gender'] == 1].reset_index(drop=True)
df_male.to_csv(info_male_destination_path, index=False)
embeddings_male = np.asarray([x[0]["embedding"] for x in df_male["face_vector_raw"].tolist()])
np.save(embeddings_male_destination_path, embeddings_male)
print(f"finished saving male info and embeddings")

print(f"script completed")
