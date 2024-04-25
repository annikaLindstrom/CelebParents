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

#%% Configure source and destination paths
sample_file_path = "df_with_embeddings.pkl"
embeddings_destination_path = "../data/full/embeddings.npy"
info_destionation_path = "../data/full/info.csv"

#%% Load all the data
df = pd.read_pickle(sample_file_path).reset_index()
#%% reduce rows
# Reduce df so that celebs with lots of entries are reduced to a max of three entries
# Step 1: Parse the 'name' column to strip extra characters and convert to string if needed
#df['name'] = df['name'].str.extract(r"\['(.*)'\]")

# Step 2: Group by 'name' and sample up to three entries from each group
#sampled_df = df.groupby('name').apply(lambda x: x.sample(min(len(x), 3))).reset_index(drop=True)


#%% Convert the embeddings to a
embeddings = np.asarray([x[0]["embedding"] for x in df["face_vector_raw"].tolist()])

#%% Write the most important columns to disk
df[
    ["dob", "gender", "name", "celeb_names", "celeb_id", "full_path", "photo_taken"]
].to_csv(info_destionation_path, index=False)
np.save(embeddings_destination_path, embeddings)
