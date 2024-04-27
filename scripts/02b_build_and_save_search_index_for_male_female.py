"""
This script will use the embeddings created in the previous script to
build a search-index, and the info.csv to fetch data about the matching images.

This script is designed to load precomputed embeddings (created in the previous script),
create a search-index with those embeddings using the FAISS library, and then save that index
to disk, along with the info.csv file.
"""
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from pathlib import Path
parent_path = Path(__file__).parent.parent
data_path = Path(__file__).parent.parent / 'data'

print("Finished loading libraries")

#%% Load the data
#embeddings_male = np.load("./data/full/embeddings_male.npy")
embeddings_male = np.load(data_path / 'full'/ 'embeddings_male.npy')
#embeddings_female = np.load("./data/full/embeddings_female.npy")
embeddings_female = np.load(data_path / 'full'/ 'embeddings_female.npy')

print("Finished loading the embeddings")

#%% normalize the embeddings
embeddings_male = normalize(embeddings_male,axis=1, norm='l2')
embeddings_female = normalize(embeddings_female,axis=1, norm='l2')

print("Finished normalizing the embeddings")

#%% Create an index with FAISS
d = 4096
#nlist = 100
#m = 16             # number of subquantizers
#k = 4

#%% For Male
# We use inner-product (IP) as this corresponds to cosine-similarity when the embeddings are normalized
index_male = faiss.IndexFlatIP(d)
index_male.add(embeddings_male)

print("Finished creating the male index")

#%% For Female
index_female = faiss.IndexFlatIP(d)
index_female.add(embeddings_female)

print("Finished creating the female index")
# # # tutorial: https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint#in-python
# quantizer = faiss.IndexFlatIP(d)
# index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
#                                     # 8 specifies that each sub-vector is encoded as 8 bits
# index.train(embeddings)
# index.add(embeddings)
print(f"Created male index for {index_male.ntotal} embeddings")
print(f"Created female index for {index_female.ntotal} embeddings")

#%% Save the FAISS indices
male_index_path = data_path / 'full' / 'faiss_flat_ip_male.index'
female_index_path = data_path / 'full' / 'faiss_flat_ip_female.index'
faiss.write_index(index_male, str(male_index_path))
faiss.write_index(index_female, str(female_index_path))
