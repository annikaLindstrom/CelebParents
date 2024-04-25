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

#%% Load the data
embeddings = np.load("./data/full/embeddings.npy")

#%% normalize the embeddings
embeddings = normalize(embeddings,axis=1, norm='l2')

#%% Create an index with FAISS
d = 4096
#nlist = 100
#m = 16             # number of subquantizers
#k = 4

# We use inner-product (IP) as this corresponds to cosine-similarity when the embeddings are normalized
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# # # tutorial: https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint#in-python
# quantizer = faiss.IndexFlatIP(d)
# index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
#                                     # 8 specifies that each sub-vector is encoded as 8 bits
# index.train(embeddings)
# index.add(embeddings)
print(f"Created index for {index.ntotal} embeddings")
#%% Save the FAISS indices
faiss.write_index(index, "./data/full/faiss_flat_ip.index")
