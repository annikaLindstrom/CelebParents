# How to run our app

## Using Docker

1. Download and install Docker for your operating system. You can find the installation packages and instructions on the official [docker page](https://hub.docker.com/)
2. Make sure that Docker is actually running (i.e, open the desktop app). You might see a Docker icon in your system tray (for Windows) or the top menu bar (for macOS). This should tell you that Docker is active.
3. Open up your terminal or command prompt. 
4. Go to the directory (i.e. folder) that contains this repository. If you have cloned or downloaded this repository to your desktop, the command might look like this:
```shell 
cd path/to/your/repository
```

5. Once you are in your working directory in the terminal, build the Dockerfile by running:

```shell
docker build . -t celebmatch
```

```shell
docker run --rm -p 8501:8501 celebmatch
```

You should then see something like this in the output:

```
CelebParents git:(main) ✗ docker run --rm -p 8501:8501 celebmatch

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  URL: http://0.0.0.0:8501



``` 
Copy and paste the URL as it is written in your terminal.

## Using conda locally

1. Make sure that you have Anaconda or Miniconda installed. [Click here for download options](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Open up terminal, and go to the folder that holds this repository. If you have cloned or downloaded this repository to your desktop, the command might look like this:
```shell 
cd path/to/your/repository
```

3. In your terminal, create a new Conda environment by running:

```shell
conda env create 
```

4. Run the application using Streamlit. In the terminal, execute:
```shell
streamlit run streamlit_app.py
```

This should open up your browser, and you should be able to use the app.

# Information on the different files

## In the `script` folder you will find 4 scripts:

1. `00_data_processing_using_tutorial.ipynb` 
This notebook loads, cleans, creates embeddings from the cropped IMDB images dataset, stores them in a DataFrame, and saves the output as a pickle. It uses some code from this [tutorial](https://sefiks.com/2019/05/05/celebrity-look-alike-face-recognition-with-deep-learning-in-keras/). 
I **do not** recommend running it, as it takes over 6 hours to do so. 
I have already created the embeddings and I've indexed them using FAISS. 

2. `01_prepare_faiss.py` This script handles and organizes the face image embeddings that we created in the previous notebook. Its main role is to extract these embeddings from our DataFrame and save them along with related metadata into structured files. We then use these files in the script below, in order to build a search index and to make it easier (and faster) to retrieve look-a-likes based on facial similarity.

3. `02_build_and_save_search_index.py` This script uses the FAISS library to create a search index with face embeddings. It uses normalized embeddings to build a cosine-similarity-based search index, making it super efficient to retrieve images based on facial similarity.

4. `03_use_index_on_test_image.py` The script performs image matching by finding the closest celebrity look-alikes to a test image using precomputed embeddings and a FAISS search index, created in the previous script. To find celeb look a likes to a test image, we extract facial embeddings from the test image, normalize them, and query the index to find the nearest celebrity faces based on those embeddings.

## `streamlit_app.py` script

This is the file that contains all the code for our app. 

It includes the Press Release, the FAQs, and the CelebMatch app itself. 


