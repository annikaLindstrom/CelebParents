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
2. Open up terminal, and go to the folder that holds this repository. This respository should include `environment.yml`.
3. Create a new Conda environment by running:

```shell
conda env create 
```

4. Run the application using Streamlit. In the terminal, execute:
```shell
streamlit run streamlit_app.py
```

This should open up your browser, and you should be able to use the app. 