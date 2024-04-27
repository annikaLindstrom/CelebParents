FROM python:3.10.14-bullseye

# see https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY Requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY data/full/info.csv data/full/info.csv
COPY data/full/faiss_flat_ip.index data/full/faiss_flat_ip.index
COPY data/full/faiss_flat_ip.index data/full/faiss_flat_ip_female.index
COPY data/full/faiss_flat_ip.index data/full/faiss_flat_ip_male.index
COPY celeb_parents.py streamlit_app.py

EXPOSE 8501
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
