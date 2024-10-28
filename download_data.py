import kagglehub
import os
import shutil

dataset_name = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
data_directory = "data"
os.makedirs(data_directory, exist_ok=True)

path = kagglehub.dataset_download(dataset_name)

for file in os.listdir(path):
    shutil.move(os.path.join(path, file), os.path.join(data_directory, file))

print("Dataset files downloaded to:", path)
