import zipfile
import os

# Path to the downloaded zip file fastai
zip_file_path = 'fer2013.zip'
extract_folder = 'path_to_extracted_folder'

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f"Files extracted to {extract_folder}")
