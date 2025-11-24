import gzip
import shutil
import os
import pandas as pd

# Step 0: Setting the directory
meta_file_path = r'C:\LocoGaze\data\metadata.csv'
meta_df = pd.read_csv(meta_file_path, nrows=1)
reldirectory = meta_df.at[0, 'reldir']

# Define the directory where all data files are located
data_directory = 'C:/LocoGaze/data/'  # Replace with your actual directory path
input_directory = data_directory + reldirectory

# Define paths for each of the .gz files and their output file names
archives = {
    'eventdata.gz': 'eventdata.txt',  # Output file name after extraction
    'gazedata.gz': 'gazedata.txt',    # Output file name after extraction
    'imudata.gz': 'imudata.txt'       # Output file name after extraction
}

# Iterate over each archive and extract it
for archive_name, output_name in archives.items():
    # Full path to the .gz file
    gzip_file_path = os.path.join(input_directory, archive_name)
    # Full path for the output file
    output_file_path = os.path.join(input_directory, output_name)

    # Check if the .gz file exists
    if os.path.exists(gzip_file_path):
        # Extract the .gz file
        with gzip.open(gzip_file_path, 'rb') as f_in:
            with open(output_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted {gzip_file_path} to {output_file_path}")
    else:
        print(f"File {gzip_file_path} does not exist.")
