import os
from datetime import datetime

# Define the directory containing the files
directory = 'trials'

# Iterate through each file in the directory
for filename in os.listdir(directory):
    # Skip files that don't match the old naming pattern
    if not filename.endswith('.npy') or '-' not in filename:
        continue

    # Extract the time and date parts
    time_part, date_part = filename.split('-')[0], filename.split('-')[1].split('.')[0]

    # Construct the old datetime object
    old_datetime_str = f"{time_part}-{date_part}"
    old_datetime = datetime.strptime(old_datetime_str, '%H%M-%d%m%y')

    # Create the new filename using the new convention
    new_datetime_str = old_datetime.strftime('%m%d-%H%M')
    new_filename = f"{new_datetime_str}.npy"

    # Rename the file
    old_filepath = os.path.join(directory, filename)
    new_filepath = os.path.join(directory, new_filename)
    os.rename(old_filepath, new_filepath)

    print(f"Renamed {filename} to {new_filename}")