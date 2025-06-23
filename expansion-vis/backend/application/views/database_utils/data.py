import os
import shutil
import PIL
import time
from global_parameter import log_dir
import re


# Get the file extension of the original dataset images
def get_file_ext(directory):
    '''
    Retrieve the file extension of images in the first subfolder.
    '''
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                return os.path.splitext(file)[1]
    return ".jpg"

# Create a dictionary based on original and generated dataset images for easy lookup
def get_train2expansion_dict(original_folder, generated_folder):
    '''
    {
        'original_image1.jpg': {
            'epoch1':
                {
                    'expanded_image1.png': prompt1,
                    'expanded_image2.png': prompt2
                },
            'epoch2':
                {
                    'expanded_image3.png': prompt3,
                    'expanded_image4.png': prompt4
                },
                ...
            ...
        }
    Note: In the code, "match" does not have "use_time" at this stage; it may need to be added later.
    '''
    image_mapping = {}
    file_ext = get_file_ext(original_folder)  
    print("file_ext", file_ext) 
    # Iterate through generated folder
    for root, _, files in os.walk(generated_folder):
        for filename in files:
            epoch, original_filename, prompt = get_detail_from_expansion_image(filename)
            original_filename += file_ext

            # Combine original folder path with filename
            original_image_folder = os.path.join(original_folder, os.path.basename(root))

            # Build the key for the dictionary
            key = os.path.join(original_image_folder, original_filename)

            # Add generated filename to dictionary
            if key in image_mapping:
                if epoch in image_mapping[key]:
                    image_mapping[key][epoch][os.path.join(root, filename)] = prompt
                else:
                    image_mapping[key][epoch] = {os.path.join(root, filename): prompt}
            else:
                image_mapping[key] = {
                    epoch: {os.path.join(root, filename): prompt}
                }
    return image_mapping

# Extract information from the filename/path of an expanded image
def get_detail_from_expansion_image(image_path):
    '''
    Retrieve details from the filename/path of an expanded image.
    '''
    match = re.match(r'epoch(\d+)_split_\d+_name_(.+)_prompt_(.+)_expanded_\d+.png', image_path)
    if match:
        epoch = match.group(1)
        original_filename = match.group(2)
        prompt = match.group(3)
        return epoch, original_filename, prompt
    print("Error: Unable to retrieve information from filename")
    return None, None, None

# Convert 'Russian_Blue_220' to 'Russian_Blue'
def get_class_name_from_filename(filename):
    match = re.match(r'([A-Za-z_]+)_[0-9]+', filename)
    if match:
        return match.group(1)
    return None

# Hide files by moving them to a hidden directory
def hide_files(image_paths, hidden_dir=".hidden"):
    # Create hidden directory
    if not os.path.exists(hidden_dir):
        os.makedirs(hidden_dir)
    # Move files to hidden directory and record original paths
    original_paths = {}
    for path in image_paths:
        if os.path.exists(path):
            original_path = os.path.join(hidden_dir, os.path.basename(path))
            shutil.move(path, original_path)
            original_paths[original_path] = path
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            write_log(log_dir, str(now) + ": Executed hide/delete for file: " + str(path))
        else:
            print("File does not exist:", path)
    return original_paths

# Unhide files by moving them back to their original locations
def unhide_files(original_paths):
    # Move files from hidden directory back to original location
    for hidden_path, original_path in original_paths.items():
        if os.path.exists(hidden_path):
            shutil.move(hidden_path, original_path)
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            write_log(log_dir, str(now) + ": Undo hide/delete for file: " + str(hidden_path))
        else:
            print("File does not exist:", hidden_path)

# Function to write logs in append mode
def write_log(log_file, content):
    try:
        with open(log_file, 'a') as f:
            f.write(content + '\n')
            print("Log entry:", content)
    except Exception as e:
        print(f"Error writing to log file: {e}")
