import os
import shutil
import pickle
import re
import json

import random
import math

def copy_select_images(class_path, files, select_data_num, output_class_path):
    os.makedirs(output_class_path, exist_ok=True)
    selected_files = random.sample(files, math.ceil(select_data_num * len(files)))
    # Copy to the new folder
    for file in selected_files:
        file_path = os.path.join(class_path, file)
        output_file_path = os.path.join(output_class_path, file)
        shutil.copy(file_path, output_file_path)
    return selected_files

# Save historical data by providing the absolute file path and the data to save
def save_history_data(file_dir, save_file):
    try:
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        with open(file_dir, 'wb') as f:
            pickle.dump(save_file, f)
            print("Successfully saved historical data at " + file_dir)
    except Exception as e:
        print(f"An error occurred while saving the history data: {e}")


def find_image_paths_with_word_from_file(file_path, word):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    matching_image_paths = set()
    
    # Iterate through each entry in the JSON data
    for item in data:
        if 'image_path' in item.keys():
            image_path = item['image_path']
        else:
            image_path = item['file_name']
        
        if 'nouns' in item.keys():
            caption_list = item['nouns']
            if word in caption_list:
                matching_image_paths.add(image_path)
        else:
            caption = item['caption']
            # Check if the specified word is contained within the caption
            # if word.lower() in caption.lower():
            if re.search(r'\b' + re.escape(word.lower()) + r'\b', caption.lower()):  # Use regex to match the whole word, not just letters, to avoid "car" matching "scarf"
                print("Found this caption:", caption)
                matching_image_paths.add(image_path)
    
    # Remove duplicates
    matching_image_paths = list(matching_image_paths)
    print(f"Found {len(matching_image_paths)} images with the word '{word}'")
    return matching_image_paths

def find_words_with_image_paths_from_file(file_path, image_paths: list):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading the JSON file: {e}")
        return []

    matching_words = set()

    for item in data:
        if 'image_path' in item.keys():
            image_path = item['image_path']
        else:
            image_path = item['file_name']
        words = item.get('nouns', [])
        
        if image_path in image_paths:
            if isinstance(words, list):  # Check if `words` is a list
                matching_words.update(words)
            else:
                matching_words.add(words)

    return list(matching_words)

def get_word_frequency(input_file, output_file):
    # Read JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Extract data from the "Text" section
    text_data = data.get("Text", {})
    # print(text_data)
    word_frequency = {entry[2]: entry[3] for entry in text_data.values()}

    # Optional: Write the result to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(word_frequency, outfile, indent=4)
    return word_frequency
