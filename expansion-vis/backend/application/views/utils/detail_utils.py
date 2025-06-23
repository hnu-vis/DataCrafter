import os
'''
{
    "originalImagePath": 
    {
        "className": "Cat",
        "flag": 0,
        "generatedImages": [
            "path1":
                {
                "id": 0,
                "prompt": "mao",
                "flag": 0
                },
            "path2":
                {
                "id": 1,
                "prompt": "mao",
                "flag": 0
            },
        ]},
        ...
}
'''
def get_selected_image_list(img_paths, image_mapping):
    # Process selected img_paths
    select_image_list = {}
    for path in img_paths:
        is_selcted = False
        class_name = os.path.basename(os.path.dirname(path))
        # Retrieve the dictionary item for the current path
        dict_item, is_parent = get_image_item_from_map(image_mapping, path)
        key = list(dict_item.keys())[0]
        
        # Check if this item already exists
        if key not in select_image_list:
            # If processing for the first time, initialize with parent image information
            select_image_list[key] = {
                "className": class_name,
                "flag": is_parent,
                "generatedImages": []
            }
            # Process child images
            value_dict = dict_item[key]
            idx = 0
            for epoch in sorted(value_dict.keys(), key=int):  # Sort by epoch
                images = value_dict[epoch]
                # Append images in the form of path:prompt
                for expansion_image_path, prompt in images.items():
                    select_image_list[key]["generatedImages"].append({
                        expansion_image_path: {
                            "id" : idx,
                            "prompt": prompt,
                            "flag": 1 if (expansion_image_path == path) else 0
                        }
                    })
                    idx += 1
        else:  # If the image was already added, set flag to 1 for parent or child image
            if is_parent:
                select_image_list[key]["flag"] = 1
            else:
                for item in select_image_list[key]["generatedImages"]:
                    if path in item:
                        item[path]["flag"] = 1
    
    # Convert the dictionary to the final format
    new_dict = convert_dict(select_image_list)
    return new_dict

# Perform fuzzy matching on image paths based on filename
def get_all_image_paths(folder_path):
    """
    Retrieve all image paths in a folder and its subfolders.
    
    :param folder_path: Parent folder path
    :return: Collection of image paths
    """
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def transfer_image_path_fuzzy(path_list, original_folder_path, expansion_folder_path):
    """
    Perform fuzzy matching on image paths by filename and return full paths.
    
    :param path_list: List of paths to match
    :param original_folder_path: Parent folder path for original images
    :param expansion_folder_path: Parent folder path for expanded images
    :return: Array of matched full paths
    """
    # Retrieve all image paths from original_folder_path and expansion_folder_path
    original_image_paths = get_all_image_paths(original_folder_path)
    expansion_image_paths = get_all_image_paths(expansion_folder_path)

    # Combine all paths into one collection
    all_image_paths = original_image_paths + expansion_image_paths
    matched_paths = []

    # Match paths by filename in path_list against the collection of all image paths
    for path in path_list:
        filename = os.path.basename(path)
        for image_path in all_image_paths:
            if filename == os.path.basename(image_path):
                matched_paths.append(image_path)
                break  # Exit once a match is found, assuming filenames are unique

    return matched_paths

# Retrieve the data entry that corresponds to the specified image path from the map
def get_image_item_from_map(image_map, image_path):
    '''
    Retrieve the data entry that corresponds to the specified image path from the map.
    '''
    for key, value in image_map.items():
        if image_path in key:
            return {key: value}, 1
        for epoch, images in value.items():
            if image_path in images.keys():
                return {key: value}, 0
    print("No corresponding image found")
    return None

import base64
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        return None

# Conversion function
def convert_dict(original_dict):
    converted_dicts = []
    for key, value in original_dict.items():
        # Directly use the path instead of filename
        original_image_path = key  # Assume the provided path is used directly
        converted_dict = {
            "originalImagePath": original_image_path,
            "className": value["className"],
            "flag": value["flag"],
            "scroll": 0,  # Assume scroll value is 0
            "generatedImages": [
                {
                    "id": img[list(img.keys())[0]]["id"],
                    "path": list(img.keys())[0],
                    "prompt": img[list(img.keys())[0]]["prompt"],
                    "flag": img[list(img.keys())[0]]["flag"]
                }
                for img in value["generatedImages"]
            ]
        }
        converted_dicts.append(converted_dict)
    return converted_dicts
