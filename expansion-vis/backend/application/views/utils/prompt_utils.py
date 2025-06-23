import os
import re
from collections import OrderedDict

def get_detail_from_expansion_image(image_path):
    match = re.match(r'epoch(\d+)_split_\d+_name_(.+)_prompt_(.+)_expanded_\d+.png', image_path)
    if match:
        epoch = match.group(1)
        original_filename = match.group(2)
        prompt = match.group(3)
        return epoch, original_filename, prompt
    print("Error: Cannot get information from file name")
    return None, None, None

# Get the corresponding prompt from the given set of paths
def get_prompt_from_image_path(path_list):
    prompts_dict = {}
    for image_path in path_list:
        class_name = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        if not file_name.startswith('epoch'):
            continue
        epoch, original_filename, prompt = get_detail_from_expansion_image(file_name)
        if class_name not in prompts_dict:
            prompts_dict[class_name] = []
        prompts_dict[class_name].append({
            "text": prompt,
            "accepted": True,
            "original_text": prompt
        })
    for class_name in prompts_dict:
        prompts_dict[class_name] = [dict(t) for t in {tuple(d.items()) for d in prompts_dict[class_name]}]
    ordered_prompts_dict = prompts_dict
    if 'Beagle' in prompts_dict:
        order = ['Beagle', 'Bombay', 'Shiba_inu', 'Havanese', 'Bengal', 'Persian', 'Birman', 'Pug', 'Russian_blue', 'Samoyed']
        ordered_prompts_dict = OrderedDict()
        for key in order:
            if key in prompts_dict:
                ordered_prompts_dict[key] = prompts_dict[key]

    return ordered_prompts_dict

# Retrieve all prompts by getting all paths in the given folder
def get_image_prompt_from_folder(input_folder):
    image_paths = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Only process image files
                image_paths.append(os.path.join(root, file))
    return get_prompt_from_image_path(image_paths)

# Add a prompt using genetic algorithm. This is generated later; save and load from file. 
# Returns the current dictionary and the new prompt dictionary, aiming to display only added prompts in the selection area.
def add_prompt_by_GA(path_list):
    prompts_dict = {}
    path_list_to_dict = {}
    for image_path in path_list:
        class_name = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        if not file_name.startswith('epoch'):
            continue
        epoch, original_filename, prompt = get_detail_from_expansion_image(file_name)
        new_prompt = prompt + " add color(temporary)"
        if class_name not in prompts_dict:
            prompts_dict[class_name] = []
            path_list_to_dict[class_name] = []
        path_list_to_dict[class_name].append({
            "text": prompt,
            "accepted": True,
            "original_text": prompt
        })
        prompts_dict[class_name].append({
            "text": new_prompt,
            "accepted": False,
            "original_text": new_prompt
        })

    return path_list_to_dict, prompts_dict

# Delete a prompt using genetic algorithm. Functionally, this should replace the prompt.
# This is generated on the backend, so save and load from file as needed.
def delete_prompt_by_GA(path_list):
    delete_image_by_path(path_list) 
    prompts_dict = {}
    for image_path in path_list:
        class_name = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        if not file_name.startswith('epoch'):
            continue
        epoch, original_filename, prompt = get_detail_from_expansion_image(file_name)
        new_prompt = prompt + " cat/dog(temporary)"
        if class_name not in prompts_dict:
            prompts_dict[class_name] = []
        prompts_dict[class_name].append({
            "text": new_prompt,
            "accepted": False,
            "original_text": prompt
        })
    for class_name in prompts_dict:
        prompts_dict[class_name] = [dict(t) for t in {tuple(d.items()) for d in prompts_dict[class_name]}]

    return prompts_dict

# Merge new prompts into the current prompt set
def merge_add_prompts(current_prompts, new_prompts):
    merged_prompts = {}

    for class_name, new_prompt_list in new_prompts.items():
        merged_prompts[class_name] = new_prompt_list + current_prompts[class_name]

    return merged_prompts

# Replace the original text with modified text in the prompt, if they match. 
# If matched, replace with the new prompt and move it to the front.
def merge_delete_prompts(current_prompts_dict, new_prompts_dict):
    result_dict = {}

    for key, new_prompts in new_prompts_dict.items():
        updated_prompts = []
        unchanged_prompts = []

        if key in current_prompts_dict:
            current_prompts = current_prompts_dict[key]

            current_mapping = {prompt['text']: prompt for prompt in current_prompts}

            for new_prompt in new_prompts:
                original_text = new_prompt['original_text']
                if original_text in current_mapping:
                    updated_prompts.append(new_prompt)
                else:
                    unchanged_prompts.append(current_mapping.get(original_text, new_prompt))
        else:
            unchanged_prompts = new_prompts

        result_dict[key] = updated_prompts + unchanged_prompts

    return result_dict

# Delete the image files at the specified paths
def delete_image_by_path(path_list):
    for path in path_list:
        # os.remove(path)
        print("Remove file: ", path)
