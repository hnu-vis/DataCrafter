from flask import jsonify
from flask import Blueprint, request
import os
import json
import os
from application.views.case_utils.case_pets import PetsCase
from application.views.utils.detail_utils import transfer_image_path_fuzzy
from application.views.utils.prompt_utils import get_prompt_from_image_path, get_image_prompt_from_folder, add_prompt_by_GA, merge_add_prompts, \
    delete_prompt_by_GA, merge_delete_prompts
prompt = Blueprint('prompt', __name__)

@prompt.route('/getPrompt', methods=['GET'])
def get_prompt():
    name = request.args.get('name')
    prompt = {}
    if os.path.exists(PetsCase.prompt_json):
        with open(PetsCase.prompt_json, 'r') as f:
            prompt = json.load(f)
        return jsonify(prompt)
    else:
        prompt = get_image_prompt_from_folder(PetsCase.data_dir)
        print("Initializing to fetch all prompts:", prompt)
        with open(PetsCase.prompt_json, 'w') as f:
            json.dump(prompt, f)
    return jsonify(prompt)

@prompt.route('/getSelectPrompt', methods=['POST'])
def get_select_prompt():
    request_json = request.get_json()
    path_list = request_json['img_path']
    prompts_dict = get_prompt_from_image_path(path_list)
    return jsonify(prompts_dict)

# Retrieve a set of generated image paths based on the prompt/key
@prompt.route('/getImageByPrompt', methods=['POST'])
def get_image_by_prompt():
    request_json = request.get_json()
    key_list = request_json['Prompt']
    print("Prompt query received from the frontend:", key_list)
    image_list = []
    for root, _, files in os.walk(PetsCase.data_dir):
        for filename in files:
            if any(key in filename for key in key_list):
                image_list.append(os.path.join(root, filename))
    return jsonify(image_list)

# Delete the corresponding prompt object based on the key category and prompt
@prompt.route('/deletePrompt', methods=['POST'])
def delete_prompt():
    request_json = request.get_json()
    class_name = request_json['Class_name']
    key = request_json['Prompt']
    with open(PetsCase.prompt_json, 'r') as f:
        prompt = json.load(f)
    prompt[class_name] = [p for p in prompt[class_name] if p['text'] != key]
    with open(PetsCase.prompt_json, 'w') as f:
        json.dump(prompt, f)
    return jsonify(prompt)

# Update the corresponding prompt object based on the key category and old prompt
@prompt.route('/acceptPrompt', methods=['POST'])
def accept_prompt():
    request_json = request.get_json()
    class_name = request_json['Class_name']
    old_key = request_json['Original_text']
    new_key = request_json['newPrompt']
    print("Prompt to be updated:", class_name, old_key, new_key)
    with open(PetsCase.prompt_json, 'r') as f:
        prompt = json.load(f)
    prompt_index = {item['text']: i for i, item in enumerate(prompt.get(class_name, []))}
    
    if old_key in prompt_index:
        index = prompt_index[old_key]
        prompt[class_name][index]['text'] = new_key
        prompt[class_name][index]['original_text'] = new_key
        print("Updated prompt", prompt[class_name][index])
    else:
        new_prompt = {
            "text": new_key,
            "accepted": True,
            "original_text": new_key
        }
        if class_name not in prompt:
            prompt[class_name] = []
        prompt[class_name].insert(0, new_prompt)
        print("This prompt hasn't been seen before, adding:", new_prompt)
    with open(PetsCase.prompt_json, 'w') as f:
        json.dump(prompt, f)
    return jsonify(prompt)

# Using the add button on the scatter plot, generate elements not in the current collection based on user-selected image feedback, evolutionary algorithms, and GPT recommendations
@prompt.route('/addPrompt', methods=['POST'])
def add_prompt():
    request_json = request.get_json()
    path_list = request_json['img_path']
    path_list_to_dict, new_prompt = add_prompt_by_GA(path_list)
    merged_prompt = merge_add_prompts(path_list_to_dict, new_prompt)
    return jsonify(merged_prompt)

# Delete selected images using the delete button on the scatter plot, updating the prompt
@prompt.route('/deleteReplacePrompt', methods=['POST'])
def delete_replace_prompt():
    request_json = request.get_json()
    path_list = request_json['img_path']
    with open(PetsCase.prompt_json, 'r') as f:
        prompt = json.load(f)
    
    if PetsCase.step == 5:
        with open(PetsCase.tsne_embedding_dir, 'r') as f:
            embedding = json.load(f)
        for key in embedding:
            embedding[key] = {index: item for index, item in embedding[key].items() if len(item) > 2 and item[2] not in path_list}
        with open(PetsCase.tsne_embedding_dir, 'w') as f:
            json.dump(embedding, f)

    new_path_list = transfer_image_path_fuzzy(path_list, PetsCase.original_data_dir, PetsCase.data_dir)
    new_prompt = delete_prompt_by_GA(new_path_list) 
    merged_prompt = merge_delete_prompts(prompt, new_prompt)
    return jsonify(merged_prompt)
