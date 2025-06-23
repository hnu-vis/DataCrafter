import time
from flask import jsonify
from flask import Blueprint, request
import os
import json
from application.views.utils.process_data_utils import  find_image_paths_with_word_from_file, get_word_frequency, find_words_with_image_paths_from_file

from application.views.case_utils.case_pets import PetsCase

from application.views.database_utils.data import hide_files, unhide_files

distribution = Blueprint('distribution', __name__)

@distribution.route('/getDistributionView', methods=['GET', 'POST'])
def get_tsne_distribution_projection():
    request_json = request.get_json()
    num = request_json['num']
    name = request_json['name']
    option = request_json['option']
    get_only_expansion_data = request_json.get('get_only_expansion_data', None)
    get_expansion_data = get_only_expansion_data
    batch_size = request_json.get('batch_size', None)
    select_data_num = request_json.get('expansion_select_num', None)
    original_data_num = request_json.get('original_select_num', None)
    n_iter = request_json.get('n_iter', None)
    
    if batch_size is None:
        batch_size = 100
    if get_expansion_data is None:
        get_expansion_data = False

    if n_iter is None:
        n_iter = 1500

    if select_data_num is None:
        select_data_num = 1
    print('select_data_num', select_data_num)
    if not name or not option:
        return 'Missing name or option', 400
    if select_data_num <= 0 or select_data_num > 1:
        return 'select_data_num should be in (0, 1]', 400
    
    print("Fetching projection data from here:", PetsCase.tsne_embedding_dir)
    if not os.path.exists(PetsCase.tsne_embedding_dir):
        label_dict = {}
        return jsonify(label_dict)
    
    with open(PetsCase.tsne_embedding_dir, 'r') as f:
        label_dict = json.load(f)

    return jsonify(label_dict)

@distribution.route('/getImagePathByWord', methods=['POST'])
def get_image_paths_by_word():
    data = request.get_json()
    word = data.get('word')
    if word is None:
        return 'Missing word', 400
    print('The word for fetching paths:', word)
    image_path = find_image_paths_with_word_from_file(PetsCase.image_caption_json, word)
    return jsonify(image_path)

@distribution.route('/getWordsByImagePath', methods=['POST'])
def get_words_by_image_path():
    request_json = request.get_json()
    path_list = request_json['img_path']
    words = find_words_with_image_paths_from_file(PetsCase.image_caption_json, path_list)
    return jsonify(words)

@distribution.route('/getTreeWordByWord', methods=['POST'])
def get_tree_word_by_word():
    data = request.get_json()
    word = data.get('word')
    if PetsCase.step == 9:
        return jsonify([word])
    with open(PetsCase.tree_cut_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        if item['click_word'] == word:
            wordlist = item['selected_words']
    return jsonify(wordlist)

# Upload JSON data via button, save to file, and update
@distribution.route('/uploadJsonData', methods=['POST'])
def upload_json_data():
    request_json = request.get_json()
    jsondata = request_json['jsondata']
    with open(PetsCase.tsne_embedding_dir, 'w') as f:
        json.dump(jsondata, f)
    embedding_dir = '/home/embeddings/'
    filename = embedding_dir + 'embedding' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.json'
    with open(filename, 'w') as f:
        json.dump(jsondata, f)
    word_frequency = get_word_frequency(PetsCase.tsne_embedding_dir, PetsCase.word_frequency_json)
    return 'success'

# Extract all words from captions in the JSON file and return word frequency
@distribution.route('/getWordFrequency', methods=['GET'])
def get_word_frequency_json():
    if os.path.exists(PetsCase.word_frequency_json):
        with open(PetsCase.word_frequency_json, 'r') as f:
            word_frequency = json.load(f)
    else:
        word_frequency = get_word_frequency(PetsCase.tsne_embedding_dir, PetsCase.word_frequency_json)
        return jsonify(word_frequency)
    return jsonify(word_frequency)

@distribution.route('/deleteSelectImages', methods=['POST'])
def delete_select_images():
    request_json = request.get_json()
    image_paths = request_json['img_paths']
    original_paths = hide_files(image_paths)
    return jsonify(original_paths)

@distribution.route('/unDeleteSelectImages', methods=['POST'])
def undelete_select_images():
    request_json = request.get_json()
    original_paths = request_json['original_paths']
    print('Executing undo delete operation!')
    unhide_files(original_paths)
    return 'undo success!'
