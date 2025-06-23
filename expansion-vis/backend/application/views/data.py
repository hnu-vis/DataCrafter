from flask import Blueprint, jsonify
from application.views.case_utils.case_pets import PetsCase
import os
import json

data = Blueprint("data", __name__)

@data.route('/getImagesData', methods=['GET'])
def get_images():
    if not os.path.exists(PetsCase.images):
        result = []
        return jsonify(result)
    
    with open(PetsCase.images, 'r') as f:
        return jsonify(json.load(f))
    
@data.route('/getWordsData', methods=['GET'])
def get_words():
    if not os.path.exists(PetsCase.words):
        result = []
        return jsonify(result)
    
    with open(PetsCase.words, 'r') as f:
        return jsonify(json.load(f))
