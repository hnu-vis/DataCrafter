from flask import Blueprint, request
from flask import jsonify
from application.views.utils.detail_utils import get_selected_image_list, transfer_image_path_fuzzy
from application.views.database_utils.data import get_train2expansion_dict
from application.views.case_utils.case_pets import PetsCase

detail = Blueprint("detail", __name__)

from flask import send_from_directory

IMAGE_DIR = '/root/SD_Expansion_Vis/data'

@detail.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@detail.route("/getSelectedImage", methods=["POST"])
def get_detail_data():
    request_json = request.get_json()
    path_list = request_json['img_path']
    image_mapping = get_train2expansion_dict(PetsCase.original_data_dir, PetsCase.data_dir)
    new_path_list = transfer_image_path_fuzzy(path_list, PetsCase.original_data_dir, PetsCase.data_dir)
    total_img_list = get_selected_image_list(new_path_list, image_mapping)
    return jsonify(total_img_list)
