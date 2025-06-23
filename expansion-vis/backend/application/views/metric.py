from flask import jsonify
from flask import Blueprint, request
import os
import json

from application.views.case_utils.case_pets import PetsCase

metric = Blueprint('metric', __name__)

@metric.route('/getMetricData', methods=['GET'])
def get_metric():
    # Retrieve the 'name' and 'option' parameters from the request
    name = request.args.get('name')
    option = request.args.get('option')

    # Check if 'name' or 'option' is missing, return an error if so
    if not name or not option:
        return 'Missing name or option', 400
    metric_data_dir = PetsCase.metric_dir
    data_dir = PetsCase.data_dir

    try:
        # If the metric data directory does not exist, return an empty list
        if not os.path.exists(metric_data_dir):
            return jsonify([])
        # Load the metric data file and return its contents
        with open(metric_data_dir, 'r') as file:
            metric_list = json.load(file)
        return jsonify(metric_list)
    except FileNotFoundError:
        # Return an error if the metric file is not found
        return 'Metric file not found', 404

@metric.route('/getClickedIndices', methods=['GET'])
def get_clicked_indices():
    # If the processing step is less than 4, return an empty list
    if int(PetsCase.step) < 4:
        return jsonify([])
    # Otherwise, return a list containing indices [0, 1]
    return jsonify([0, 1])
