from flask import Blueprint, request
from application.views.case_utils.case_pets import PetsCase

case = Blueprint("case", __name__)

@case.route('/changeCase', methods=['POST'])
def change_case():
    request_json = request.get_json()
    case = request_json['case']
    PetsCase.change_case(name=str(case))
    print(f"change to case {case}")
    return "success"


@case.route('/changeStep', methods=['POST'])
def change_step():
    request_json = request.get_json()
    step = request_json['step']
    PetsCase.update(step=int(step))
    print(f"change to step {step}")
    return "success"


@case.route('/changeScale', methods=['POST'])
def change_scale():
    request_json = request.get_json()
    scale = request_json['guidanceScale']
    totalGeneration = request_json['totalGeneration']
    print("change scale: ", scale)
    if int(scale) == 5:
        PetsCase.update(1)
        print("change scale to 5")
        return "success"
    if int(scale) == 100:
        PetsCase.update(2)
        print("change scale to 100")
        return "success"
    if int(scale) == 20:
        PetsCase.update(3)
        print("change scale to 20")
        return "success"
    PetsCase.update(step=int(totalGeneration))
    return "success"


@case.route('/deleteScale', methods=['POST'])
def delete_scale():
    request_json = request.get_json()
    clickindex = request_json['clickindex']
    if int(clickindex) == 1:
        PetsCase.update(4)
        print("delete scale 5和100")

    return "success"


@case.route('/processPromptByScatter', methods=['POST'])
def process_prompt_by_scatter():
    request_json = request.get_json()
    action = request_json['action']
    if action == 'delete':
        PetsCase.update(5)
        print("delete scatter and update prompt")
    if action == 'add':
        PetsCase.update(6)
        print("add prompt")
    if action == 'transfer':
        PetsCase.update(7)
        print("transfer prompt")

    return "success"
