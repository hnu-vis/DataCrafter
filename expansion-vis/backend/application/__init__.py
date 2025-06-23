from flask import Flask
from flask_cors import *
from flask_session import Session
from .views.metric import metric
from .views.distribution import distribution
from .views.detail import detail
from .views.prompt import prompt
from .views.case import case
from .views.data import data


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['JSON_SORT_KEYS'] = False
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config["SECRET_KEY"] = b'\xf4S\xef2R&\x06\xbd\xf0\xf3\xb5\x86o\xca\x95\x14\x8e\x0f\x8c\xd3;\\S6'
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    Session(app)

    app.register_blueprint(metric)
    app.register_blueprint(distribution)
    app.register_blueprint(detail)
    app.register_blueprint(prompt)
    app.register_blueprint(case)
    app.register_blueprint(data)

    return app
