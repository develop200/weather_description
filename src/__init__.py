from flask import Flask


def create_app():
    app = Flask(__name__)
    app.app_context().push()
    
    from .api import api
    api.init_app(app)

    return app
