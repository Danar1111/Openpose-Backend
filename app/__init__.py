from flask import Flask

def create_app():
    app = Flask(__name__)

    from .routes import pose_bp
    app.register_blueprint(pose_bp)

    return app
