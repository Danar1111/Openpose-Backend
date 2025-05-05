from flask import Flask

def create_app():
    app = Flask(__name__, static_url_path="", static_folder="output_images")

    from .routes import pose_bp
    app.register_blueprint(pose_bp)

    return app
