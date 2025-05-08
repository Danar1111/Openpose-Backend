from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)

    from .routes import pose_bp
    app.register_blueprint(pose_bp)

    socketio.init_app(app)

    return app
