from app import create_app, socketio
import app.ws_handlers

app = create_app()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)