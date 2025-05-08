from flask_socketio import emit, disconnect
from . import socketio
import time
import threading

clients = {}

TIMEOUT = 300  # 5 menit

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    clients[sid] = {'last_active': time.time()}
    print(f"Client {sid} connected")

    emit('control', {'command': 'start_capture', 'interval': 5000})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in clients:
        del clients[sid]
    print(f"Client {sid} disconnected")

@socketio.on('frame')
def handle_frame(data):
    sid = request.sid
    clients[sid]['last_active'] = time.time()

    image_data = data['image']  # base64 image
    print(f"Received frame from {sid}")

    # Simulasi proses (bisa panggil process_pose_from_bytes disini)
    result = "Dummy Processed"
    
    emit('processed_result', {'result': result})

def monitor_clients():
    while True:
        now = time.time()
        for sid in list(clients.keys()):
            if now - clients[sid]['last_active'] > TIMEOUT:
                print(f"Auto disconnect {sid} (idle)")
                socketio.emit('auto_disconnect', {'reason': 'Idle timeout'}, room=sid)
                disconnect(sid)
                del clients[sid]
        time.sleep(60)

# Mulai background task saat import
threading.Thread(target=monitor_clients, daemon=True).start()