from flask_socketio import emit, disconnect
from flask import request
from . import socketio
import time
import threading
import os
import base64

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

    # Simpan gambar ke folder
    save_dir = os.path.join("saved_frames", sid)
    os.makedirs(save_dir, exist_ok=True)

    # Buat nama file unik berdasarkan timestamp
    filename = f"frame_{int(time.time())}.png"
    file_path = os.path.join(save_dir, filename)

    # Decode base64 dan simpan
    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)

    with open(file_path, "wb") as f:
        f.write(binary_data)
    
    print(f"Gambar disimpan: {file_path}")

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