from flask_socketio import emit, disconnect
from flask import request
from . import socketio
import time
import threading
import base64
import shutil
from app.services.ws_pose_estimation import process_pose_from_bytes
from app.utils.summarize_results import summarize_results

clients = {}
session_results = {}
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

    if sid in session_results:
        all_results = session_results[sid]

        if all_results:
            summary = summarize_results(all_results)
            print(f"Summary for {sid}:")
            print(summary)

        del session_results[sid]

@socketio.on('frame')
def handle_frame(data):
    sid = request.sid
    clients[sid]['last_active'] = time.time()

    image_data = data['image']
    header, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    print(f"Received frame from {sid}")

    result = process_pose_from_bytes(image_bytes)

    if sid not in session_results:
        session_results[sid] = []

    session_results[sid].append(result)
    
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