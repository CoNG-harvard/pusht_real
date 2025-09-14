import numpy as np
import paramiko
import pickle

import gzip
import time
from PIL import Image
import cv2

import matplotlib.pyplot as plt

SERVER_HOST = '10.243.85.171'
USERNAME = 'hbalim'
PASSWORD = 'Balohalo123!'
REMOTE_UPLOAD_PATH = '/home/hbalim/wdir/FoundationPose/input_data.pkl.gz'
REMOTE_RESPONSE_PATH = '/home/hbalim/wdir/FoundationPose/output_data.pkl.gz'

class SFTPClient:
    def __init__(self, server_host, username, password, port=22):
        self.transport = paramiko.Transport((server_host, port))

        
        self.transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)
        print("[Client] Connected to server using password.")

    def send_data(self, data, remote_path):
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        with self.sftp.open(remote_path, 'wb') as f:
            f.write(gzip.compress(serialized))
        print(f"[Client] Sent data to {remote_path}")

    def receive_response(self, remote_path, poll_interval=2):
        for i in range(5):
            try:
                with self.sftp.open(remote_path, 'rb') as f:
                    compressed_data = f.read()
                response = pickle.loads(gzip.decompress(compressed_data))
                print("[Client] Response received.")
                self.sftp.remove(remote_path)
                return response
            except IOError as e:
                print("[Client] Waiting for response...")
                time.sleep(poll_interval)
        return None

    def close(self):
        self.sftp.close()
        self.transport.close()
        print("[Client] Connection closed.")
