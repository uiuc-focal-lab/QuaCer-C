# server.py

import socket
import json
import time
import argparse
from utils import *

checker_llm = None
def get_args():
    parser = argparse.ArgumentParser('Run Subgraph experiments')
    parser.add_argument('--checker_llm_device', type=str, default='cuda:1')

    return parser.parse_args()
def process_request(data):
    # Simulate processing a request by modifying the input data
    question = data['question']
    correct_answer = data['correct_answer']
    model_answer = data['model_answer']
    result = check_answer(question, correct_answer, model_answer)
    data['result'] = result
    return data

def check_answer(question, correct_answer, model_answer):
    global checker_llm
    return checker_llm.raw_checker(question=question, correct_ans=correct_answer, model_ans=model_answer)[0]

def main():
    host = 'localhost'
    port = 12345

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()

        print("Server is listening on", port)

        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                data = conn.recv(1024)
                if not data:
                    break
                # Process the request
                print('blocking...')
                try:
                    request_data = json.loads(data.decode('utf-8'))
                    response_data = process_request(request_data)
                    # Send response back to client
                    print('Sending response')
                    print(response_data)
                    conn.sendall(json.dumps(response_data).encode('utf-8'))
                except Exception as e:
                    print(e)
                    response_data = {}
                    response_data['result'] = 0
                    conn.sendall(json.dumps(response_data).encode('utf-8'))

if __name__ == '__main__':
    args = get_args()
    checker_llm = MistralChecker('mistralai/Mistral-7B-Instruct-v0.2', args.checker_llm_device)
    main()
