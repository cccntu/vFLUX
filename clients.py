import os
import requests
import sys
import base64
import multiprocessing
import time
import signal
import argparse

# Global variables
global_count = multiprocessing.Value('i', 0)
start_time = multiprocessing.Value('d', 0.0)
stop_event = multiprocessing.Event()
max_requests = multiprocessing.Value('i', 0)

def call_generate_image(text: str, url: str = "http://localhost:8001/generate_image", save_path=None):
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}
    try:

        start = time.time()
        response = requests.post(url, json=payload, headers=headers)
        end = time.time()
        if response.status_code == 200:
            if save_path is not None:
                # Decode base64 and save the image
                img_data = base64.b64decode(response.json()["image"])
                # Create parent directories if they don't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(img_data)

            return {"latency": end - start, "save_path": save_path}
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def worker_process(process_id):
    while not stop_event.is_set():
        # Check if we should make another request
        with global_count.get_lock():
            if max_requests.value > 0 and global_count.value >= max_requests.value:
                stop_event.set()
                return
            # Pre-increment the counter before making the request
            current_count = global_count.value
            if max_requests.value > 0 and current_count >= max_requests.value:
                stop_event.set()
                return
            global_count.value += 1
            current_idx = global_count.value
            
        # Only make the request if we successfully incremented the counter

        output = call_generate_image(f"a cat holding a sign that says {process_id}", save_path=f"outputs/{process_id}.png")
        request_latency = output["latency"]
        print(f"Request id: {current_idx}. Request latency: {request_latency:.4f} seconds, save path: {output['save_path']}")
        sys.stdout.flush()
        time.sleep(0.1)  # Small delay to prevent overwhelming the server

def signal_handler(signum, frame):
    print("Interrupt received, stopping processes...")
    stop_event.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-client image generation script")
    parser.add_argument("--max-requests", type=int, default=4, help="Maximum number of requests to send (0 for unlimited)")
    parser.add_argument("--num-processes", type=int, default=4, help="Number of concurrent worker processes")
    args = parser.parse_args()
    
    max_requests.value = args.max_requests
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Record start time
    start_time.value = time.time()
    
    # Start worker processes
    processes = []
    for i in range(args.num_processes):
        p = multiprocessing.Process(target=worker_process, args=(i,))
        p.start()
        processes.append(p)
    
    # Wait for processes to finish
    for p in processes:
        p.join()
    
    # Set stop event to ensure status thread exits
    stop_event.set()
    # Calculate final statistics
    final_time = time.time()
    total_time = final_time - start_time.value
    avg_time_per_request = total_time / global_count.value if global_count.value > 0 else 0
    print(f"\nFinal Statistics:")
    print(f"Final total successful requests: {global_count.value}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per request: {avg_time_per_request:.4f} seconds")