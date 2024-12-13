import subprocess
import os
import time

# For easier to start the website demo
WEBSITE_DEMO_PATH = "./Website_Demo"

def start_celery():
    celery_command = ["celery", "-A", "backend.celery", "worker", "--loglevel=info", "--concurrency=1"]
    return subprocess.Popen(celery_command, cwd=WEBSITE_DEMO_PATH)

def start_backend():
    backend_command = ["python3", "backend.py"]
    return subprocess.Popen(backend_command, cwd=WEBSITE_DEMO_PATH)

if __name__ == "__main__":
    print("Starting services...")

    try:
        celery_process = start_celery()
        time.sleep(2)
        backend_process = start_backend()
        print("Services are running. Press Ctrl+C to stop.")
        celery_process.wait()
        backend_process.wait()

    except KeyboardInterrupt:
        print("Shutting down services...")
        celery_process.terminate()
        backend_process.terminate()
        celery_process.wait()
        backend_process.wait()
        print("Services stopped.")
