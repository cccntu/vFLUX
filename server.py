import asyncio
from collections import deque
import os
import torch
from diffusers import FluxPipeline
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import base64
from PIL import Image
import io

from contextlib import asynccontextmanager
from torch.profiler import profile, ProfilerActivity
import numpy as np
import cv2


# for profiling
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.enable_profiling = os.environ.get("ENABLE_PROFILING", "1") == "1"
    # Startup
    if app.state.enable_profiling:
        prof= profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        prof.start()
    else:
        prof=None
    yield
    # Shutdown
    if app.state.enable_profiling:
        prof.stop()
        path = "server-trace.json"
        prof.export_chrome_trace(path)

app = FastAPI(lifespan=lifespan)

# Load the model
device = torch.device('cuda')
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.to(device)

pipe.transformer = torch.compile(pipe.transformer, mode='max-autotune')
pipe.vae.decode = torch.compile(pipe.vae.decode, mode='max-autotune')
pipe.text_encoder = torch.compile(pipe.text_encoder, mode='max-autotune')
pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, mode='max-autotune')

# warmup
for i in range(2):
    images = pipe('warmup', num_inference_steps=4, output_type="pt").images
    images = (images * 255).round().clamp(0, 255).to(torch.uint8)


class ImageRequest(BaseModel):
    text: str

def post_process(image):
    """Convert the generated image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def pt_to_b64(images):
    # Convert from PyTorch tensor to NumPy array
    image_np = images.squeeze().permute(1, 2, 0).numpy()
    assert image_np.dtype == np.uint8, f"Expected dtype uint8, got {image_np.dtype}"
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    img_str = base64.b64encode(buffer).decode()
    return img_str

# a semaphore that ensures first come first serve, so no request will starve
class OrderedSemaphore:
    def __init__(self, value):
        self._value = value
        self._queue = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        # Create a future for this acquisition request
        future = asyncio.Future()
        
        async with self._lock:
            if self._value > 0:
                # If semaphore is available, acquire immediately
                self._value -= 1
                future.set_result(None)
            else:
                # Otherwise, add to queue
                self._queue.append(future)
        
        # Wait until we get the semaphore
        await future
    
    async def release(self):
        async with self._lock:
            if self._queue:
                # If there are waiters, give it to next in line
                next_future = self._queue.popleft()
                next_future.set_result(None)
            else:
                # Otherwise, increment the value
                self._value += 1
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


ordered_semaphore = OrderedSemaphore(2)

@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    text = request.text
    async with ordered_semaphore:
        images = pipe(text, num_inference_steps=4, output_type="pt").images
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        images = images.to('cpu', non_blocking=True)
        output_ready_event = torch.cuda.Event()
        output_ready_event.record(torch.cuda.current_stream())
        image = images[0] # still in pt format
        while not output_ready_event.query():
            # yield control flow and allow another requst to schedule async GPu execution
            await asyncio.sleep(0.001)
    
        # Convert to base64
        img_str = pt_to_b64(image)
        return {"image": img_str}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)