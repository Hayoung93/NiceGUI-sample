#!/usr/bin/env python3
import asyncio
import base64
import concurrent.futures
import signal
import time

import torch
import cv2
import os
import shutil
import numpy as np
from fastapi import Response
import PIL
from PIL import Image

import nicegui.globals
from nicegui import app, ui

import threading
import multiprocessing

from viz import viz

static_root = "/workspace/static"
app.add_static_files(static_root, static_root)
# We need an executor to schedule CPU-intensive tasks with `loop.run_in_executor()`.
process_pool_executor = concurrent.futures.ProcessPoolExecutor()
# In case you don't have a webcam, this will provide a black placeholder image.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')
# OpenCV is used to access the webcam.
TIME = 0
webcam_dir = "/workspace/static/webcam_images/"
webcam_res_dir = "/workspace/static/webcam_result/"
model_path = "/data/weights/deformable_detr/checkpoint_fcs.pth"
if __name__ == "__mp_main__":
    video_capture = cv2.VideoCapture(0)


def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


@app.get('/video/frame')
# Thanks to FastAPI's `app.get`` it is easy to create a web route which always provides the latest image from OpenCV.
async def grab_video_frame() -> Response:
    if not video_capture.isOpened():
        return placeholder
    loop = asyncio.get_running_loop()
    # The `video_capture.read` call is a blocking function.
    # So we run it in a separate thread (default executor) to avoid blocking the event loop.
    _, frame = await loop.run_in_executor(None, video_capture.read)
    if frame is None:
        return placeholder
    # `convert` is a CPU-intensive function, so we run it in a separate process to avoid blocking the event loop and GIL.
    jpeg = await loop.run_in_executor(process_pool_executor, convert, frame)
    return Response(content=jpeg, media_type='image/jpeg')

with ui.element('div').classes('justify-center p-2 bg-blue-100 w-full'):
    label_inputDir = ui.label().classes("invisible")
    ui.input(label="Input path of model to be used for inference (default: {})."\
        .format(model_path), on_change=lambda e: label_inputDir.set_text(e.value)).classes("w-full text-sm")
# For non-flickering image updates an interactive image is much better than `ui.image()`.
with ui.element('div').classes('justify-center p-2 bg-blue-100 w-full'):
    with ui.row().classes("justify-center no-wrap"):
        with ui.element('div').classes('justify-center w-full'):
            ui.label("Webcam")
            video_image = ui.interactive_image().classes('w-full h-full')
        with ui.element('div').classes('justify-center w-full'):
            ui.label("Processed")
            proc_image = ui.interactive_image().classes('w-full h-full')
# A timer constantly updates the source of the image.
# Because data from same paths are cached by the browser,
# we must force an update by adding the current timestamp to the source.
def frame_path():
    global TIME
    TIME = time.time()
    # loop = asyncio.new_event_loop()
    # loop.run_in_executor(executor_show, video_image.set_source, f'/video/frame?{TIME}')
    video_image.set_source(f'/video/frame?{TIME}')
    res_files = sorted(os.listdir(webcam_res_dir))
    if len(res_files) > 0:
        proc_image.set_source(os.path.join(webcam_res_dir, res_files[-1]))
ui.timer(interval=0.1, callback=lambda: frame_path())
ui.timer(interval=5, callback=lambda: run_model())

executor_show = concurrent.futures.ProcessPoolExecutor()
executor_move = concurrent.futures.ProcessPoolExecutor()
def run_model():
    fname = f'{TIME}.jpg'
    ui.download(f'/video/frame?{TIME}', fname)
    # proc_image.source = f'/video/frame?{TIME}'
    # Currently NiceGUI does not support passing local path.
    # So moving file is required.
    loop = asyncio.get_event_loop()
    # loop = asyncio.new_event_loop()
    future = loop.run_in_executor(None, move_file_and_run, fname)
    # future.add_done_callback(lambda x: executor_move.shutdown())
    # asyncio.gather(future)

    # loop_show = asyncio.new_event_loop()
    # loop_show.run_in_executor(executor_show, show_result, os.path.join(webcam_res_dir, "res_" + fname.replace(".png", ".jpg")))

    # task_move = asyncio.create_task(move_file_and_run(fname))
    # await task_move
    # res_img = task_move.result()
    # future = loop.run_until_complete(move_file_and_run(fname))
    # res_img = future.result()
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(show_result, os.path.join(webcam_res_dir, "res_" + fname))
#         return_value = future.result()
# def show_result(res_fp):
#     start_time_show = time.time()
#     while True:
#         if time.time() - start_time_show > 10:
#             print("break {}".format(res_fp))
#             break
#         if os.path.exists(res_fp):
#             proc_image.source = res_fp
#             print(res_fp, proc_image.source, proc_image)
#             break
#         else:
#             continue
# res_img = ""
# thread_res = threading.Thread(target=show_result, args=(res_img, ))
# thread_res.start()
# thread_res.join()
def move_file_and_run(fname):
    tgt_path = os.path.join(webcam_dir, fname)
    start_time_move = time.time()
    while True:
        if time.time() - start_time_move > 5:
            break
        try:
            shutil.move(f"/data/downloads/{fname}", tgt_path)
            start_time_viz = time.time()
            while True:
                if time.time() - start_time_viz > 2:
                    break
                try:
                    Image.open(tgt_path)
                    res_img = viz([label_inputDir.text if label_inputDir.text != "" else model_path], ["webcam_result"], tgt_path, static_root)[0]
                    torch.cuda.empty_cache()
                    return res_img
                except PIL.UnidentifiedImageError:
                    continue
            break
        except FileNotFoundError:
            continue


async def disconnect() -> None:
    """Disconnect all clients from current running server."""
    for client in nicegui.globals.clients.keys():
        await app.sio.disconnect(client)


def handle_sigint(signum, frame) -> None:
    # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
    ui.timer(0.1, disconnect, once=True)
    # Delay the default handler to allow the disconnect to complete.
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)


async def cleanup() -> None:
    # This prevents ugly stack traces when auto-reloading on code change,
    # because otherwise disconnected clients try to reconnect to the newly started server.
    await disconnect()
    # Release the webcam hardware so it can be used by other applications again.
    video_capture.release()
    # The process pool executor must be shutdown when the app is closed, otherwise the process will not exit.
    process_pool_executor.shutdown()
    executor_move.shutdown()
    executor_show.shutdown()

app.on_shutdown(cleanup)
# We also need to disconnect clients when the app is stopped with Ctrl+C,
# because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
signal.signal(signal.SIGINT, handle_sigint)

ui.run(host="0.0.0.0", port=8080)
