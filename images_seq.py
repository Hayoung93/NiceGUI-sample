import os
import io
from PIL import Image

from nicegui import ui, app

from viz import viz

# app settings
static_root = "/workspace/static"
app.add_static_files(static_root, static_root)

# variables
imgs_root = os.path.join(static_root, "fcs_berlin_samples")
imgs_names = sorted(os.listdir(imgs_root))
img_fp = os.path.join(imgs_root, imgs_names[0])
img_fp_s = img_fp
img_fp_d = img_fp
model_source = "/data/weights/deformable_detr/checkpoint_fcs_source.pth"
model_domain = "/data/weights/deformable_detr/checkpoint_fcs.pth"


img_idx = 0

def show_next_image():
    global img_idx
    global img_fp
    img_idx = (img_idx + 1) % len(imgs_names)
    img_fp = os.path.join(imgs_root, imgs_names[img_idx])
    holder_before.source = img_fp
    holder_after.source = img_fp
    os.remove(img_fp_s)
    os.remove(img_fp_d)

def handle_upload(e):
    ui.notify(f'Uploaded {e.name}')
    uploaded_img = io.BytesIO(e.content.read())
    uploaded_img = Image.open(uploaded_img)
    global img_fp
    img_fp = os.path.join(static_root, e.name)
    uploaded_img.save(img_fp)
    holder_before.source = img_fp
    holder_after.source = img_fp
    label_upload.set_text("Uploaded " + img_fp)

def perform_detection():
    global img_fp
    global img_fp_s
    global img_fp_d
    # print(img_fp_s, img_fp_d)
    img_fp_s, img_fp_d = viz(model_source, model_domain, img_fp, static_root)
    holder_before.source = img_fp_s
    holder_after.source = img_fp_d

# pop-up card for uploading an image file
with ui.dialog() as dialog, ui.card():
    holder_upload = ui.upload(max_file_size=100000000, max_total_size=100000000, max_files=1, auto_upload=True,
                on_upload=lambda e: handle_upload(e)).classes('max-w-full')
    holder_upload.props('accept=".png,.jpeg,.jpg,.PNG,.JPEG,.JPG"')
    ui.button('Close', on_click=dialog.close)

# paths for inference models, inputs, and other settings
with ui.element('div').classes('justify-center p-2 bg-blue-100 w-full'):
    label_inputDir = ui.label().classes("invisible")
    ui.input(label="1. Input path of image directory (default: {}).".format(imgs_root), on_change=lambda e: label_inputDir.set_text(e.value)).classes("w-full text-sm")

    label_inputModelS = ui.label().classes("invisible")
    ui.input(label="2. Input DOMAIN model path (default: {}).".format(model_source), on_change=lambda e: label_inputModelS.set_text(e.value)).classes("w-full text-sm")

    label_inputModelD = ui.label().classes("invisible")
    ui.input(label="3. Input SOURCE model path (default: {}).".format(model_domain), on_change=lambda e: label_inputModelD.set_text(e.value)).classes("w-full text-sm")

    with ui.row().classes("no-wrap align-middle pt-3"):
        ui.label("You can also upload a single image file manually :").classes("mt-2")
        button_uploadDiag = ui.button('UPLOAD', on_click=dialog.open)
        label_upload = ui.label("").classes("mt-2")

# button for performing object detection
ui.button('detect!', on_click=lambda _: perform_detection()).classes("w-full")

# displaying area for results
with ui.element('div').classes('justify-center p-2 bg-blue-100 w-full'):
    with ui.row().classes("justify-center no-wrap"):
        with ui.element('div').classes('justify-center w-full'):
            ui.label("Before")
            holder_before = ui.image(img_fp_s).props("width=100%")
        with ui.element('div').classes('justify-center w-full'):
            ui.label("After")
            holder_after = ui.image(img_fp_d).props("width=100%")
    with ui.row().classes("justify-center pt-2"):
        button_next = ui.button('Next Image', on_click=show_next_image).props("size=14px")


ui.run(host="0.0.0.0", port=8081)
