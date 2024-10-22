import os
#os.system("wget https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx")
#os.system("pip install onnxruntime imageio")
import cv2
import paddlehub as hub
import gradio as gr
import torch
from PIL import Image, ImageOps
import numpy as np
import imageio
# os.mkdir("data")
# os.mkdir("dataout")
model = hub.Module(name='U2Net')
import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image
sess_options = onnxruntime.SessionOptions()
rmodel = onnxruntime.InferenceSession('lama_fp32.onnx', sess_options=sess_options)

# Source https://github.com/advimman/lama
def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


def predict(jpg, msk):

    imagex = Image.open(jpg)
    mask = Image.open(msk).convert("L")

    image, mask = prepare_img_and_mask(imagex.resize((512, 512)), mask.resize((512, 512)), 'cpu')
    # Run the model
    outputs = rmodel.run(None, {'image': image.numpy().astype(np.float32), 'mask': mask.numpy().astype(np.float32)})

    output = outputs[0][0]
    # Postprocess the outputs
    output = output.transpose(1, 2, 0)
    output = output.astype(np.uint8)
    output = Image.fromarray(output)
    output = output.resize(imagex.size)
    output.save("./dataout/data_mask.png")


def infer(img,option):
    imageio.imwrite("./data/data.png", img)
    if option == "automatic":
        result = model.Segmentation(
            images=[cv2.cvtColor(img, cv2.COLOR_RGB2BGR)],
            paths=None,
            batch_size=1,
            input_size=320,
            output_dir='output',
            visualization=True)
        im = Image.fromarray(result[0]['mask'])
        im.save("./data/data_mask.png")
    else:
        imageio.imwrite("./data/data_mask.png", img["mask"])
    predict("./data/data.png", "./data/data_mask.png")    
    return "./dataout/data_mask.png","./data/data_mask.png"

iface = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(label="Input Image", type="numpy"),
        gr.Radio(choices=["automatic", ], 
                 type="value", label="Masking Option")
    ],
    outputs=[
        gr.Image(type="filepath", label="Inpainted Image"),
        gr.Image(type="filepath", label="Generated Mask")
    ],
    title="LaMa Image Inpainting",
    description="Image inpainting with LaMa and U^2-Net. Upload your image and choose automatic.",
)

iface.launch()