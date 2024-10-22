import os
os.system("wget https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx")
os.system("pip install onnxruntime imageio")
import os
import cv2
import gradio as gr
import torch
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import imageio
import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image

import requests

def box(x1, y1, x2, y2, path):

    data = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "path": path,
    }
    url = "http://192.168.41.106:8005/box"

    res = requests.post(url, json=data)
    
    with open('./data/data_mask_by_box.jpg', "wb") as f:
        f.write(res.content)
    return res.content

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
    output.save("./dataout/data_mask.jpg")


def extract_brush_strokes(image_dict):
    # 提取用户涂抹的图层
    layers = image_dict['layers'][0]  # 假设只有一个图层，取第一个

    alpha_channel = layers[:, :, 3]  # 获取图层的Alpha通道
    # 创建一个mask模板，标记非透明的区域（即用户涂抹的区域）
    mask = alpha_channel > 0  # True表示用户涂抹的区域
    # 将蒙版转换为图像形式
    mask_img = Image.fromarray(np.uint8(mask) * 255, mode="L")  # 转换为灰度图像
    mask_img.save('./data/data_mask.jpg')

def get_bounding_box(image: np.ndarray):
    """
    获取涂抹区域的最小边界框(x_min, y_min, x_max, y_max)，分别表示边界框的左上角和右下角的坐标。
    """
    # 假设涂抹区域是非零值，可以是二值图像或其他格式
    # 对于RGBA图像，可以取alpha通道来判断哪些区域被涂抹
    if image.shape[-1] == 4:  # 如果是RGBA图像
        smudge_mask = image[..., 3] > 0  # 获取Alpha通道非透明区域
    else:
        smudge_mask = image > 0  # 假设其他通道非零值表示涂抹

    # 获取所有涂抹区域的坐标点
    coords = np.argwhere(smudge_mask)

    if coords.shape[0] == 0:
        raise ValueError("没有找到涂抹区域")

    # 分别计算最小和最大坐标
    y_min, x_min = coords.min(axis=0)  # 找到最小的x, y
    y_max, x_max = coords.max(axis=0)  # 找到最大的x, y

    return int(x_min), int(y_min), int(x_max), int(y_max)


def infer(img_dict, channel):
    extract_brush_strokes(img_dict)
    img = img_dict['background']
    img_box = img_dict['layers'][0]
    x1,y1,x2,y2 = get_bounding_box(img_box)
    image_rgb = Image.fromarray(img).convert("RGB")
    imageio.imwrite("./data/data.jpg", np.array(image_rgb))

    # 使用矩形框
    box(x1,y1,x2,y2,'/data2/qinxb/LaMa-Demo-ONNX/data/data.jpg')

    predict("./data/data.jpg", "./data/data_mask.jpg")   
     
    return "./dataout/data_mask.jpg","./data/data_mask.jpg","./data/data_mask_by_box.jpg"


if __name__=="__main__":

    with gr.Blocks() as iface:
        gr.Markdown("# LaMa Image Inpainting")

        with gr.Row():
            #input_image = gr.Image(label="Input Image", type="numpy")
            input_image = gr.ImageEditor(type="numpy")
            with gr.Column():
                channel = gr.Slider(minimum=0, maximum=3, value=0, step=1,label="Select a mask channel")
                btn_infer = gr.Button("Run")
        with gr.Row():
            output_inpainted = gr.Image(type="filepath", label="Inpainted Image")
            output_mask = gr.Image(type="filepath", label="Generated Mask")
        # with gr.Row():
            output_mask_box = gr.Image(type="filepath", label="Mask Image")
            # output_maskout = gr.Image(type="filepath", label="Maskout Image")
        # input_image.select(get_xy)
        btn_infer.click(fn=infer, inputs=[input_image, channel], outputs=[output_inpainted, output_mask,output_mask_box])

    iface.launch(server_name="0.0.0.0", server_port=8006)

iface.launch(server_name="0.0.0.0", server_port=8006)
