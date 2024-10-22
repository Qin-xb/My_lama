import gradio as gr
from PIL import Image
import numpy as np

def extract_brush_strokes(image_dict):
    # 提取用户涂抹的图层
    layers = image_dict['layers'][0]  # 假设只有一个图层，取第一个
    
    # 获取图层中的透明度（Alpha 通道），即第4个通道
    alpha_channel = layers[:, :, 3]  # 获取图层的Alpha通道
    
    # 创建一个蒙版，标记非透明的区域（即用户涂抹的区域）
    mask = alpha_channel > 0  # True表示用户涂抹的区域
    
    # 将蒙版转换为图像形式
    mask_img = Image.fromarray(np.uint8(mask) * 255, mode="L")  # 转换为灰度图像
    
    return mask_img  # 返回用户涂抹的区域

# 使用 ImageEditor 组件
interface = gr.Interface(
    fn=extract_brush_strokes,
    inputs=gr.ImageEditor(type="numpy"),  # 使用 ImageEditor 作为输入
    outputs=gr.Image()
)

interface.launch()
