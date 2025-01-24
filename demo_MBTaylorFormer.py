import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.archs.MB_TaylorFormer import MB_TaylorFormer


def pre_process(image: np.array, device: str, factor: int=8):
    """
    :param image: Input image to transform to the model input
    :param device: Device to send input to
    :returns: Tensor input to model, in the shape [b, c, h, w]
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2,0,1)
    image = image.unsqueeze(0).to(device)

    h, w = image.shape[2], image.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    image = F.pad(image, (0, padw, 0, padh), 'reflect')

    return image


def post_process(model_output: torch.Tensor, input_hw: Tuple[int, int]):
    """
    :param model_output: Output tensor produced by the model [b, c, h, w]
    :param input_hw: Tuple containing input image height and width
    :returns: Output image which can be displayed by OpenCV
    """
    h, w = input_hw
    model_output = model_output[:,:,:h,:w]

    image_rgb = model_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255 + 0.5).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr


network_b = dict(
    model_name = 16, # 'cat:0' or 'mp:1' or 'o:2' or 'se:3'or'rotaryem:7'or"SU_CRPE:13"
    inp_channels = 3,
    out_channels = 3,
    dim = [24,48,72,96],
    num_blocks = [2,3,3,4],
    num_refinement_blocks = 2,
    heads = [1,2,4,8],
    ffn_expansion_factor = 2.66,
    bias = False,
    LayerNorm_type = "WithBias",
    dual_pixel_task = False,
    num_path = [2,2,2,2],
    qk_norm = 0.5,
    offset_clamp = [-3,3],
)

network_l = dict(
    model_name = 16, # 'cat:0' or 'mp:1' or 'o:2' or 'se:3'or'rotaryem:7'or"SU_CRPE:13"
    inp_channels = 3,
    out_channels = 3,
    dim = [24,48,72,96],   #[24,48,72,96]  #[30,60,92,112]    # [36,72,100,136]   #
    num_blocks = [4,6,6,8],
    num_refinement_blocks = 4,
    heads = [1,2,4,8],
    ffn_expansion_factor = 2.66,
    bias = False,
    LayerNorm_type = "WithBias",
    dual_pixel_task = False,
    num_path = [2,3,3,3],
    qk_norm = 0.5,
    offset_clamp = [-3,3],
)

if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/29_hazy_video.mp4"
    device = "cpu"
    weights_path = "weights/OTS-MB-TaylorFormer-B.pth"
    network_size = "b"  # b or l

    model_configurations = {
        "b": network_b,
        "l": network_l,
    }.get(network_size, None)

    assert model_configurations is not None, f"Unable to get the model config for network size {network_size}"

    model = MB_TaylorFormer(**model_configurations)
    weights = torch.load(weights_path, map_location='cpu')["params"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)

    while True:
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)

        in_tensor = pre_process(frame, device)
        with torch.no_grad():
            model_outputs = model(in_tensor)
        out_image = post_process(model_outputs, frame.shape[:2])

        display_image = np.vstack([frame, out_image])

        cv2.imshow("output", display_image)
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
