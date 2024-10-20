import os
import sys
import torch
from pathlib import Path
import cv2
import numpy as np

# Add YOLOv5 root directory to PYTHONPATH
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (non_max_suppression, scale_coords, check_img_size,
                           set_logging, xyxy2xywh)
from utils.plots import plot_one_box
from utils.torch_utils import select_device

def main():
    # Initialize
    set_logging()
    device = select_device('')  # Select device ('' for CPU, '0' for GPU)
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    # Load model
    weights = 'weights/yolov5sSe.pt'  # Path to your trained model
    model = attempt_load(weights, map_location=device)  # Load model
    stride = int(model.stride.max())  # Model stride
    imgsz = 640  # Inference size (pixels)
    imgsz = check_img_size(imgsz, s=stride)  # Check image size

    if half:
        model.half()  # Convert to half precision

    # Set Dataloader
    source = 'data/images'  # Directory with images
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    names = model.module.names if hasattr(model, 'module') else model.names  # Get class names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]  # Random colors

    model.eval()
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # Convert to float16/32
        img /= 255.0  # Normalize to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # Add batch dimension

        # Inference
        with torch.no_grad():
            pred = model(img)[0]

        # Apply NMS
        conf_thres = 0.25  # Confidence threshold
        iou_thres = 0.45   # IoU threshold for NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, im0 = path, im0s.copy()
            p = Path(p)  # Path object
            save_path = str(Path('results') / p.name)  # Save path

            s += '%gx%g ' % img.shape[2:]  # Print image size

            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # Number of detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Save results
            os.makedirs('results', exist_ok=True)
            cv2.imwrite(save_path, im0)

            print(f"{s}Done. Results saved to {save_path}")

if __name__ == '__main__':
    main()
