from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch
import cv2
from PIL import Image
import numpy as np
import io
import os

# 初始化 FastAPI 应用
app = FastAPI()

# 加载 YOLOv5 模型 (确保你在 yolov5 文件夹下运行代码，或者正确指定权重文件路径)
personmodel = torch.hub.load('D:/project/yolov5', 'custom', path='../029行人检测模型/output/weights/best.pt',
                             source='local')  # 使用训练好的权重文件


# 定义接收图片并返回检测结果的 API
@app.post("/predict/person")
async def predict(file: UploadFile = File(...)):
    try:
        # 将上传的文件读取为字节流
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # 将 PIL 图片转换为 OpenCV 格式
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 使用 YOLOv5 进行推理
        results = personmodel(image_cv2)

        # 解析检测结果
        parsed_detections = []
        label_counts = {"person": 0}

        for detection in results.xyxy[0]:  # 遍历每个检测结果
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            class_name = results.names[int(class_id)]  # 获取类别名称

            # 只关注 "person" 类别
            if class_name == "person":
                parsed_detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence),
                    "class": class_name
                })
                # 增加 "person" 类别计数
                label_counts["person"] += 1

                # 在图像上绘制边界框和标签
                cv2.rectangle(image_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image_cv2, f'{class_name} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 将标注后的图片保存到文件
        output_path = "person.jpg"
        cv2.imwrite(output_path, image_cv2)

        # 返回检测结果和标签数量，并提供图片下载链接
        return {
            "detections": parsed_detections,
            "label_counts": label_counts,  # 返回标签和对应的数量
            "annotated_image_url": f"http://192.168.10.4:8510/download/{output_path}"  # 返回标注图像的下载 URL
        }

    except Exception as e:
        # 捕获并记录异常
        return {"error": str(e)}


firemodel = torch.hub.load('D:/project/yolov5', 'custom', path='yolov5s_best.pt', source='local')
print(firemodel.names)


@app.post("/predict/fire")
async def predict(file: UploadFile = File(...)):
    try:
        # 将上传的文件读取为字节流
        image_data = await file.read()
        # 使用 PIL.Image 打开图片
        image = Image.open(io.BytesIO(image_data))
        # 将 PIL 图片转换为 numpy 数组（保持 RGB 格式）
        image_np = np.array(image)
        # 使用 YOLOv5 进行推理
        results = firemodel(image_np)
        # 解析检测结果
        parsed_detections = []
        label_counts = {"fire": 0}  # 修正此处
        for detection in results.xyxy[0]:  # 遍历每个检测结果
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            class_name = results.names[int(class_id)]  # 获取类别名称
            # 只关注 "fire" 类别
            if class_name == "fire":
                parsed_detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence),
                    "class": class_name
                })
                # 增加 "fire" 类别计数
                label_counts["fire"] += 1

        # 在绘制之前，将图像转换为 BGR 格式
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # 在图像上绘制边界框和标签
        for detection in parsed_detections:
            x1, y1, x2, y2 = detection["box"]
            confidence = detection["confidence"]
            class_name = detection["class"]
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv2, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
        # 将标注后的图片保存到文件
        output_path = "fire_image.jpg"
        if label_counts["fire"] >= 1:
            cv2.imwrite(output_path, image_cv2)

        # 返回检测结果和图片下载链接
        return {
            "detections": parsed_detections,
            "label_counts": label_counts,
            "annotated_image_url": f"http://192.168.10.4:8510/download/{output_path}"
        }

    except Exception as e:
        # 捕获并记录异常
        return {"error": str(e)}


# 定义图片下载端点，返回标注后的图片
@app.get("/download/{filename}")
async def download_image(filename: str):
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}


# 运行 FastAPI 应用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8510)
