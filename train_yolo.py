import os
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 解决OMP报错: https://blog.csdn.net/m0_50736744/article/details/121799432
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # 创建缓存目录
    cache_dir = ''  # 地址
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    os.environ['ULTRALYTICS_CACHE_DIR'] = ''  # 地址
    # 加载模型
    model = YOLO('yolo11n.pt')

    # 1. 训练
    print("Start to Train.")
    model.train(data=r'',  # 数据集地址(yaml)
                # cache=False,
                imgsz=640,
                epochs=10,
                # single_cls=False,  # 是否是单类别检测
                batch=8,
                # close_mosaic=10,
                # workers=0,
                device='0',
                # optimizer='SGD',
                # amp=True,
                # project='runs/train',
                # name='exp',
                )
    print("Training End.")

    # 2. 预测
    print("Start to Predict.")
    model = YOLO('runs/detect/train/weights/best.pt')
    image_path = "datasets/traffic_sign/valid/images/"

    results = model.predict(source=image_path,
                            imgsz=640,
                            project='runs/detect',
                            name='exp',
                            save=True,
                            conf=0.2,
                            iou=0.7,
                            )
    print("Predict End.")
