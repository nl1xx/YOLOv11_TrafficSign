from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class YOLOv11App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 Detection")

        self.upload_button = tk.Button(root, text="上传图片", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.predict_button = tk.Button(root, text="开始预测", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", wraplength=400)
        self.result_label.pack(pady=10)

        self.image_path = None

        self.model = YOLO('runs/detect/train/weights/best.pt')

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if self.image_path:
            img = Image.open(self.image_path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def predict(self):
        if not self.image_path:
            messagebox.showwarning("警告", "请先上传图片！")
            return

        # 进行预测
        results = self.model(self.image_path)

        # 提取预测结果
        result_text = ""
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = self.model.names[class_id]
                result_text += f"类别: {class_id}, {class_name}, 置信度: {confidence:.2f}\n"

        # 显示预测结果
        self.result_label.config(text=result_text)

if __name__ == '__main__':
    root = tk.Tk()
    app = YOLOv11App(root)
    root.mainloop()