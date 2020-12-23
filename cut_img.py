import cv2
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.datasets import LoadImages
from detect_config import Config
import torch

#设置cuda使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = Config
def cut_img(img_path):
    model = attempt_load(config.weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(config.imgsz, s=model.stride.max())  # check img_size
    #读取图片
    dataset = LoadImages(img_path, img_size=640)
    for path, img, im0s, _ in dataset:
        #path:图片根目录路径
        #img:(3, 640, 384)，图片按照img_size重置后的大小
        #im0s:(4032, 2268, 3)，图片原始大小
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0#正则化
        img = img.unsqueeze(0)#由于yolo的预测模型接收输入是按batchsize来的，而此函数只需要一个图片，因此扩张维度
        pred = model(img, augment=True)[0]
        pred = non_max_suppression(pred, config.conf_thres, config.iou_thres)

    #yolo5标记图片
    #载入模型
    #切割图片
    #返回图片

if __name__ == "__main__":
    cut_img('./test.jpg')