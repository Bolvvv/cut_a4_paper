import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math
import os
import cv2
from pyzbar.pyzbar import decode

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from detect_config import Config

config =Config#生成参数对象
result_log = open(os.path.join(config.result_log, time.strftime('%m%d_%H%M_%S',time.localtime(time.time()))+'_log.txt'), 'a')#日志记录
def log_result(str_log):
    """
    记录日志
    """
    result_log.writelines(str_log+'\n')

def cut_img(p_name, det_info_list, width, height, save_dir, label_list):
    #判断是否符合长度(应该为7个框，6个图片，1个二维码)
    if len(det_info_list) != 7:
        log_result(p_name+'框选数量有误，总框选数量为'+str(len(det_info_list))+' 框选结果存入:'+str(save_dir))
        return None
    sorted_list = sort_img(det_info_list)
    if sorted_list == None:
        log_result(p_name+'未找到qr码，总框选数量为'+str(len(det_info_list))+' 框选结果存入:'+save_dir)
        return None
    cut_img_step(p_name, sorted_list, width, height, label_list)

def trans_yolo_to_normal(width, height, xywh):
    """
    将yolo格式的方框转换为像素值
    """
    x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
    x_l = (2*width*x - w*width)/2
    x_r = (2*width*x + w*width)/2
    y_l = (2*height*y - h*height)/2
    y_r = (2*height*y + h*height)/2
    return (int(x_l), int(y_l), int(x_r), int(y_r))

def cut_img_step(p_name, sorted_list, width, height, label_list):
    #打开图片
    img = cv2.imread(os.path.join(config.source, p_name))
    #获取二维码结果
    cropped_list = []
    for i in range(len(sorted_list)):
        (x_l, y_l, x_r, y_r) = trans_yolo_to_normal(width, height, sorted_list[i][1])
        cropped = img[y_l:y_r, x_l:x_r]
        cropped_list.append(cropped)
    number = decode_qr(cropped_list[len(cropped_list)-1])
    if number == None:
        log_result(p_name+' 二维码解析器无法识别图片二维码或识别到多个二维码，未完成分割，未储存分割图片')
        return None
    temp_label_list = []
    for i in label_list:
        if number == i[1]:
            temp_label_list.append(i)
    if len(cropped_list)-1 != len(temp_label_list):
        #由于在cropped_list中添加了二维码，因此在比较时需要去除
        log_result(p_name+" 完成切割，但无法与label中的标签在数量上对应，未储存分割图片")
        return None
    for i in range(len(cropped_list)-1):
        cv2.imwrite(os.path.join(config.save_cut_img_path, temp_label_list[i][0], temp_label_list[i][2]), cropped_list[i])
        
def sort_img(l):
    """
    该排序算法只适用于图片排布为从左到右，从上到下，且二维码相对位置为左。拍摄时的A4纸上下颠倒或者位置变动均不会对排序结果造成影响
    """
    img_list = []
    #找寻qr码
    for i in range(len(l)):
        if l[i][0] == 1:
            img_list.append(l[i])
            l.pop(i)
            break
    if len(img_list) == 0:
        #如果未找到qr码，则返回None
        return None
    l, img_list, aim_index = sort_step(l, img_list, 0)
    fianl_list = sort_mapping(img_list)
    return fianl_list   
 
def sort_step(l, img_list, aim_index):
    if aim_index == len(l):
        return l, img_list, aim_index
    min_len = [2, -1]#最近距离和编号
    for m in range(len(l)):
        if l[m] != None:
            length_to_aim = math.pow(l[m][1][0] - img_list[aim_index][1][0], 2) + math.pow(l[m][1][1] - img_list[aim_index][1][1], 2)
            if length_to_aim < min_len[0]:
                min_len = [length_to_aim, m]
                continue
    img_list.append(l[min_len[1]])
    l[min_len[1]] = None
    aim_index = len(img_list)-1
    return sort_step(l, img_list, aim_index)

def sort_mapping(img_list):
    img_list_len = len(img_list)
    sorted_list = [None]*img_list_len
    if img_list_len%2 == 0:
        sorted_list[img_list_len-1] = img_list[0]#先讲二维码所属位置赋值
        img_list.pop(0)#再将二维码的值去掉
        img_list_len -= 1#将长度减一
    left_len = math.ceil(img_list_len/2)
    right_len = img_list_len-left_len
    for i in range(0, 2*left_len, 2):
        sorted_list[img_list_len-i-1] = img_list[int(i/2)]
    for i in range(2, 2*right_len+2, 2):
        sorted_list[i-1] = img_list[int(i/2)+left_len-1]
    return sorted_list

def rename_file():
    """
    由于统计的数据每个人都是不一样的，可能存在重名的照片，因此需要将数据重新命名
    """
    file_list = os.listdir(config.source)
    for i in range(len(file_list)):
        old_name = os.path.join(config.source, file_list[i])
        new_name = os.path.join(config.source, str(i)+time.strftime('_%H%M_%S',time.localtime(time.time()))+'.jpg')
        os.rename(old_name, new_name)

def decode_qr(img):
    _,trans_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)#对图片进行二值化
    result = decode_qr_step(trans_img)
    if result != None:
        return result
    else:
        flag = True
        for i in range(20, 240, 20):
            _,trans_img = cv2.threshold(img,i,255,cv2.THRESH_BINARY)#对图片进行二值化
            result = decode_qr_step(trans_img)
            if result != None:
                flag = False
                return result
        if flag:
            return None
def decode_qr_step(img):
    qr_data = decode(img)#对二维码进行解码
    if len(qr_data) == 1:
        number =qr_data[0].data.decode('UTF-8')#将解析出的二进制二维码结果转换为utf-8编码
        return number
    else:
        return None

def detect():
    source, weights, save_txt, imgsz, save_img = config.source, config.weights, config.save_txt, config.img_size, config.save_img

    #对图片进行重命名
    # rename_file()

    # Directories
    save_dir = Path(increment_path(Path(config.project) / config.name, exist_ok=config.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(config.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    #载入标签文件，用于将切割好的图片进行归档
    label_list = []
    with open(config.label_list_path, 'r')  as f:
        temp_list = f.readlines()
        for i in temp_list:
            temp = i.strip().split(',',maxsplit=3)
            label_list.append(temp)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, config.conf_thres, config.iou_thres)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)#im0是原图[长，宽，通道]
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                #存储经转换后的标记框信息
                det_info_list = []
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = [int(cls), xywh, float(conf.cpu())] if config.save_conf else [int(cls), xywh]  # label format
                    det_info_list.append(line)
                    if save_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                #处理图片信息
                cut_img(p.name, det_info_list, im0.shape[1], im0.shape[0], save_dir, label_list)
            else:
                #此时表明没有框选出图片，需要进行记录：
                log_result(p.name + '不存在框选目标')
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 存储识别结果
            if save_img:
                cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    with torch.no_grad():
        #创建日志文件
        detect()
