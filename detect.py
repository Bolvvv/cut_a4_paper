import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math
from PIL import Image, ExifTags
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from detect_config import Config

config =Config#生成参数对象

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

def cut_img(p_name, det_info_list, width, height):
    #判断是否符合长度(应该为7个框，6个图片，1个二维码)
    if len(det_info_list) != 7:
        log_result(p_name+'未完整框选出所有图片，总框选数量为'+str(len(det_info_list)))
        return None
    print(det_info_list)
    sorted_list = sort_img(det_info_list)
    if sorted_list == None:
        log_result(p_name+'未找到qr码，总框选数量为'+str(len(det_info_list)))
        return None
    cut_img_step(p_name, sorted_list, width, height)
def cut_img_step(p_name, sorted_list, width, height):

    #打开图片
    img = cv2.imread(os.path.join(config.source, p_name))
    for i in range(len(sorted_list)):
        (x_l, y_l, x_r, y_r) = trans_yolo_to_normal(width, height, sorted_list[i][1])
        cropped = img[y_l:y_r, x_l:x_r]
        cv2.imwrite(os.path.join(config.save_cut_img_path, str(i)+'.jpg'), cropped)
    

def sort_img(l):
    img_list = []
    src_len = len(l)
    #找寻qr码
    for i in range(len(l)):
        if l[i][0] == 1:
            img_list.append(l[i])
            l.pop(i)
            break
    if len(img_list) == 0:
        #如果未找到qr码，则返回None
        return None
    l, img_list, aim_index = sort_step(l, img_list, 0, src_len)
    # img_list = img_list[::-1]
    return img_list   
 
def sort_step(l, img_list, aim_index, src_len):
    if aim_index == src_len -1:
        return l, img_list, aim_index
    min_len = [2, -1]#最近距离和编号
    min_2_len = [2, -1]#次近距离和编号
    for m in range(len(l)):
        if l[m] != None:
            length_to_aim = math.pow(l[m][1][0] - img_list[aim_index][1][0], 2) + math.pow(l[m][1][1] - img_list[aim_index][1][1], 2)
            if length_to_aim < min_len[0]:
                min_2_len = min_len.copy()#进行浅拷贝
                min_len = [length_to_aim, m]
            elif length_to_aim < min_2_len[0]:
                min_2_len = [length_to_aim, m]
            else:
                continue
    img_list.append(l[min_2_len[1]])
    img_list.append(l[min_len[1]])
    l[min_2_len[1]] = None
    l[min_len[1]] = None
    aim_index = len(img_list)-1
    return sort_step(l, img_list, aim_index, src_len)

def log_result(str_log):
    with open(config.result_log, 'a') as f:
        f.writelines(str_log+'\n')

def detect():
    source, weights, save_txt, imgsz, save_img = config.source, config.weights, config.save_txt, config.img_size, config.save_img

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
                cut_img(p.name, det_info_list, im0.shape[1], im0.shape[0])
            else:
                #此时表明没有框选出图片，需要进行记录：
                with open('result.txt', 'a') as f:
                    f.writelines(p.name + '不存在框选目标')
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    with torch.no_grad():
        detect()
