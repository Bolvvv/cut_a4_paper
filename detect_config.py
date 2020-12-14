class Config(object):
    weights = './weights/best.pt'#训练好的模型参数
    source = './img'#需要预测的图片存储地点
    img_size = 1280#图片尺寸
    save_img = True#是否要存储预测的图片
    conf_thres = 0.25#置信度低于0.25则不将目标勾画出来
    iou_thres = 0.45#交并比，参考:https://zhuanlan.zhihu.com/p/63180762
    device = '0'#cuda设备
    save_txt = False#存储结果
    save_conf = False#将置信度存储于txt文本里
    project = 'runs/detect'#所有结果文件夹目录
    name = 'exp'#一次预测的结果文件夹
    exist_ok = False#判断结果文件夹