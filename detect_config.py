class Config(object):
    weights = './weights/best.pt'#训练好的模型参数
    source = './img'#需要预测的图片存储地点
    save_cut_img_path = './save_cut_img'
    result_log = './result.txt'#切割结果日志路径
    project = 'runs/detect'#所有结果文件夹目录
    name = 'exp'#一次预测的结果文件夹
    label_list_path = '../data/deepbc/data.csv'#图片和数据集对应的表格存放地址
    img_size = 1280#图片尺寸
    save_img = True#是否要存储预测的图片
    conf_thres = 0.5#置信度低于0.25则不将目标勾画出来
    iou_thres = 0.4#交并比，参考:https://zhuanlan.zhihu.com/p/63180762
    device = '0'#cuda设备
    save_txt = True#存储结果
    save_conf = True#将置信度存储于txt文本里
    exist_ok = False#判断结果文件夹是否存在，为False的话则会覆盖创建exp文件夹