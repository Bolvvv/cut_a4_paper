import json
import os
def trans(src_dir_path, tgt_dir_path):
    """
    参数：
    src_dir_path:源标签的文件夹地址
    tgt_dir_path:转换标签的文件夹地址
    标签类型：使用labelme标注的json文件，主要为方框
    """
    json_file_list = os.listdir(src_dir_path)
    for file_path in json_file_list:
        #去除readme文件
        if file_path == 'README.md':
            continue
        with open(os.path.join(src_dir_path, file_path), 'r') as f:
            trans_result = normalise_to_yolo5(f)
            with open(os.path.join(tgt_dir_path, file_path.split('.', maxsplit=2)[0]+'.txt'), 'w') as r:
                for i in trans_result:
                    (class_name, x_center, y_center, w, h) = i
                    r.writelines(str(class_name)+' '+str(x_center)+' '+str(y_center)+' '+str(w)+' '+str(h)+'\n')


def normalise_to_yolo5(file):
    """
    将标注信息转换为yolo5需要的格式：<object-class> <x_center> <y_center> <width> <height>
    <object-class> 类名：从0到（类别数-1）
    <x_center> 中心点横坐标：这里的坐标并不是绝对坐标（真实坐标），而是相对于图片宽度的相对坐标，转换公式为<x_center> = <absolute_x> / <image_width>，这里的absolute_x是指标注框中心的横坐标
    <y_center> 中心点纵坐标：这里的坐标并不是绝对坐标（真实坐标），而是相对于图片高度的相对坐标，转换公式为<x_center> = <absolute_y> / <image_height>，这里的absolute_x是指标注框中心的纵坐标
    <width> 标注框宽度：这里的宽度并不是绝对宽度（真实宽度），而是相对于图片宽度的相对宽度，转换公式为<width> = <absolute_width> / <image_width>
    <height> 标注框宽度：这里的宽度并不是绝对宽度（真实宽度），而是相对于图片宽度的相对高度，转换公式为<height> = <absolute_height> / <image_height>
    """
    json_dict = json.load(file)
    result = []
    width = json_dict['imageWidth']
    height = json_dict['imageHeight']
    for temp in json_dict['shapes']:
        class_name = temp['label']
        #TODO:修改实现逻辑
        if class_name == 'img':
            class_name = 0
        else:
            class_name = 1
        x_left = temp['points'][0][0]
        y_left = temp['points'][0][1]
        x_right = temp['points'][1][0]
        y_right = temp['points'][1][1]

        #转换为相对位置
        x_center = (x_left + x_right) / (2 * width)
        y_center = (y_left + y_right) / (2 * height)
        w = (x_right - x_left) / width
        h = (y_right - y_left) /height
        one_result = (class_name, x_center, y_center, w, h)#yolo5的标签格式
        result.append(one_result)
    return result
if __name__ == "__main__":
    trans('./src_labels', './tgt_labels')
