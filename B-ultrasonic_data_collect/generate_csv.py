import os
def generate_list(label_file, image_file):
    image_label_list = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.rstrip().split()
            name = content[0]
            label = content[1]
            if os.path.exists(image_file+name):
                image_label_list.append(name)
            else:
                print("not find:"+name)
    return image_label_list

def write_file(file_list, file_name, dataset_name):
    """
    将read_file文件生成的数据写入到新的文件中
    参数说明：
    file_list:存储label文件的list，形如：[(str_img_name, str_label), (str_img_name, str_label)]
    file_name:形如：待写入label的文件名称，label.txt
    """
    with open(file_name, 'a') as f:
        for i in file_list:
            img_name = i
            f.writelines(dataset_name+" "+img_name+"\n")

def generate_csv(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        iter = 0
        for i in range(0, len(lines), 6):
            for j in range(0, 6, 1):
                if i+j >=len(lines):
                    break
                content = lines[i+j].rstrip().split()
                data_set_name = content[0]
                a4_number = "p"+str(iter)
                img_name = content[1]
                with open("data.csv", "a") as m:
                    m.write('%s,%s,%s\n' % (data_set_name, a4_number, img_name))
            iter+=1

if __name__ == "__main__":
    generate_csv("collect_data.txt")