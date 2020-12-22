from PIL import Image
import matplotlib.pyplot as plt
import qrcode

def save_img_step(img_root_path, lines):
    plt.figure(figsize=(14.15, 20)) #设置窗口大小
    a4_name = lines[0].rstrip().split(',')[1]
    plt.suptitle(a4_name, fontsize=35)
    for i in range(len(lines)):
        content = lines[i].rstrip().split(',')
        img_path = content[2]

        img = Image.open(img_root_path+"/"+img_path)
        plt.subplot(4,2,i+1)#i是从0开始
        plt.imshow(img), plt.axis('off')
    a4_name_qr = qrcode.make(a4_name)
    a4_name_qr.save("tmp_qr.jpg")#由于存储的图片是反色的，因此先将其存储于本地再重新打开即可恢复
    tmp_qr = Image.open("tmp_qr.jpg")
    plt.subplot(4,2,7)#i是从0开始
    plt.imshow(tmp_qr), plt.axis('off')
    plt.savefig("./print_img/"+a4_name+".jpg")
    #关闭当前figure
    fig = plt.gcf()
    plt.close(fig)

def save_img(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 6):
            temp_lines = []
            for j in range(0, 6, 1):
                if i+j >=len(lines):
                    break
                temp_lines.append(lines[i+j])
            save_img_step("../usg_images_cutted_v3", temp_lines)

if __name__ == "__main__":
    save_img("data.csv")