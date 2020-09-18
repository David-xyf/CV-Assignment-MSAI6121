import PIL.Image as Image
import os

IMAGES_PATH = r'F:/CV/assignments/Assignments1/images/original images/'
IMAGES_FORMAT = ['.jpg', '.jpeg']
IMAGE_SIZE = 256
IMAGE_ROW = 3
IMAGE_COLUMN = 8
IMAGES_SAVE_PATH = '../comparision.jpg'

# get all iamges' names from the current dir
image_names = [name for name in os.listdir(
    IMAGES_PATH) for item in IMAGES_FORMAT if os.path.splitext(name)[1] == item]
image_names=image_names[16:24]+image_names[8:16]+image_names[:8]
print(image_names)
# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")


def images_compose():
    # define the function of images' compositions
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE,
                                 IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    for y in range(1, IMAGE_ROW+1):
        for x in range(1, IMAGE_COLUMN+1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (
                y - 1) + x - 1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(
                from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGES_SAVE_PATH)


images_compose()
