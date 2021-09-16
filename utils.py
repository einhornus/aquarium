import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorflow as tf
import json
from collections import defaultdict


def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    plt.show()


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


"""
Функция визуализации одного из обнаруженных объектов на изображения
Код позаимствован из https://www.tensorflow.org/hub/tutorials/object_detection
"""
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=thickness,
              fill=color)

    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


"""
Функция визуализации набора обнаруженных объектов на изображения
Код позаимствован из https://www.tensorflow.org/hub/tutorials/object_detection и немного модифицирован
"""
def draw_boxes(image, objects, min_score):
    font = ImageFont.load_default()

    for i in range(len(objects)):
        obj = objects[i]
        score = obj["score"]
        if score >= min_score:
            ymin, xmin, ymax, xmax = tuple(obj["bbox"])
            display_str = "{}: {}%".format(obj["class"], int(100 * obj["score"]))
            if obj['type'] == 'groundtruth':
                color = 'blue'
            if obj['type'] == 'prediction':
                color = 'red'
            if obj['type'] == 'match':
                color = 'green'

            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font,
                                       display_str_list=[display_str]), np.copyto(image, np.array(image_pil))
    return image


"""
Агрегируем данные из всех .json-файлов с аннотациями

Формат выходных данных: map, в котором:
ключ - конкатенация директории и номера изображения
значение - словарь, содержащий путь к файлу, размеры изображения и список объектов
Каждый объект - словарь, который содержит bounding box (список относительных координат в формате x1,y1,x2,y2), метку класса и confidence score (в данном случае, всегда 1.0)
"""
def collect_data(folders, CLASSES):
    res = defaultdict()

    for j in range(len(folders)):
        folder = folders[j]
        with open(folder + '//_annotations.coco.json', mode='r') as f:
            data = json.loads(f.read())

            images_list = data["images"]
            for i in range(len(images_list)):
                id = folder + "_" + str(images_list[i]["id"])
                file_path = folder + "//" + images_list[i]["file_name"]
                res[id] = {"path": file_path, "objects": [], "width": images_list[i]["width"],
                           "height": images_list[i]["height"]}

            annotations_list = data["annotations"]
            for i in range(len(annotations_list)):
                id = folder + "_" + str(annotations_list[i]["image_id"])
                box = [0, 0, 0, 0]
                box[1] = annotations_list[i]['bbox'][0] / res[id]['width']
                box[0] = annotations_list[i]['bbox'][1] / res[id]['height']
                box[3] = annotations_list[i]['bbox'][0] / res[id]['width'] + annotations_list[i]['bbox'][2] / res[id][
                    'width']
                box[2] = annotations_list[i]['bbox'][1] / res[id]['height'] + annotations_list[i]['bbox'][3] / res[id][
                    'height']
                object = {'bbox': box, 'class': CLASSES[annotations_list[i]['category_id'] - 1], 'score': 1.0,
                          'type': 'groundtruth'}
                res[id]['objects'].append(object)
    return res
