import tensorflow as tf
import tensorflow_hub as hub
import utils
import time
import matching
import os
import gc

"""
Для каждого изображения запустим модель обнаружения объектов и сравним результаты её работы с groundtruth 
Будем считать, что модель обнаружила объект, если его detection score >= SCORE_THRESHOLD
Для каждого изображения определим набор true positives: сопоставим объекты groundtruth с обнаруженными, и объявим true positives те из них, у которых мера intersection over union >= IOU_THRESHOLD

Вычислим метрики (для каждого класса):
precision - отношение количества true positives к количеству обнаруженных объектов (суммируя по всем изображениям)
recall - отношение количества true positives к количеству groundtruth (суммируя по всем изображениям)



Достоинства и недостатки решения

Достоинства:
1)Используемая модель была предобучена на изображениях из огромного датасета, поэтому она обладает хорошей обобщающей способностью
2)Стсутствие обучения модели на кастомном датасете позволяет вычислять метрики precision и recall на основе всей выборки изображений, а не только её валидационной части

Недостатки:
1)Модель дает предсказания для всех 600 классов Openimages, что значительно замедляет ее работу (когда как нам нужны предсказания только для 7 классов)
2)Метрики precision и recall зависят от двух произвольно выбираемых значений констант: SCORE_THRESHOLD и IOU_THRESHOLD
3)В модели отсутствует класс 'Puffin', поэтому вместо него был использован класс 'Bird'
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #При раскомментировании будет использоваться CPU

DO_VISUALIZE = False

#Порог prediction score, при превышении которого будет считаться, что модель обнаружила объект
SCORE_THRESHOLD = 0.3

#Порог меры 'intersection over union' для сопоставления обнаруженного объекта с groundtruth
IOU_THRESHOLD = 0.6

CLASSES = ['Fish', 'Jellyfish', 'Penguin', 'Bird', 'Shark', 'Starfish', 'Rays and skates']

"""
Запускаем модель на одном изображении
Функция возвращает 
1)изображение (для визуализации)
2)список обнаруженных объектов в формате: каждый объект - словарь, который содержит bounding box (список относительных координат в формате x1,y1,x2,y2), метку класса и confidence score
3)время работы модели
"""
def run_detector(detector, path):
    img = utils.load_img(path)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()
    result = {key: value.numpy() for key, value in result.items()}

    res_objects = []
    for i in range(len(result["detection_boxes"])):
        object = {}
        object['bbox'] = list(result["detection_boxes"][i])
        object['score'] = result["detection_scores"][i]
        object['class'] = str(result["detection_class_entities"][i]).replace('b\'', '').replace('\'', '')
        object['type'] = 'prediction'
        res_objects.append(object)

    time_elapsed = end_time - start_time
    gc.collect()
    return img, res_objects, time_elapsed

#Загружаем модель
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

#Собираем метаинформацию обо всех входных данных (пути к файлам и groundtruth объекты)
res = utils.collect_data(["data//train", "data//test", "data//valid"], CLASSES)
keys = list(res.keys())

#Список groundtruth объектов (по классам)
groundtruth_arrays = [[] for i in range(len(CLASSES))]

#Список обнаруженных объектов (по классам)
predictions_arrays = [[] for i in range(len(CLASSES))]

#Список true positive объектов (по классам)
matched_arrays = [[] for i in range(len(CLASSES))]

#Список времен работы модели для каждого изображения
time_array = []

for i in range(len(keys)):
    key = keys[i]
    path = res[key]["path"]
    img, predicted, t = run_detector(detector, path)
    groundtruth = res[key]["objects"]
    time_array.append(t)

    extra = []
    for j in range(len(CLASSES)):
        #Для каждого класса запускаем алгоритм выделения true positives
        #объект считается true positive, если его score >= SCORE_THRESHOLD и он сопоставляется с каким-то объектом из groundtruth (IOU >= IOU_THRESHOLD)
        matched, class_predictions, class_groundtruth = matching.match(predicted, groundtruth, CLASSES[j],
                                                                       SCORE_THRESHOLD, IOU_THRESHOLD)
        groundtruth_arrays[j].extend(class_groundtruth)
        predictions_arrays[j].extend(class_predictions)
        matched_arrays[j].extend(matched)

        if DO_VISUALIZE:
            for k in range(len(matched)):
                matched[k][0]['type'] = 'match'
                extra.append(matched[k][0])

    #При визуализации рисуем:
    #синим цветом - рамки для groundtruth
    #красным цветом - рамки для predicted
    #зеленым цветом - рамки для true positives
    #Поскольку для любого true positive будет нарисована зеленая рамка поверх красной,
    #визуально будет казаться, что красные рамки построены только для тех predictions, для которых не нашлось пары
    if DO_VISUALIZE:
        objects = []
        objects.extend(groundtruth)
        objects.extend(predicted)
        objects.extend(extra)

        image_with_boxes = utils.draw_boxes(img.numpy(), objects, SCORE_THRESHOLD)
        utils.display_image(image_with_boxes)

#Вычисляем метрики
for j in range(len(CLASSES)):
    matched_count = len(matched_arrays[j])
    groundtruth_count = len(groundtruth_arrays[j])
    predictions_count = len(predictions_arrays[j])

    if predictions_count != 0 and groundtruth_count != 0:
        #precision - отношение true positives к количеству всех обнаруженных объектов данного класса
        #recall - отношение true positives к количеству всех groundtruth объектов данного класса
        precision = matched_count / predictions_count
        recall = matched_count / groundtruth_count
        print(CLASSES[j] + ": ", 'Precision = ', round(precision, 3), '  Recall = ', round(recall, 3))

print('Average time', round(sum(time_array) / len(time_array), 2), 'sec')
