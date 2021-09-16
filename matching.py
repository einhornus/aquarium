import utils

"""
Вычисление длины пересечения отрезков
"""
def overlap_1d(from1, to1, from2, to2):
    if from1 <= from2 and to1 >= to2:
        return to2 - from2
    if from2 <= from1 and to2 >= to1:
        return to1 - from1
    if from1 <= from2 and to1 >= from2:
        return to1 - from2
    if from2 <= from1 and to2 >= from1:
        return to2 - from1
    return 0

"""
Вычисление площади пересечения двух bounding boxes
"""
def overlap(box1, box2):
    overlap_x = overlap_1d(box1[0], box1[2], box2[0], box2[2])
    overlap_y = overlap_1d(box1[1], box1[3], box2[1], box2[3])
    return overlap_x * overlap_y

"""
Вычисление меры 'intersection over union' для двух bounding boxes
"""
def iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    intersection = overlap(box1, box2)
    union = area1 + area2 - intersection
    res = intersection / union
    return res

"""
Сопоставление обнаруженных объектов с groundtruth для класса cl
predictions - список обнаруженных объектов
groundtruth - список groundtruth объектов
cl - метка класса
score_threshold - порог степени уверенности обнаруженного объекта
iou_threshold - порог меры 'intersection over union' для сопоставления

Функция возвращает:
matched - список true positive пар объектов
class_predictions - подсписок predictions, включающий только объекты класса cl со score больше score_threshold
class_groundtruth - подсписок groundtruth, включающий только объекты класса cl со score больше score_threshold
"""
def match(predictions, groundtruth, cl, score_threshold, iou_threshold):
    #Сначала фильтруем списки predictions и groundtruth
    class_predictions = []
    class_groundtruth = []
    for i in range(len(predictions)):
        predictions[i]["class"] = predictions[i]["class"].replace('b\'', '').replace('\'', '')
        if cl == predictions[i]["class"] and predictions[i]["score"] >= score_threshold:
            class_predictions.append(predictions[i])
    for i in range(len(groundtruth)):
        if cl == groundtruth[i]["class"] and groundtruth[i]["score"] >= score_threshold:
            class_groundtruth.append(groundtruth[i])

    #Составляем список всех возможных пар объектов, которые можно сопоставить
    similarities = []
    for i in range(len(class_predictions)):
        for j in range(len(class_groundtruth)):
            iou_metric = iou(class_predictions[i]["bbox"], class_groundtruth[j]["bbox"])
            similarities.append((i, j, iou_metric))
    #Сортируем его по убыванию iou
    similarities.sort(key=lambda x: -x[2])

    #Делаем сопоставление жадным алгоритмом: идем по списку similarities и матчим пару, если оба её элемента еще не были использованы
    #Вместо жадного алгоритма можно использовать, например, венгерский алгоритм
    matched_class_predictions = set()
    matched_class_groundtruth = set()
    matched = []
    for i in range(len(similarities)):
        if not (similarities[i][0] in matched_class_predictions):
            if not (similarities[i][1] in matched_class_groundtruth):
                if similarities[i][2] >= iou_threshold:
                    matched.append((class_predictions[similarities[i][0]], class_groundtruth[similarities[i][1]]))
                    matched_class_predictions.add(similarities[i][0])
                    matched_class_groundtruth.add(similarities[i][1])
    return matched, class_predictions, class_groundtruth


#print(overlap_1d(-3, 2, 0, 0.5))
#pr = [{'bbox': [0.6362368, 0.62010753, 0.8589742, 0.66686994], 'score': 0.90429866, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.12519206, 0.6101476, 0.24400336, 0.75714356], 'score': 0.74647385, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.5715549, 0.093804404, 0.82819766, 0.20080344], 'score': 0.71052593, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.90828574, 0.61789, 0.95021385, 0.63534766], 'score': 0.48280552, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.6428412, 0.8883457, 0.9831361, 0.9934382], 'score': 0.4677391, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.49692205, 0.78604954, 0.7210407, 0.82842946], 'score': 0.41472638, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.65030766, 0.8885453, 0.98702234, 0.99466634], 'score': 0.39829555, 'class': "b'Tree'", 'type': 'prediction'}, {'bbox': [0.17921634, 0.577181, 0.21822396, 0.610638], 'score': 0.3598838, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.510792, 0.49062878, 0.5340241, 0.5211273], 'score': 0.289437, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.12407054, 0.60078925, 0.23625417, 0.75078434], 'score': 0.2864389, 'class': "b'Marine mammal'", 'type': 'prediction'}, {'bbox': [0.5707174, 0.8446256, 0.6421037, 0.9173937], 'score': 0.284542, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.77382135, 0.49830058, 0.8003222, 0.5405676], 'score': 0.22770266, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.57590383, 0.08526244, 0.8305169, 0.19861682], 'score': 0.22473767, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.13386068, 0.60719746, 0.23189232, 0.7507282], 'score': 0.1787337, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.009186434, 0.17770873, 0.1760716, 0.4723451], 'score': 0.13658893, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.6523452, 0.67101455, 0.6925452, 0.6892447], 'score': 0.07661937, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.7451047, 0.7161177, 0.76793665, 0.73235077], 'score': 0.069418974, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.57001764, 0.8381446, 0.64553016, 0.9226664], 'score': 0.06615756, 'class': "b'Bird'", 'type': 'prediction'}, {'bbox': [0.0, 0.0, 0.9823681, 0.38745573], 'score': 0.05938895, 'class': "b'Tree'", 'type': 'prediction'}, {'bbox': [0.89213634, 0.39024094, 0.94074047, 0.41375214], 'score': 0.056568198, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.73870945, 0.71443844, 0.77072227, 0.7339479], 'score': 0.041387737, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.53266406, 0.4511419, 0.56035846, 0.48952675], 'score': 0.03943871, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.124859415, 0.6001527, 0.24254929, 0.748809], 'score': 0.034609515, 'class': "b'Dolphin'", 'type': 'prediction'}, {'bbox': [0.0, 0.01959876, 0.12106557, 0.2324099], 'score': 0.030605381, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.9044196, 0.6214804, 0.949538, 0.6390985], 'score': 0.029641483, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.0011786397, 0.0, 0.92745996, 0.30654857], 'score': 0.023230366, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.1278816, 0.6017435, 0.23841955, 0.74972373], 'score': 0.022616662, 'class': "b'Whale'", 'type': 'prediction'}, {'bbox': [0.0, 0.09316101, 0.105093926, 0.17362128], 'score': 0.022180503, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.6464965, 0.6344498, 0.6974987, 0.6629532], 'score': 0.021590285, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.64575154, 0.6417723, 0.7264621, 0.6688573], 'score': 0.021117534, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.6375357, 0.6181487, 0.85596305, 0.67891747], 'score': 0.018630069, 'class': "b'Marine mammal'", 'type': 'prediction'}, {'bbox': [0.6474179, 0.67204434, 0.69205326, 0.6899507], 'score': 0.018571705, 'class': "b'Marine invertebrates'", 'type': 'prediction'}, {'bbox': [0.00071978255, 0.2517826, 0.03796022, 0.2952301], 'score': 0.018261123, 'class': "b'Flower'", 'type': 'prediction'}, {'bbox': [0.12354413, 0.6048895, 0.24611975, 0.75595516], 'score': 0.01735647, 'class': "b'Bird'", 'type': 'prediction'}, {'bbox': [0.0, 0.013212776, 0.076244816, 0.10608919], 'score': 0.01669376, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.0, 0.07826471, 0.24660793, 0.47160643], 'score': 0.016061757, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.6316563, 0.62398595, 0.87137425, 0.66447854], 'score': 0.015809348, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.67903244, 0.866226, 0.7230634, 0.93151975], 'score': 0.014543521, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.5143761, 0.48699713, 0.5418077, 0.5259279], 'score': 0.014157498, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.9018405, 0.6162651, 0.9475732, 0.6351897], 'score': 0.013548102, 'class': "b'Marine invertebrates'", 'type': 'prediction'}, {'bbox': [0.9709424, 0.41671172, 0.999081, 0.4794583], 'score': 0.012333071, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.6416357, 0.6267203, 0.8052684, 0.6629034], 'score': 0.012284822, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [5.989075e-05, 0.088776596, 0.0778631, 0.12046521], 'score': 0.011797493, 'class': "b'Fashion accessory'", 'type': 'prediction'}, {'bbox': [0.13096528, 0.6036751, 0.23013176, 0.7484911], 'score': 0.01163591, 'class': "b'Duck'", 'type': 'prediction'}, {'bbox': [0.0063562044, 0.09016461, 0.0731987, 0.11979882], 'score': 0.011269544, 'class': "b'Flower'", 'type': 'prediction'}, {'bbox': [0.12683825, 0.61169213, 0.23380733, 0.7416578], 'score': 0.011153356, 'class': "b'Shark'", 'type': 'prediction'}, {'bbox': [0.0, 0.056066446, 0.13983122, 0.30094078], 'score': 0.010310184, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.0, 0.01340332, 0.07522232, 0.10506158], 'score': 0.010133556, 'class': "b'Tree'", 'type': 'prediction'}, {'bbox': [0.49336314, 0.7867524, 0.7077059, 0.827913], 'score': 0.010079484, 'class': "b'Marine mammal'", 'type': 'prediction'}, {'bbox': [0.018301774, 0.17048196, 0.1924146, 0.45679197], 'score': 0.009551565, 'class': "b'Marine invertebrates'", 'type': 'prediction'}, {'bbox': [0.5684358, 0.84731555, 0.6414789, 0.9180851], 'score': 0.009136734, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.5292179, 0.45230472, 0.5564343, 0.4890513], 'score': 0.008517012, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.5307509, 0.45043677, 0.56105447, 0.49001205], 'score': 0.008408253, 'class': "b'Bird'", 'type': 'prediction'}, {'bbox': [0.9034278, 0.6209811, 0.9508012, 0.63781613], 'score': 0.00837409, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.0, 0.09118266, 0.10723244, 0.17220855], 'score': 0.008073567, 'class': "b'Person'", 'type': 'prediction'}, {'bbox': [0.64554214, 0.6210512, 0.8449423, 0.6623108], 'score': 0.008031137, 'class': "b'Mammal'", 'type': 'prediction'}, {'bbox': [0.0007194297, 0.09691736, 0.10286189, 0.16682327], 'score': 0.007854006, 'class': "b'Footwear'", 'type': 'prediction'}, {'bbox': [0.973629, 0.1795644, 0.99855775, 0.22118343], 'score': 0.0077552367, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.013433431, 0.010635032, 0.8911839, 0.56759024], 'score': 0.0073776576, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.6519605, 0.64988434, 0.7153827, 0.6709402], 'score': 0.007052427, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.12507418, 0.6094624, 0.23921871, 0.74202245], 'score': 0.006841859, 'class': "b'Mammal'", 'type': 'prediction'}, {'bbox': [0.9683902, 0.18987271, 0.99776196, 0.21895777], 'score': 0.0067452895, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.57361346, 0.09072603, 0.83078384, 0.19410156], 'score': 0.006567569, 'class': "b'Mammal'", 'type': 'prediction'}, {'bbox': [0.73375916, 0.7159103, 0.769545, 0.7339781], 'score': 0.0064184368, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.66670984, 0.869629, 0.72503155, 0.9494768], 'score': 0.006369319, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.50019485, 0.78942764, 0.7136209, 0.83001983], 'score': 0.006314938, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.6244884, 0.6173501, 0.86242557, 0.66666263], 'score': 0.00614435, 'class': "b'Bird'", 'type': 'prediction'}, {'bbox': [0.64925796, 0.6687445, 0.69649446, 0.68615216], 'score': 0.0060084993, 'class': "b'Marine invertebrates'", 'type': 'prediction'}, {'bbox': [0.08978153, 0.34917817, 0.16651647, 0.4415139], 'score': 0.005950343, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.64899635, 0.8828591, 0.99659437, 0.99927574], 'score': 0.005184646, 'class': "b'Houseplant'", 'type': 'prediction'}, {'bbox': [0.6407411, 0.6198731, 0.848505, 0.66319007], 'score': 0.0049854796, 'class': "b'Person'", 'type': 'prediction'}, {'bbox': [0.5246681, 0.44477156, 0.5631194, 0.50806487], 'score': 0.004952191, 'class': "b'Bird'", 'type': 'prediction'}, {'bbox': [0.5085556, 0.49086452, 0.5340658, 0.52011734], 'score': 0.004934921, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.01111667, 0.17974849, 0.17374837, 0.456587], 'score': 0.004922891, 'class': "b'Animal'", 'type': 'prediction'}, {'bbox': [0.17567118, 0.5762185, 0.22057188, 0.6161119], 'score': 0.004798463, 'class': "b'Marine invertebrates'", 'type': 'prediction'}, {'bbox': [0.90690047, 0.6189058, 0.9500584, 0.6353868], 'score': 0.004588131, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.63684195, 0.8958809, 0.70393026, 0.9575122], 'score': 0.00433938, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.0, 0.09332301, 0.10821429, 0.17063218], 'score': 0.0041805184, 'class': "b'Clothing'", 'type': 'prediction'}, {'bbox': [0.17212264, 0.5771304, 0.2220184, 0.63087136], 'score': 0.0041598976, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.0, 0.0927876, 0.108291246, 0.17224601], 'score': 0.004091678, 'class': "b'Mammal'", 'type': 'prediction'}, {'bbox': [0.640658, 0.637339, 0.79934424, 0.6688963], 'score': 0.0040685404, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.5598705, 0.0975685, 0.844025, 0.20005801], 'score': 0.003934475, 'class': "b'Marine mammal'", 'type': 'prediction'}, {'bbox': [0.51342106, 0.78527504, 0.7239648, 0.82842034], 'score': 0.0038016473, 'class': "b'Marine invertebrates'", 'type': 'prediction'}, {'bbox': [0.58890826, 0.84490377, 0.63840514, 0.89537376], 'score': 0.003773982, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.0012145551, 0.14435604, 0.18022396, 0.45825472], 'score': 0.0037470919, 'class': "b'Tree'", 'type': 'prediction'}, {'bbox': [0.97165877, 0.18402351, 0.9982367, 0.22454917], 'score': 0.0035632201, 'class': "b'Person'", 'type': 'prediction'}, {'bbox': [0.0, 0.09321966, 0.09035591, 0.13605632], 'score': 0.003556243, 'class': "b'Fashion accessory'", 'type': 'prediction'}, {'bbox': [0.0, 0.09441006, 0.09705236, 0.16357982], 'score': 0.0034284333, 'class': "b'Fashion accessory'", 'type': 'prediction'}, {'bbox': [0.00027020773, 0.0, 0.5746267, 0.39834106], 'score': 0.0033114492, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.8264297, 0.6302201, 0.8680645, 0.6542715], 'score': 0.003303229, 'class': "b'Flower'", 'type': 'prediction'}, {'bbox': [0.0, 0.0882641, 0.089394175, 0.1276953], 'score': 0.0032118587, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.6493179, 0.67188096, 0.69431645, 0.6896725], 'score': 0.003167264, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.0027547933, 0.08786205, 0.0787706, 0.12325407], 'score': 0.003137902, 'class': "b'Footwear'", 'type': 'prediction'}, {'bbox': [0.56662816, 0.84206843, 0.64752644, 0.92144835], 'score': 0.0031328097, 'class': "b'Plant'", 'type': 'prediction'}, {'bbox': [0.5654673, 0.8513882, 0.6274615, 0.9203169], 'score': 0.0030055684, 'class': "b'Bird'", 'type': 'prediction'}, {'bbox': [0.67062926, 0.86602134, 0.726965, 0.9421079], 'score': 0.0029874942, 'class': "b'Marine invertebrates'", 'type': 'prediction'}, {'bbox': [0.7369697, 0.7139321, 0.7722394, 0.7348115], 'score': 0.0029841939, 'class': "b'Bird'", 'type': 'prediction'}, {'bbox': [0.0, 0.096651904, 0.10291198, 0.16868919], 'score': 0.0029798893, 'class': "b'Fish'", 'type': 'prediction'}, {'bbox': [0.96793145, 0.42001274, 1.0, 0.4782153], 'score': 0.0029447952, 'class': "b'Mammal'", 'type': 'prediction'}, {'bbox': [0.6448106, 0.620859, 0.875523, 0.6766025], 'score': 0.0029171556, 'class': "b'Marine invertebrates'", 'type': 'prediction'}]
#gt = [{'bbox': [0.5755208333333334, 0.103515625, 0.8333333333333334, 0.1708984375], 'class': 'Shark', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.6380208333333334, 0.623046875, 0.875, 0.662109375], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.13932291666666666, 0.6103515625, 0.24348958333333331, 0.7470703125], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.5455729166666666, 0.798828125, 0.6497395833333333, 0.810546875], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.03125, 0.7919921875, 0.10546875, 0.919921875], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.5859375, 0.83984375, 0.64453125, 0.8828125], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.8984375, 0.6123046875, 0.95703125, 0.6357421875], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.6705729166666666, 0.873046875, 0.7291666666666666, 0.9208984375], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.17578125, 0.572265625, 0.21875, 0.61328125], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.7643229166666666, 0.501953125, 0.8033854166666666, 0.5419921875], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.5299479166666666, 0.4541015625, 0.5625, 0.4912109375], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}, {'bbox': [0.5078125, 0.4912109375, 0.5364583333333334, 0.5185546875], 'class': 'Fish', 'score': 1.0, 'type': 'groundtruth'}]
#match(pr, gt, 'Fish', utils.SCORE_THRESHOLD, 0.05)
