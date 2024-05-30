import torch
import cv2
import numpy as np
import os
import json
from craft import CRAFT
from collections import OrderedDict
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import loadImage, normalizeMeanVariance, resize_aspect_ratio
from file_utils import saveResult

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.'
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def test_single_image(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = torch.unsqueeze(x, 0)

    # move to GPU if specified
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, _ = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys, score_text

# Configuración de parámetros
trained_model = 'detection/CRAFT/CRAFT-pytorch/craft_mlt_25k.pth'  # Path to pretrained model
image_folder = 'dataset/archive(4)/Challenge2_Test_Task12_Images/'  # Path to the input images folder
result_image_folder = 'detection/CRAFT/CRAFT-pytorch/result/'  # Path to save the result images
results_json_path = 'detection/CRAFT/CRAFT-pytorch/results.json'  # Path to save the results JSON
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4
cuda = torch.cuda.is_available()  # Use GPU if available
canvas_size = 1280
mag_ratio = 1.5
poly = False

# Cargar modelo
net = CRAFT()
print(f'Loading weights from checkpoint ({trained_model})')
net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cuda' if cuda else 'cpu')))
if cuda:
    net = net.cuda()
net.eval()

results = {"unknown": "###", "annots": {}}

# Procesar cada imagen en la carpeta
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        try:
            image_path = os.path.join(image_folder, image_name)
            result_image_path = os.path.join(result_image_folder, os.path.splitext(image_name)[0] + '_result')

            # Cargar imagen
            image = loadImage(image_path)

            # Procesar la imagen
            bboxes, polys, score_text = test_single_image(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio)

            # Guardar los resultados de imagen
            saveResult(image_path, image[:, :, ::-1], polys, dirname=os.path.dirname(result_image_path))
            print(f'Results saved to {result_image_path}')

            # Guardar los resultados en el formato requerido
            results["annots"][image_name] = {"bbox": bboxes, "text": ["###" for _ in bboxes]}
        except Exception as e:
            print(f'Error processing {image_name}: {e}')

# Guardar el JSON con los resultados obtenidos
with open(results_json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f'Results JSON saved to {results_json_path}')
