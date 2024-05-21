from PIL import Image, ImageDraw, ImageFont
import random
import string
import os 
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import convert_bbox_to_yolo
from params import *


fonts = os.listdir(path_fonts)
text_colors = ["#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
background_colors = [ "#F8F8F8", "#E5E5E5", "#D2D2D2", "#FFFFFF", "#F0F0F0", "#FAFAFA", "#EFEFEF", "#F5F5F5"]

dict_char = {k:i for i,k in enumerate(string.ascii_lowercase + string.digits)}

def generate_images(n, label_dir, images_dir, multiclass, type_generate, xy = (0,0)):
    if type_generate == "random_words":
        with open("Datasets/lexicon.txt", 'r') as file:
            words = file.readlines()
            words = [w[:-1] for w in words]
    for i in range(n):
        if type_generate == "random_words":
            new_str = words[random.randint(0, len(words)-1)]
        else:
            z = random.randint(3, 10)
            new_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=z))
        
        with open(os.path.join(label_dir, f"{new_str}.txt"), 'w') as file:

            font = fonts[random.randint(0, len(fonts)-1)]
            
            text_color = random.choice(text_colors)
            background_color = random.choice(background_colors)
            while text_color == background_color:
                background_color = random.choice(background_colors)
            
            font = ImageFont.truetype(path_fonts+f'/{font}', random.randint(25,40), encoding="unic")
            _, _, w, h = font.getbbox(new_str)
            size = (w + random.randint(10, 20), h + random.randint(10,20))
            img = Image.new(mode="RGB", size=size, color= background_color)

            draw = ImageDraw.Draw(img)
            draw.text(xy, new_str, font = font, fill = text_color)
            for i, char in enumerate(new_str):
                _, _, right, _ = font.getbbox(new_str[:i+1])
                _, _, _, bottom = font.getbbox(char)
                width, height = font.getmask(char).size
                right += xy[0] # bbox[2]
                bottom += xy[1] # bbox[3]
                top = bottom - height #bbox[1]
                left = right - width #bbox[0]
                
                #draw.rectangle((left, top, right, bottom), None, "#f00")

                bbox_yolo = convert_bbox_to_yolo((left, top, width, height), size[0], size[1])
                if multiclass:
                    char_index = dict_char[char]
                else:
                    char_index = 0 

                file.write(f"{char_index} {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")
            
            file.close()
            img.save(os.path.join(images_dir, f"{new_str}.jpg"))


generate_images(n_train_images, train_labels, train_images, multiclass, type_generate)
generate_images(n_test_images, test_labels, test_images, multiclass, type_generate)