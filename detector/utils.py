import fitz
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from PIL import ImageFont

class UnionFindSet:
    def __init__(self, max_n=100):
        self.parent = [i for i in range(max_n)]
        
    def union(self, i, j):
        self.parent[i] = self.find_parent(i)
        self.parent[j] = self.parent[i]

    def find_parent(self, i):
        if i == self.parent[i]:
            return i
        else:
            parent = self.find_parent(self.parent[i])
            self.parent[i] = parent
            return parent

def pil2cv(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

def cv2pil(cv2_img):
    img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    return img

def page2img(page):
    pix = page.get_pixmap()
    mode = "RGBA" if pix.alpha else 'RGB'
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img

def doc2imgs(doc_fp, img_type='cv2'):
    doc = fitz.open(doc_fp)
    images = []
    for page in doc:
        p_img = page2img(page)
        if img_type == 'cv2':
            p_img = pil2cv(p_img)
        images.append(p_img)
    return images

def merge_boxes(boxes):
    x0s = [b[0] for b in boxes]
    y0s = [b[1] for b in boxes]
    x1s = [b[2] for b in boxes]
    y1s = [b[3] for b in boxes]
    return (min(x0s), min(y0s), max(x1s), max(y1s))
        
def is_vh_line_interact(line_a, line_b, margin=1):
    a0, a1 = line_a
    b0, b1 = line_b
    if a0 > a1:
        a0, a1 = a1, a0
    if b0 > b1:
        b0, b1 = b1, b0
    if a1 < b0 - margin or a0 > b1 + margin:
        return False
    else:
        return True

def equal_margin(a, b, margin=1):
    if abs(a - b) <= margin:
        return True
    else:
        return False
    
def is_two_rect_share_edge(rect_a, rect_b, margin=1):
    xa0,ya0,xa1,ya1 = rect_a[0], rect_a[1], rect_a[2], rect_a[3]
    xb0,yb0,xb1,yb1 = rect_b[0], rect_b[1], rect_b[2], rect_b[3]
#     if is_vh_line_interact((xa0, xa1), (xb0,xb1)) or \
#         is_vh_line_interact((xa0, xa1), (yb0,yb1)) or \
#         is_vh_line_interact((ya0, ya1), (xb0,xb1)) or \
#         is_vh_line_interact((ya0, ya1), (yb0,yb1)):
#         return True
    if ((equal_margin(ya0, yb0, margin) or equal_margin(ya0, yb1, margin) or equal_margin(ya1, yb0, margin) or equal_margin(ya1,yb1, margin)) and is_vh_line_interact((xa0, xa1), (xb0, xb1), margin)) or \
    ((equal_margin(xa0, xb0, margin) or equal_margin(xa0, xb1, margin) or equal_margin(xa1, xb0, margin) or equal_margin(xa1, xb1, margin)) and is_vh_line_interact((ya0, ya1), (yb0, yb1), margin)) :
        return True
    else:
        return False

def rect_area(rect):
    return (rect[2]-max(rect[0], 0)) * (rect[3] - max(rect[1], 0)) * 1.0

def intersect_area(rect_a, rect_b):
    # rect_1 Rect(x_a_0,y_a_0, x_b_)
    x_a_0, y_a_0, x_a_1, y_a_1 = rect_a
    x_b_0, y_b_0, x_b_1, y_b_1 = rect_b
    w_a = x_a_1 - x_a_0
    h_a = y_a_1 - y_a_0
    w_b = x_b_1 - x_b_0
    h_b = y_b_1 - y_b_0
    return max(0.0, min(y_a_1, y_b_1) - max(y_a_0, y_b_0)) * max(0.0, min(x_a_1, x_b_1) - max(x_a_0, x_b_0))
#     x_a_0, y_a_0 = max(0, x_a_0), max(0, y_a_0)
#     x_b_0, y_b_0 = max(0, x_b_0), max(0, y_b_0)
#     if x_a_1 <= x_b_0 or y_a_1 <= y_b_0: 
#         return 0
#     return min((x_a_1 - x_b_0), w_a, w_b) * min((y_a_1 - y_b_0), h_a, h_b) * 1.0

def union_ratio(rect_a, rect_b):
    intersect = intersect_area(rect_a, rect_b)
    rect_a_area = rect_area(rect_a)
    if rect_a_area > 10:
        return intersect_area(rect_a, rect_b) / (rect_area(rect_a) * 1.0)
    else:
        return 1

def iou(rect_a, rect_b):
    union = rect_area(rect_a) + rect_area(rect_b) - intersect_area(rect_a, rect_b)
    return intersect_area(rect_a, rect_b) / union
    
def filt_inner_drawing(drawings):
    n = len(drawings)
    out_index = []
    for i in range(n):
        draw_i = drawings[i]
        rect_i = draw_i['rect']
        for j in range(n):
            if i == j:
                continue
            draw_j = drawings[j]
            rect_j = draw_j['rect']
            
            if union_ratio(rect_i, rect_j) >= 1:
                temp = j
#                 if i == 8:
#                     import pdb;pdb.set_trace()
                out_index.append(i)
                break
    filtered_drawings = []
    for i, draw in enumerate(drawings):
        if i not in out_index:
            filtered_drawings.append(draw)
    return filtered_drawings    

def inside(rect_a, rect_b):
    area_a = rect_area(rect_a)
    area_b = rect_area(rect_b)
    if area_a < area_b:
        rect_a, rect_b = rect_b, rect_a
    
    xa0,ya0,xa1,ya1 = rect_a[0], rect_a[1], rect_a[2], rect_a[3]
    xb0,yb0,xb1,yb1 = rect_b[0], rect_b[1], rect_b[2], rect_b[3]
    
    if xb0 >= xa0 and yb0 >= ya0 and xa1 >= xb1 and ya1 >= yb1:
        return True
    else:
        return False
    
def pil_draw_bbox(img, xy, color=(255,0,0), label=None):
    draw = ImageDraw.Draw(img)
    draw.rectangle(xy, outline=color)
    if label is not None:
        draw.text((xy[0], xy[1]-10), label, fill=color)
    return img