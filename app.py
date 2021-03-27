import numpy as np
from PIL import Image, ImageFont, ImageDraw
import sys

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def warp(im, size_new, pa):
    size_old = im.size
    pb = ((0, 0), (size_old[0], 0), size_old, (0, size_old[1]))
    coeffs = find_coeffs(pa, pb)
    out = im.transform(size_new, Image.PERSPECTIVE, coeffs, Image.BICUBIC) 
    return out

def make_text(text, size=(100, 140)):
    out = Image.new('RGBA', size, (255, 255, 255, 0))

    txt_len = len(text)
    if txt_len == 1:
        txt_size = 120
        position = (13,9)
    elif txt_len == 2:
        txt_size = 90
        position = (0,0)
    elif txt_len == 3:
        txt_size = 120
        position = (0,0)
    else:
        txt_size = 100
        position = (0,0)

    fnt = ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', txt_size)
    d = ImageDraw.Draw(out)
    d.text(position, text, font=fnt, fill=(0,0,0,255))
    return out

def compositor(im_arr):
    assert(len(im_arr) >= 2)
    out = im_arr[0]
    for im in im_arr[1:]:
        out = Image.alpha_composite(out, im)
    return out


def main(a_text, b_text):

    base = Image.open('./base.png').convert('RGBA')
    base_size = base.size

    a_pos = ((980, 160), (1087, 153), (1136, 289), (1026, 304))
    a_im = warp(make_text(a_text), base_size, a_pos)

    b_pos = ((958, 342), (1063, 341), (1102, 477), (1000, 480))
    b_im = warp(make_text(b_text), base_size, b_pos)

    out = compositor((base, a_im, b_im))

    out.save('./out.png', 'PNG')
    return 1

if __name__ == '__main__':
    a_text = sys.argv[1]
    b_text = sys.argv[2]

    main(a_text, b_text)
