#!/usr/bin/python

import sys
from PIL import Image
import numpy as np

def avhash(im):
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    im = im.resize((8, 8), Image.ANTIALIAS).convert('1').convert('L')
    avg = reduce(lambda x, y: x + y, im.getdata()) / 64.
    return reduce(lambda x, (y, z): x | (z << y),
                  enumerate(map(lambda i: 0 if i < avg else 1, im.getdata())),
                  0)

def hamming(h1, h2):
    h, d = 0, h1 ^ h2
    while d:
        h += 1
        d &= d - 1
    return h

def phash_simmilarity(img1,img2):
    
    hash1 = avhash(img1)
    hash2 = avhash(img2)
    dist = hamming(hash1, hash2)
    simm = (64 - dist) * 100 / 64
    # print simm
    # if not isinstance (img1,Image.Image):
    #     img1 = Image.open(img1)
    # if not isinstance(img2,Image.Image):
    #     img2 = Image.open(img2)
    # hash1 = imagehash.phash(img1)
    # hash2 = imagehash.phash(img2)
    # dist = hamming(hash2,hash1)
    # simm = (64 - dist) * 100 / 64
    return simm

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: %s img1 img2" % sys.argv[0]
    else:
        img1 = sys.argv[1]
        img2 = sys.argv[2]
        print "similarity = %d%%" % (phash_simmilarity(img1,img2))