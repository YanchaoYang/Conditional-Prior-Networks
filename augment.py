import numpy as np
import random

def left_right( img ):
    img = img[:,:,::-1,:]
    return img

def top_down( img ):
    img = img[:,::-1,:,:]
    return img
    
def flip_left_right(img1, img2, edge1, edge2, flow):
    img1  = left_right(img1)
    img2  = left_right(img2)
    edge1 = left_right(edge1)
    edge2 = left_right(edge2)
    flow  = left_right(flow)
    flow[:,:,:,0] *= -1.0
    return img1, img2, edge1, edge2, flow

def flip_top_down(img1, img2, edge1, edge2, flow):
    img1  = top_down(img1)
    img2  = top_down(img2)
    edge1 = top_down(edge1)
    edge2 = top_down(edge2)
    flow  = top_down(flow)
    flow[:,:,:,1] *= -1.0
    return img1, img2, edge1, edge2, flow

def rotate_180(img1, img2, edge1, edge2, flow):
    img1, img2, edge1, edge2, flow = flip_left_right(img1, img2, edge1, edge2, flow)
    img1, img2, edge1, edge2, flow = flip_top_down(  img1, img2, edge1, edge2, flow)
    return img1, img2, edge1, edge2, flow

def random_flip(img1, img2, edge1, edge2, flow):
    t = random.randint(0, 3)
    #print(t)
    if t==1:
        img1, img2, edge1, edge2, flow = flip_left_right(img1, img2, edge1, edge2, flow)
    elif t==2:
        img1, img2, edge1, edge2, flow = flip_top_down(img1, img2, edge1, edge2, flow)
    elif t==3:
        img1, img2, edge1, edge2, flow = rotate_180(img1, img2, edge1, edge2, flow)

    return img1, img2, edge1, edge2, flow

def flip_left_right_no_edge(img1, img2, flow, vmap):
    img1  = left_right(img1)
    img2  = left_right(img2)
    flow  = left_right(flow)
    vmap  = left_right(vmap)
    flow[:,:,:,0] *= -1.0
    return img1, img2, flow, vmap

def flip_top_down_no_edge(img1, img2, flow, vmap):
    img1  = top_down(img1)
    img2  = top_down(img2)
    flow  = top_down(flow)
    vmap  = top_down(vmap)
    flow[:,:,:,1] *= -1.0
    return img1, img2, flow, vmap

def rotate_180_no_edge(img1, img2, flow, vmap):
    img1, img2, flow, vmap = flip_left_right_no_edge(img1, img2, flow, vmap)
    img1, img2, flow, vmap = flip_top_down_no_edge(  img1, img2, flow, vmap)
    return img1, img2, flow, vmap

def random_flip_no_edge(img1, img2, flow, vmap):
    t = random.randint(0, 3)
    #print(t)
    if t==1:
        img1, img2, flow, vmap = flip_left_right_no_edge(img1, img2, flow, vmap)
    elif t==2:
        img1, img2, flow, vmap = flip_top_down_no_edge(img1, img2, flow, vmap)
    elif t==3:
        img1, img2, flow, vmap = rotate_180_no_edge(img1, img2, flow, vmap)

    return img1, img2, flow, vmap







def flip_left_right_fb(img1, img2, vmap_f, vmap_b):
    img1  = left_right(img1)
    img2  = left_right(img2)
    vmap_f  = left_right(vmap_f)
    vmap_b  = left_right(vmap_b)
    return img1, img2, vmap_f, vmap_b

def flip_top_down_fb(img1, img2, vmap_f, vmap_b):
    img1  = top_down(img1)
    img2  = top_down(img2)
    vmap_f  = top_down(vmap_f)
    vmap_b  = top_down(vmap_b)
    return img1, img2, vmap_f, vmap_b

def rotate_180_fb(img1, img2, vmap_f, vmap_b):
    img1, img2, vmap_f, vmap_b = flip_left_right_fb(img1, img2, vmap_f, vmap_b)
    img1, img2, vmap_f, vmap_b = flip_top_down_fb(  img1, img2, vmap_f, vmap_b)
    return img1, img2, vmap_f, vmap_b

def random_flip_fb(img1, img2, vmap_f, vmap_b):
    t = random.randint(0, 3)
    #print(t)
    if t==1:
        img1, img2, vmap_f, vmap_b = flip_left_right_fb(img1, img2, vmap_f, vmap_b)
    elif t==2:
        img1, img2, vmap_f, vmap_b = flip_top_down_fb(img1, img2, vmap_f, vmap_b)
    elif t==3:
        img1, img2, vmap_f, vmap_b = rotate_180_fb(img1, img2, vmap_f, vmap_b)

    return img1, img2, vmap_f, vmap_b

