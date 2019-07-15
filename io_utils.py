import numpy as np
import os
import struct

def load_flow_file(filename):
    TAG_FLOAT = 202021.25;  # check for this when READING the file

    # sanity check
    if filename == '':
        print ('readFlowFile: empty filename')
        return

    idx = filename.rfind('.')

    if idx == -1:
        print ('readFlowFile: extension required in filename %s' % filename)
        return


    if filename[idx:] != '.flo':
        print ('readFlowFile: filename %s should have extension ''.flo''' % filename)
        return

    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        # sanity check

        if tag != TAG_FLOAT :
            print ('readFlowFile(%(filename)s): wrong tag (possibly due to big-endian machine?)' % {"filename":filename});
            return

        if width < 1 or width > 99999:
            print ('readFlowFile(%s): illegal width %d' % (filename,width))
            return

        if height < 1 or height > 99999:
            print ('readFlowFile(%s): illegal height %d' % (filename, height))
            return

        nBands = 2


        # arrange into matrix form
        data = np.fromfile(f, np.float32, count=nBands*width*height)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (height, width, 2))

        return data2D

def save_flow_file(flow, filename):
    TAG_STRING = 'PIEH'
    # sanity check

    if filename == '':
        print ('writeFlowFile: empty filename')
        return

    idx = filename.rfind('.')

    if idx == -1:
        print ('writeFlowFile: extension required in filename %s' % filename)
        return


    if filename[idx:] != '.flo':
        print ('writeFlowFile: filename %s should have extension ''.flo''' % filename)
        return

    (height, width, nBands) = flow.shape

    if nBands != 2:
        print ('writeFlowFile: image must have two bands');
        return

    with open(filename, 'wb') as f:
        f.write(TAG_STRING)
        f.write(struct.pack('<i', width))
        f.write(struct.pack('<i', height))
        np.asarray(flow, np.float32).tofile(f)

def load_edges_file(edges_file_name, width, height):
    edges_img = np.ndarray((height,width),dtype=np.float32)
    with open(edges_file_name, 'rb') as f:
        f.readinto(edges_img)
    return edges_img

def load_matching_file(filename, width, height):
    img = np.zeros([height,width,2])
    mask = -np.ones([height,width])

    if os.path.getsize(filename) == 0:
        print ('empty file: %s' % filename)
    else:
        x1,y1,x2,y2 = np.loadtxt(filename, dtype=np.float32, delimiter=' ', unpack=True, usecols=(0,1,2,3))

        img[np.array(y1, dtype=int),np.array(x1, dtype=int),:] = np.stack((x2 - x1, y2 - y1), axis=1)
        mask[np.array(y1, dtype=int),np.array(x1, dtype=int)] = 1
        if np.any(np.isnan(img)):
            print ("Nan value found")

    return img, mask
