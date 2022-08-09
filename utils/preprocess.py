import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.transform
import skimage.filters


# for generating ground truth heatmaps
def gen_hmaps(img, pts, sigma_valu=2):
    '''
    Generate 16 heatmaps
    :param img: np arrray img, (H,W,C)
    :param pts: joint points coords, np array, same resolu as img
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: vaalue for gaussian blur
    :return: np array heatmaps, (H,W,num_pts)
    '''
    H, W = img.shape[0], img.shape[1]
    num_pts = len(pts)
    heatmaps = np.zeros((H, W, num_pts + 1))
    for i, pt in enumerate(pts):
        # Filter unavailable heatmaps
        if pt[0] == 0 and pt[1] == 0: continue

        # Filter some points out of the image
        if pt[0] >= W: pt[0] = W-1
        if pt[1] >= H: pt[1] = H-1
        heatmap = heatmaps[:, :, i]
        heatmap[int(pt[1])][int(pt[0])] = 1  # reverse sequence
        heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)  ##(H,W,1) -> (H,W)
        am = np.max(heatmap)
        heatmap = heatmap / am  # scale to [0,1]
        heatmaps[:, :, i] = heatmap

    heatmaps[:, :, num_pts] = 1.0 - np.max(heatmaps[:, :, :num_pts], axis=2) # add background dim
    
    return heatmaps


def crop(img, ele_anno, crop_size=256):

    # get bbox
    pts = ele_anno['landmarks']

    bbox = ele_anno['bbox']
    vis = np.array(ele_anno['visibility'])
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()

    xs = np.array(pts[0::2]).T
    ys = np.array(pts[1::2]).T
    # print("ptsx", xs)
    # print("ptsy", ys)

    cen = np.array((1,2))
    cen[0] = int(bbox[0] + bbox[2]/2)
    cen[1] = int(bbox[1] + bbox[3]/2)


    H,W = img.shape[0], img.shape[1]

    # topleft:x1,y1  bottomright:x2,y2
    bb_x1 = int(bbox[0])
    bb_y1 = int(bbox[1])
    bb_x2 = int(bbox[0] + bbox[2])
    bb_y2 = int(bbox[1] + bbox[3])

    newX = bb_x2-bb_x1
    newY = bb_y2-bb_y1
    
    if(newX>newY):
        dif = newX-newY
        bb_y1-=int(dif/2)
        bb_y2+=int(dif/2)
    else:
        dif=newY-newX
        bb_x1-=int(dif/2)
        bb_x2+=int(dif/2)

    if bb_x1<0 or bb_x2>W or bb_y1<0 or bb_y2>H:
        pad = int(max(-bb_x1, bb_x2-W, -bb_y1, bb_y2-H))
        img = np.pad(img, ((pad, pad),(pad,pad),(0,0)), 'constant')
    else:
        pad = 0
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show() 
    img = img[bb_y1+pad:bb_y2+pad, bb_x1+pad:bb_x2+pad]

    xs = np.where(xs != 0, xs-bb_x1, xs)
    ys = np.where(ys != 0, ys-bb_y1, ys)
    #ys = np.array([ys[i]-bb_y1 for i in range(len(ys)) if i in pts_nonzero])
    bbox[0] -= bb_x1
    bbox[1] -= bb_y1
    

    cen[0] = int((bb_x2-bb_x1)/2)
    cen[1] = int((bb_y2-bb_y1)/2)


    # resize
    H,W = img.shape[0], img.shape[1]
    xs = xs*crop_size/W
    ys = ys*crop_size/H
    cen[0] = cen[0]*crop_size/W
    cen[1] = cen[1]*crop_size/H
    
    # print("scale", scale)
    # print("bbox", bbox)

    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()
    img = cv2.resize(img, (crop_size, crop_size))
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()

    pts = [[xs[i], ys[i]] for i in range(len(xs))]


    return img, pts, cen