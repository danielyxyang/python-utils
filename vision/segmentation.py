import numpy as np


def compute_bbox_of_mask(mask):
    r_min, r_max = np.flatnonzero(np.any(mask, axis=1))[[0,-1]]
    c_min, c_max = np.flatnonzero(np.any(mask, axis=0))[[0,-1]]
    return r_min, c_min, r_max, c_max

def compute_bbox_union(bbox_A, bbox_B):
    return (
        min(bbox_A[0], bbox_B[0]),
        min(bbox_A[1], bbox_B[1]),
        max(bbox_A[2], bbox_B[2]),
        max(bbox_A[3], bbox_B[3]),
    )

def compute_bbox_intersection(bbox_A, bbox_B):
    return (
        max(bbox_A[0], bbox_B[0]),
        max(bbox_A[1], bbox_B[1]),
        min(bbox_A[2], bbox_B[2]),
        min(bbox_A[3], bbox_B[3]),
    )

def compute_bbox_iou(bbox_A, bbox_B, ret_more=False):
    # compute area
    def area(bbox):
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        else:
            return 0
    area_inner = area(compute_bbox_intersection(bbox_A, bbox_B))
    area_A = area(bbox_A)
    area_B = area(bbox_B)
    # compute IoU and related metrics
    iou = area_inner / (area_A + area_B - area_inner)
    if ret_more:
        ioA = area_inner / area_A
        ioB = area_inner / area_B
        return iou, ioA, ioB
    else:
        return iou

def compute_pixel_iou(mask_A, mask_B, ret_more=False):
    # compute area
    area_inner = np.count_nonzero(mask_A * mask_B)
    area_A = np.count_nonzero(mask_A)
    area_B = np.count_nonzero(mask_B)
    # compute IoU and related metrics
    iou = area_inner / (area_A + area_B - area_inner)
    if ret_more:
        ioA = area_inner / area_A
        ioB = area_inner / area_B
        return iou, ioA, ioB
    else:
        return iou
