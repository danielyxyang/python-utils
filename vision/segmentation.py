import numpy as np

def is_bbox_valid(bbox):
    return bbox is not None and len(bbox) == 4 and bbox[0] <= bbox[2] and bbox[1] <= bbox[3]

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
    bbox = (
        max(bbox_A[0], bbox_B[0]),
        max(bbox_A[1], bbox_B[1]),
        min(bbox_A[2], bbox_B[2]),
        min(bbox_A[3], bbox_B[3]),
    )
    return bbox if is_bbox_valid(bbox) else None

def compute_bbox_iou(bbox_A, bbox_B, ret_more=False):
    # compute area
    def area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if is_bbox_valid(bbox) else 0
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

def compute_pixel_iou2(mask_and_bbox_A, mask_and_bbox_B, ret_more=False):
    mask_A, bbox_A = mask_and_bbox_A
    mask_B, bbox_B = mask_and_bbox_B
    if compute_bbox_intersection(bbox_A, bbox_B) is None:
        # short-cut if intersection is empty
        return (0, 0, 0) if ret_more else 0
    else:
        # pad masks to same bbox and compute IoU
        def pad_to_union(mask, bbox, bbox_union):
            pad_row = bbox[0] - bbox_union[0], bbox_union[2] - bbox[2]
            pad_col = bbox[1] - bbox_union[1], bbox_union[3] - bbox[3]
            return np.pad(mask, (pad_row, pad_col))
        bbox_union = compute_bbox_union(bbox_A, bbox_B)
        return compute_pixel_iou(
            pad_to_union(mask_A, bbox_A, bbox_union),
            pad_to_union(mask_B, bbox_B, bbox_union),
            ret_more=ret_more,
        )
