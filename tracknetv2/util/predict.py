import cv2

def find_pos(pred):
    contours, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in contours]

    if len(rects) == 0:
        return None, None

    # max_rect = max(rects, key=cv2.countourArea)
    max_rect = max(rects, key=lambda x: x[2] * x[3])

    cx, cy = int(max_rect[0] + max_rect[2]/2), int(max_rect[1] + max_rect[3]/2)
    return cx, cy
