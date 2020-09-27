import cv2
import numpy as np
import time


def get_clip(box_to_clip, base):
    corner1_x = box_to_clip[1][0]
    if corner1_x < 0:
        corner1_x = 0
    elif corner1_x > base.shape[1]:
        corner1_x = base.shape[1]

    corner2_x = box_to_clip[3][0]
    if corner2_x < 0:
        corner2_x = 0
    elif corner2_x > base.shape[1]:
        corner2_x = base.shape[1]

    corner1_y = box_to_clip[2][1]
    if corner1_y < 0:
        corner1_y = 0
    elif corner1_y > base.shape[0]:
        corner1_y = base.shape[0]

    corner2_y = box_to_clip[0][1]
    if corner2_y < 0:
        corner2_y = 0
    elif corner2_y > base.shape[0]:
        corner2_y = base.shape[0]

    corner1 = (corner1_x, corner1_y)
    corner2 = (corner2_x, corner2_y)

    clipped = base[corner1[1]:corner2[1], corner1[0]:corner2[0]]
    clip_box = (corner1, corner2)

    return clip_box, clipped


def get_box_area(corner1, corner2):
    length = np.abs(corner1[0] - corner2[0])
    width = np.abs(corner1[1] - corner2[1])
    return length * width


def check_overlap(box1, box2):
    # Box format - ((corner1_x, corner1_y), (corner2_x, corner2_y))
    for x in range(box1[0][0], box1[1][0]):
        for y in range(box1[0][1], box1[1][1]):
            if box2[0][0] <= x <= box2[1][0] and box2[0][1] <= y < box2[1][1]:
                return True

    return False


def euclidean_distance(corner1, corner2):
    return np.sqrt(np.power((corner1[0] - corner2[0]), 2) + np.power((corner1[1] - corner2[1]), 2))


template_width_mm = 85

backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=150, detectShadows=False)
surf = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

capture = cv2.VideoCapture("IMG_2909.MOV")
frame_rate = capture.get(cv2.CAP_PROP_FPS)

box_list = []
clips = []
frame_num = 0
obj_index = 0
current_obj_index = 0
init = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    frame = cv2.resize(frame, (852, 480))

    gray = frame.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if init == 0:
        init = 1
        prev_frame = frame.copy()

        template = cv2.imread("card_template.jpg")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]

        pixels_per_mm = w / template_width_mm

        res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        reference_loc = gray.copy()
        cv2.rectangle(reference_loc, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imshow('Reference', reference_loc)
        cv2.waitKey(30)


    frame_num += 1

    clean = frame.copy()
    frame_to_show = clean.copy()
    match_show = clean.copy()

    frame = cv2.blur(frame, (5, 5))
    clean_blurred = clean.copy()
    clean_blurred = cv2.blur(clean_blurred, (5, 5))

    fgMask = backSub.apply(frame, learningRate=-1)
    fgMask_clean = fgMask.copy()

    # fgMask = cv2.blur(fgMask, (5, 5))

    contours, heirarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    check_found = False
    box_img = clean.copy()
    if len(contours) > 0:
        prev_box_list = box_list
        prev_clips = clips
        box_list = []
        clips = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)  # corner1 corner2 other
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box_size = get_box_area(box[0], box[2])
            if box_size > 3000:
                clip_box, clip = get_clip(box, clean)

                template = cv2.cvtColor(clip, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                w, h = template.shape[::-1]
                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                clip_loc = prev_frame.copy()
                cv2.rectangle(clip_loc, top_left, bottom_right, (0, 255, 0), 2)
                prev_frame_box = ()
                current_frame_box = ()

                check_found = False
                for old_clip in prev_clips:
                    if check_overlap((top_left, bottom_right), old_clip[2]) and max_val > 0.865:
                        check_found = True

                        current_frame_box = clip_box
                        prev_frame_box = old_clip[2]

                        current_obj_index = old_clip[1]
                        cv2.imshow('Matching clip', old_clip[0])
                        cv2.imshow('Current clip', clip)
                        cv2.putText(clip_loc, str(current_obj_index), top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
                        cv2.imshow('Clip Found in Prev Frame', clip_loc)
                        cv2.waitKey(30)
                        clips.append((clip, current_obj_index, clip_box))

                        cv2.rectangle(frame_to_show, clip_box[0], clip_box[1], (0, 255, 0), 2)
                        cv2.putText(frame_to_show, str(current_obj_index), clip_box[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

                        break

                if check_found:
                    distance_in_pixels = euclidean_distance(prev_frame_box[0], current_frame_box[0])
                    distance_in_mm = distance_in_pixels / pixels_per_mm
                    speed_in_m_s = (distance_in_mm * frame_rate) / 1000
                    cv2.putText(frame_to_show, str(speed_in_m_s), clip_box[1], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
                    print("Match found, obj_index is " + str(current_obj_index) +
                          ", distance change = " + str(distance_in_mm) + "mm"
                          + ", speed = " + str(speed_in_m_s) + "m/s")

                else:
                    clips.append((clip, obj_index, clip_box))
                    obj_index += 1

    prev_frame = clean
    cv2.imshow('Frame', frame_to_show)
    cv2.waitKey(30)
    if check_found:
        time.sleep(1)
