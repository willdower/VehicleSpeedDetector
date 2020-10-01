import cv2
import numpy as np
import time
import sys


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


def get_box_real_area(corner1, corner2, camera_details):
    width = get_real_width(np.abs(corner1[0] - corner2[0]), camera_details)
    height = get_real_height(np.abs(corner1[1] - corner2[1]), camera_details)
    return height * width


def check_overlap(box1, box2):
    # Box format - ((corner1_x, corner1_y), (corner2_x, corner2_y))
    for x in range(box1[0][0], box1[1][0]):
        for y in range(box1[0][1], box1[1][1]):
            if box2[0][0] <= x <= box2[1][0] and box2[0][1] <= y < box2[1][1]:
                return True

    return False


def euclidean_distance(corner1, corner2):
    return np.sqrt(np.power((corner1[0] - corner2[0]), 2) + np.power((corner1[1] - corner2[1]), 2))


def check_out_bounds(box, img_width):
    box = np.reshape(box, (8, 1))
    for point in box:
        if point < 0 or point > img_width:
            return True
    return False


def get_real_width(pixel_width, camera_details):
    real_width = ((camera_details['object_distance'] * pixel_width * camera_details['sensor_width']) /
                  (camera_details['focal_length'] * camera_details['image_width']))
    return real_width


def get_real_height(pixel_height, camera_details):
    real_height = ((camera_details['object_distance'] * pixel_height * camera_details['sensor_height']) /
                  (camera_details['focal_length'] * camera_details['image_height']))
    return real_height


if __name__ == "__main__":
    file = open(str(sys.argv[1]), "r")

    camera_details = dict()
    video_filename = str(file.readline().strip('\n'))
    camera_details['focal_length'] = float(file.readline().strip('\n'))
    camera_details['object_distance'] = float(file.readline().strip('\n'))
    camera_details['sensor_width'] = float(file.readline().strip('\n'))
    camera_details['sensor_height'] = float(file.readline().strip('\n'))


    debug = False
    if len(sys.argv) == 8:
        if int(sys.argv[7]) == 1:
            debug = True

    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=150, detectShadows=False)

    capture = cv2.VideoCapture(video_filename)
    frame_rate = capture.get(cv2.CAP_PROP_FPS)

    clips = []
    frame_num = 0
    obj_index = 0
    current_obj_index = 0
    init = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        # resolutions - 360p, 480p, 720p, 1080p
        resolutions = ((640, 360), (852, 480), (1280, 720), (1920, 1080))
        current_resolution = resolutions[1]
        camera_details['image_width'] = current_resolution[0]
        camera_details['image_height'] = current_resolution[1]
        frame = cv2.resize(frame, current_resolution)

        gray = frame.copy()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        if init == 0:
            init = 1
            prev_frame = frame.copy()

        frame_num += 1

        clean = frame.copy()
        frame_to_show = clean.copy()
        match_show = clean.copy()

        frame = cv2.blur(frame, (5, 5))
        clean_blurred = clean.copy()
        clean_blurred = cv2.blur(clean_blurred, (5, 5))

        fgMask = backSub.apply(frame, learningRate=-1)
        fgMask = cv2.blur(fgMask, (5, 5))
        fgMask_clean = fgMask.copy()

        contours, heirarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        check_found = False
        box_img = clean.copy()
        if len(contours) > 0:
            prev_clips = clips
            box_list = []
            clips = []
            for cnt in contours:
                # Create a rectnagle around each contour
                rect = cv2.minAreaRect(cnt)  # corner1 corner2 other
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box_size = get_box_real_area(box[0], box[2], camera_details)

                # Check if box is large enough to be a vehicle and isn't partly out of frame
                box_size_threshold = 1500*2700  # Smallest car height (mm) * smallest car width (mm)
                if box_size > box_size_threshold and not check_out_bounds(box, frame.shape[1]):
                    # Take a clip of the box and use it as a template to match in the previous frame
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

                    # Search through the locations of all clips in the previous frame
                    # If the current clip is found in the previous frame with sufficient confidence, and it overlaps
                    # with a clip from the previous frame, they are the same object
                    check_found = False
                    for old_clip in prev_clips:
                        if check_overlap((top_left, bottom_right), old_clip[2]) and max_val > 0.865:
                            check_found = True
                            clips.append((clip, current_obj_index, clip_box))
                            current_frame_box = clip_box
                            prev_frame_box = old_clip[2]
                            current_obj_index = old_clip[1]
                            break

                    # Set the object index of the clip to the same as the previous one to indicate its the same
                    # Calculate distance between object in two frames and calculate speed
                    # Draw information to the display frame
                    if check_found:
                        # Write data to frame
                        if debug:
                            cv2.putText(clip_loc, str(current_obj_index), top_left, cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (255, 0, 0))
                        cv2.rectangle(frame_to_show, clip_box[0], clip_box[1], (0, 255, 0), 2)
                        distance_in_pixels = euclidean_distance(prev_frame_box[0], current_frame_box[0])
                        distance_in_mm = get_real_width(distance_in_pixels, camera_details)
                        speed_in_m_s = (distance_in_mm * frame_rate) / 1000
                        string = str(speed_in_m_s) + "m/s"
                        cv2.putText(frame_to_show, string, clip_box[1], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (255, 0, 0))

                        # Write data to console
                        if debug:
                            print("Match found, obj_index is " + str(current_obj_index) +
                                  ", distance change = " + str(distance_in_mm) + "mm"
                                  + ", speed = " + str(speed_in_m_s) + "m/s")

                    # If the clip was never in the previous frame, it's a new object so assign it a new object index
                    else:
                        clips.append((clip, obj_index, clip_box))
                        obj_index += 1

        # Show display frame and pause if there was a detection to allow viewing time
        prev_frame = clean
        cv2.imshow('Frame', frame_to_show)
        cv2.waitKey(30)
        if check_found:
            time.sleep(0.75)
