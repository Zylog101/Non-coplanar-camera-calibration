import sys
import cv2
import numpy as np
import math
import random
from decimal import *

DEFAULT_IMAGE = 'chessboard.jpg'
FILE = 'selected_points'
WIN_1 = 'window1'
# 1 Feature point extraction by opencv
# 2 Manual selection of points
# 3 point file
SELECTION = 1
# worldPt.txt and imagePt.txt
IMAGE_POINT_VALUE_FILE = ''
WORLD_POINT_VALUE_FILE = ''

selected_points = []
chess_board = []
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# creating world point and image point correspondance from world and image point files
def generate_correspondence_file_from_point_file():
    with open(WORLD_POINT_VALUE_FILE, 'r') as world_fp, open(IMAGE_POINT_VALUE_FILE, 'r') as image_fp:
        temp_world_point_lines = world_fp.readlines()
        temp_image_point_lines = image_fp.readlines()

        world_point_lines = [x.strip() for x in temp_world_point_lines]
        image_point_lines = [x.strip() for x in temp_image_point_lines]

        fp = open('world_point_correspondence.txt', 'w')
        for world_line, image_line in zip(world_point_lines, image_point_lines):
            # fp.write(' '.join('{} {}'.format(world_line, image_line)))
            fp.write('{} {}\n'.format(world_line, image_line))
        fp.close()


# mouse call back
def mouse_event_callback(event, x, y, flags, param):
    global chess_board
    print("mouse_event_callback")
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(chess_board, (x, y), 2, (255, 0, 0), -1)
        selected_points.append((x, y))
        cv2.imshow(WIN_1, chess_board)


# def extract_feature_points(chess_board_arg):
#     print('extract_feature_point')
#     chess_board = cv2.cvtColor(chess_board_arg, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(chess_board, (7, 6))
#     if ret:
#         corners2 = cv2.cornerSubPix(chess_board, corners, (11, 11), (-1, -1), criteria)
#         chess_board_arg = cv2.drawChessboardCorners(chess_board_arg, (7, 6), corners2, ret)
#         cv2.imshow(WIN_1, chess_board_arg)
#         cv2.waitKey()
#     return corners2


def extract_feature_points(chess_board_arg):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    objp = objp * 100

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    chess_board = cv2.cvtColor(chess_board_arg, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(chess_board, (7, 6),None)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(chess_board, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        chess_board_arg = cv2.drawChessboardCorners(chess_board_arg, (7, 6), corners2, ret)
        cv2.imshow(WIN_1, chess_board_arg)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        obj_list = objpoints[0]
        img_list = imgpoints[0]

        with open('world_point_correspondence.txt', 'w') as world_fp:
            world_fp.write('{} {}\n'.format(len(objpoints[0]), len(imgpoints[0])))
            for obj_elem, img_elem in zip(obj_list, img_list):
                world_fp.write('{} {} {} {} {}\n'.format(obj_elem[0], obj_elem[1], obj_elem[2], img_elem[0, 0], img_elem[0, 1]))



def parse_command_line():
    global SELECTION, POINT_VALUE_FILE, WORLD_POINT_VALUE_FILE, IMAGE_POINT_VALUE_FILE
    commandline_args_length = len(sys.argv)
    SELECTION = int(sys.argv[1])
    if commandline_args_length == 4:
        WORLD_POINT_VALUE_FILE = sys.argv[2]
        IMAGE_POINT_VALUE_FILE = sys.argv[3]
        # WORLD_POINT_VALUE_FILE = 'worldPt.txt'
        # IMAGE_POINT_VALUE_FILE = 'imagePt.txt'
    elif commandline_args_length == 3:
        WORLD_POINT_VALUE_FILE = sys.argv[2]
    return


def initialize_image():
    cv2.namedWindow(WIN_1)
    if SELECTION == 3:
        # read the point file
        pass
    chess_board_img = cv2.imread(DEFAULT_IMAGE)
    # chess_board = cv2.resize(chess_board, SIZE)
    return chess_board_img


def camera_calibrate(v):

    stacked_M = v[:, 11]
    M1 = stacked_M[:4, 0]
    M2 = stacked_M[4:8, 0]
    M3 = stacked_M[8:12, 0]
    m1 = M1.transpose()
    m2 = M2.transpose()
    m3 = M3.transpose()
    M = np.vstack((m1, m2, m3))
    a1_t = M[0, :3]
    a1 = a1_t.transpose()
    a2_t = M[1, :3]
    a2 = a2_t.transpose()
    a3_t = M[2, :3]
    a3 = a3_t.transpose()
    b = M[:, 3]
    magnitude_of_a3 = np.linalg.norm(a3)
    # magnitude of row calculation
    magnitude_of_row = 1 / magnitude_of_a3
    square = magnitude_of_row ** 2
    square_a1 = square * a1
    # u_zero calculation
    u_zero = np.tensordot(square_a1, a3)
    # v_zero calculation
    square_a2 = square * a2
    v_zero = np.tensordot(square_a2, a3)
    # alpha v calculation
    square_a2_dot_a2 = float(np.tensordot(square_a2, a2))
    square_v0 = v_zero ** 2
    getcontext().prec = 28
    # a = Decimal(square_a2_dot_a2).quantize(Decimal('0.01'), rounding=ROUND_UP)
    # x = Decimal(square_v0).quantize(Decimal('0.01'), rounding=ROUND_UP)
    a = Decimal(square_a2_dot_a2)
    x = Decimal(square_v0)
    temp = a - x
    try:
        alpha_v = math.sqrt(temp)
    except:
        exit()
        print("failed")
    # s calculation
    quadriple_row = magnitude_of_row ** 4
    temp1 = quadriple_row / alpha_v
    a1_cross_a3 = np.cross(a1, a3, axis=0)
    temp2 = temp1 * a1_cross_a3
    a2_cross_a3 = np.cross(a2, a3, axis=0)
    s = np.tensordot(temp2, a2_cross_a3)
    # alpha u calculation
    square_a1_dot_a1 = np.tensordot(square_a1, a1)
    square_u0 = u_zero ** 2
    temp = square_a1_dot_a1 - (s ** 2) - square_u0
    alpha_u = math.sqrt(temp)
    # k*
    k_star = np.matrix([[alpha_u, s, u_zero], [0, alpha_v, v_zero], [0, 0, 1]])
    # epsilon
    eps = -1
    # T*
    k_star_inverse = np.linalg.inv(k_star)
    T_star = eps * magnitude_of_row * k_star_inverse * b
    # r3
    r3 = eps * magnitude_of_row * a3
    r3_t = r3.transpose()
    r1 = (square / alpha_v) * np.cross(a2, a3, axis=0)
    r1_t = r1.transpose()
    r2 = np.cross(r3, r1, axis=0)
    r2_t = r2.transpose()
    # R*
    r_star = np.vstack((r1_t, r2_t, r3_t))
    projection_matrix = k_star * np.hstack((r_star, T_star))
    fp = open("camera_parameters.txt", 'w')
    fp.write("alphaU = {}\nalphaV = {}\ns = {}\nu0 = {}\nv0 = {}\nR* = {}\nT* = {}\nunknown scale {}\n".format( alpha_u, alpha_v, s, u_zero, v_zero, r_star, T_star, eps*magnitude_of_row))
    fp.close()
    return projection_matrix


def draw_point(n, world_point_list, image_point_list):
    i = 0
    rand_array = random.sample(range(0,len(world_point_list)-2), n)
    #test
    rand_array.sort()
    rand_world_point_list = []
    remaining_world_point_list = world_point_list.copy()
    del remaining_world_point_list[len(world_point_list)-1]
    rand_image_point_list = []
    remaining_image_point_list = image_point_list.copy()
    del remaining_image_point_list[len(world_point_list)-1]

    while i < n:
        rand = rand_array[i]
        rand_image_point_list.append(image_point_list[rand])
        rand_world_point_list.append(world_point_list[rand])
        try:
            remaining_image_point_list.remove(image_point_list[rand])
            remaining_world_point_list.remove(world_point_list[rand])
        except:
            continue
        i += 1
    return rand_world_point_list,rand_image_point_list,remaining_world_point_list,remaining_image_point_list


# todo how to deal with the camera parameters should I return or print
def calibrate_camera_parameters(world_point_list, image_point_list):
    zero_point_matrix = np.matrix([0, 0, 0, 0])
    wolll = []
    trail = []
    length = len(world_point_list)
    i = 0

    while i < length:

        world_point, image_point = world_point_list[i], image_point_list[i]
        # world_point.append(1.0)
        # image_point.append(1.0)

        world_point_list.append(world_point)
        image_point_list.append(image_point)

        world_point_matrix = np.matrix(world_point)
        image_point_matrix = np.matrix(image_point)

        tempx_point_matrix = image_point_matrix.item(0) * (-1) * world_point_matrix
        try:
            tempy_point_matrix = image_point_matrix.item(1) * (-1) * world_point_matrix
        except:
            i += 1
            continue

        merged = []

        # creating first line
        first_line = world_point_matrix.tolist()
        for element in first_line:
            merged.append(element)

        first_line = zero_point_matrix.tolist()
        for element in first_line:
            merged.append(element)

        first_line = tempx_point_matrix.tolist()
        for element in first_line:
            merged.append(element)

        temp = np.asarray(merged)
        temp2 = temp.flatten()
        # creating second line
        merged2 = []

        first_line = zero_point_matrix.tolist()
        for element in first_line:
            merged2.append(element)

        first_line = world_point_matrix.tolist()
        for element in first_line:
            merged2.append(element)

        first_line = tempy_point_matrix.tolist()
        for element in first_line:
            merged2.append(element)

        temp1 = np.asarray(merged2)
        temp3 = temp1.flatten()
        wolll.append(temp2)
        wolll.append(temp3)
        i += 1
    trial = np.matrix(wolll)

    svd_result = np.linalg.svd(trial)
    t = svd_result[2]
    transpose_v = t.transpose()
    projection_matrix = camera_calibrate(transpose_v)
    return projection_matrix


def compute_inlier_list(remaining_world_point_list, remaining_image_point_list, t, d, projection_matrix):
    number_of_inliers = 0
    inlier_world_point_list = []
    inlier_image_point_list = []

    M1 = projection_matrix[0, :4]
    M2 = projection_matrix[1, :4]
    M3 = projection_matrix[2, :4]

    length = len(remaining_world_point_list)
    i = 0

    while i < length:
        world_point, image_point = remaining_world_point_list[i], remaining_image_point_list[i]

        world_point_matrix = np.matrix(world_point).transpose()
        image_point_matrix = np.matrix(image_point)
        try:
            computed = projection_matrix * world_point_matrix
        except:
            i += 1
            continue

        x_num = M1 * world_point_matrix
        x_den = M3 * world_point_matrix
        y_num = M2 * world_point_matrix
        y_den = x_den
        x = image_point_matrix[0, 0]
        y = image_point_matrix[0, 1]
        x_diff_square = (x - (x_num / x_den)) ** 2
        y_diff_square = (y - (y_num / y_den)) ** 2

        sum_err = x_diff_square + y_diff_square
        if sum_err < t:
            inlier_world_point_list.append(world_point)
            inlier_image_point_list.append(image_point)
            number_of_inliers += 1
            # if number_of_inliers > d:
            #     projection_matrix = calibrate_camera_parameters(inlier_world_point_list,inlier_image_point_list)
            #     # todo have to handle termination
            #     terminate()
        i += 1

    return number_of_inliers, inlier_world_point_list, inlier_image_point_list


def compute_ransac_algo(median_err, world_point_list, image_point_list):
    fp = open('Ransac.config','r')
    string_list = fp.readline().split()
    fp.close()
    float_point_list = [float(i) for i in string_list]
    d = int(float_point_list[0])
    n = int(float_point_list[1])
    p = float_point_list[2]
    k_max = int(float_point_list[3])
    inlier_value_per_iteration = []

    # d = 6
    # n = 7
    k = 0
    t = 1.5 * median_err

    # p = 0.99
    w = 0.5
    k = math.log10(1 - p) / math.log10(1 - (w ** n))
    # k_max = 100
    i = 0
    flag = 1
    while i < k and i < k_max:
        rand_world_point_list, rand_image_point_list, remaining_world_point_list, remaining_image_point_list = draw_point(
            n, world_point_list, image_point_list)
        # construct M (camera calibration)
        try:
            projection_matrix = calibrate_camera_parameters(rand_world_point_list, rand_image_point_list)
        except:
            flag = terminate()
            break

        # travers point list and find E if less than T then add to inlier_list
        number_of_inliers, inlier_world_point_list, inlier_image_point_list = compute_inlier_list(
            remaining_world_point_list, remaining_image_point_list, t, d, projection_matrix)

        # if number of inlier_list > d then recompute M(camera calibration) and terminate
        if number_of_inliers > d:
            inlier_value_per_iteration.append((number_of_inliers, inlier_world_point_list, inlier_image_point_list, k))
            calibrate_camera_parameters(inlier_world_point_list, inlier_image_point_list)

            # todo have to handle termination
            flag = terminate()
            # break
        # elif number_of_inliers > 0:
            # w = number_of_inliers / len(remaining_world_point_list)
        if number_of_inliers > 0:
            w = number_of_inliers / len(world_point_list)
            w_power = w ** n
            den_log = np.log(1 - w_power)
            num_log = math.log(1 - p)
            if den_log != 0.0:
                k = num_log / den_log
        i += 1
    if flag == 1:
        fp = open("camera_parameters.txt", 'a')
        fp.write("Ransac failed to calibrate\n")
        fp.close()
        return
    inlier_value_per_iteration.sort(reverse=True)
    calibrate_camera_parameters(inlier_value_per_iteration[0][1], inlier_value_per_iteration[0][2])
    fp = open("camera_parameters.txt", 'a')
    fp.write("n = {}\nw = {}\nk = {}\n".format(n, w, k))
    fp.close()

def compute_initial_calibration():
    fp = open("world_point_correspondence.txt", 'r')
    line = fp.readline()
    number_of_lines = line.split()
    zero_point_matrix = np.matrix([0, 0, 0, 0])
    wolll = []
    trail = []
    world_point_list = []
    image_point_list = []

    while line != '':
        line = fp.readline()
        string_point_list = line.split()

        float_point_list = [float(i) for i in string_point_list]

        world_point, image_point = float_point_list[:3], float_point_list[3:]
        world_point.append(1.0)
        image_point.append(1.0)

        world_point_list.append(world_point)
        image_point_list.append(image_point)

        world_point_matrix = np.matrix(world_point)
        image_point_matrix = np.matrix(image_point)

        tempx_point_matrix = image_point_matrix.item(0) * (-1) * world_point_matrix
        try:
            tempy_point_matrix = image_point_matrix.item(1) * (-1) * world_point_matrix
        except:
            continue

        merged = []

        # creating first line
        first_line = world_point_matrix.tolist()
        for element in first_line:
            merged.append(element)

        first_line = zero_point_matrix.tolist()
        for element in first_line:
            merged.append(element)

        first_line = tempx_point_matrix.tolist()
        for element in first_line:
            merged.append(element)

        temp = np.asarray(merged)
        temp2 = temp.flatten()
        # creating second line
        merged2 = []

        first_line = zero_point_matrix.tolist()
        for element in first_line:
            merged2.append(element)

        first_line = world_point_matrix.tolist()
        for element in first_line:
            merged2.append(element)

        first_line = tempy_point_matrix.tolist()
        for element in first_line:
            merged2.append(element)

        temp1 = np.asarray(merged2)
        temp3 = temp1.flatten()
        wolll.append(temp2)
        wolll.append(temp3)
    trial = np.matrix(wolll)

    svd_result = np.linalg.svd(trial)
    t = svd_result[2]
    transpose_v = t.transpose()

    projection_matrix = camera_calibrate(transpose_v)
    fp.close()

    M1 = projection_matrix[0, :4]
    M2 = projection_matrix[1, :4]
    M3 = projection_matrix[2, :4]
    line = 'aa'
    fp = open('world_point_correspondence.txt', 'r')
    line = fp.readline()
    distance = 0
    err = []
    # initial computation of camera calibration using all the points
    while line != '':
        line = fp.readline()
        string_point_list = line.split()

        float_point_list = [float(i) for i in string_point_list]

        world_point, image_point = float_point_list[:3], float_point_list[3:]
        world_point.append(1.0)
        image_point.append(1.0)

        world_point_matrix = np.matrix(world_point).transpose()
        image_point_matrix = np.matrix(image_point)
        try:
            computed = projection_matrix * world_point_matrix
        except:
            continue

        x_num = M1 * world_point_matrix
        x_den = M3 * world_point_matrix
        y_num = M2 * world_point_matrix
        y_den = x_den
        x = image_point_matrix[0, 0]
        y = image_point_matrix[0, 1]
        x_diff_square = (x - (x_num / x_den)) ** 2
        y_diff_square = (y - (y_num / y_den)) ** 2

        sum_err = x_diff_square + y_diff_square
        err.append(sum_err)
        distance += sum_err
    err_val = distance
    median_err = np.median(err)
    return world_point_list, image_point_list, median_err


def terminate():
    flag = 0
    return flag


def main():
    global chess_board, IMAGE_POINT_VALUE_FILE
    parse_command_line()
    print (SELECTION)
    chess_board = initialize_image()
    if SELECTION == 3:
        # work on the point list files
        generate_correspondence_file_from_point_file()
    elif SELECTION == 2:
        # manually enter points
        cv2.setMouseCallback(WIN_1, mouse_event_callback)
        cv2.imshow(WIN_1, chess_board)
        cv2.waitKey()
        print(selected_points)
        IMAGE_POINT_VALUE_FILE = 'imagePt_manual.txt'
        with open(IMAGE_POINT_VALUE_FILE, 'w') as image_fp:
            image_fp.write(str(len(selected_points))+'\n')
            for point in selected_points:
                image_fp.write((str(point[0]))+' '+(str(point[1]))+'\n')
        generate_correspondence_file_from_point_file()

    elif SELECTION == 1:
        # opencv feature extraction
        extract_feature_points(chess_board)

    world_point_list, image_point_list, median_err = compute_initial_calibration()
    compute_ransac_algo(median_err, world_point_list, image_point_list)


if __name__ == '__main__':
    main()

