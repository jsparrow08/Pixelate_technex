import gym
import pix_main_arena
import time
import pybullet as p
import pybullet_data
import os
import numpy as np
import cv2.aruco as aruco
import cv2
import math
import statistics as stat

if __name__ == "__main__":
    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    env = gym.make("pix_main_arena-v0")
    time.sleep(2)
    env.remove_car()
    l = env.camera_feed()
    env.respawn_car()

    lhsv = cv2.cvtColor(l, cv2.COLOR_BGR2HSV)
    # print(l.shape)
    # black_img = np.zeros(l.shape)
    # # Constant parameters used in Aruco methods
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    """codes # white=0,green=1,yellow=2,red=3,purple=4,blue=5
    direction---
    up-1
    down-2
    right-3
    left-4

    """

    def getrange(i):
        # here i = colourcode
        # white=0,green=1,yellow=2,red=3,purple=4,lightblue(hospital)=5,darkblue=6
        if(i == 0):
            rslt = [np.array([0, 0, 200]), np.array([180, 30, 255])]

        elif(i == 1):
            rslt = [np.array([36, 50, 70]), np.array([89, 255, 255])]
        elif(i == 2):
            rslt = [np.array([25, 50, 70]), np.array([35, 255, 255])]
        elif(i == 3):
            rslt = [np.array([0, 50, 70]), np.array([9, 255, 255]),
                    np.array([159, 50, 70]), np.array([180, 255, 255])]
        elif (i == 4):
            rslt = [np.array([140, 100, 100]), np.array([155, 255, 255])]
        elif (i == 5):
            rslt = [np.array([85, 30, 30]), np.array([100, 255, 255])]
        elif (i == 6):
            rslt = [np.array([100, 80, 80]), np.array([130, 255, 255])]

        else:
            rslt = 0
        return rslt

    def contfind(colour, lhsv, purple=False):
        rval = getrange(colour)
        if(len(rval) == 4):
            lower = np.array(rval[0], np.uint8)
            upper = np.array(rval[1], np.uint8)
            mask1 = cv2.inRange(lhsv, lower, upper)

            lower = np.array(rval[2], np.uint8)
            upper = np.array(rval[3], np.uint8)
            mask2 = cv2.inRange(lhsv, lower, upper)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = np.array(rval[0], np.uint8)
            upper = np.array(rval[1], np.uint8)
            mask = cv2.inRange(lhsv, lower, upper)

        contours, her = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        c = []
        if purple:
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > 200:
                    c.append(contours[i])
        else:
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > 500 and area < 6000:
                    c.append(contours[i])

        apr = []
        for i in range(len(c)):
            epsilon = 0.03 * cv2.arcLength(c[i], True)
            approx = cv2.approxPolyDP(c[i], epsilon, True)
            apr.append(approx)
        return apr

    def findcenter(cont):
        M = cv2.moments(cont)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center

    def findshape(approx):
        # perimeter = cv2.arcLength(approx, True)
        # area = cv2.contourArea(approx)
        # temp = area / (perimeter * perimeter)
        # if (temp > 0.04 and temp < 0.055):
        #     shape = "triangle"
        # elif (temp >= 0.055 and temp < 0.069):
        #     shape = "square"
        # elif (temp >= 0.069 and temp < 0.09):
        #     shape = "circle"

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square"
        else:
            shape = "circle"
        return shape

    def gettrivector(tricount):#up-1 down-2 right-3 left-4
        count = tricount.cont
        center = tricount.center
        cont = np.squeeze(count)
        p = 0
        direction = 0
        for i in cont:

            if abs(center[0] - i[0]) <= 3:
                p = 1  # 1 means up down
                if (i[1] - center[1]) > 0:
                    direction = 2  # down
                else:
                    direction = 1  # up
                break
            if abs(center[1] - i[1]) <= 3:
                p = 2  # 1 means rightleft
                if (i[0] - center[0]) > 0:
                    direction = 3  # right
                else:
                    direction = 4  #
                break

        if p == 0:
            direction = "none"

        return direction

    class contdetail:
        def __init__(self, cont, colorcode):
            self.cont = cont
            self.color = colorcode
            self.shape = findshape(cont)
            self.center = findcenter(cont)
            self.area = cv2.contourArea(cont)
            self.postion = [-1, -1]

    contours = []
    for i in range(6):
        # finding and drawing contour and saving them in count
        cnt = contfind(i, lhsv)
        # cv2.drawContours(black_img, cnt, -1, (0, 255, 0), 1)
        for j in cnt:
            contours.append(contdetail(j, i))

    # finding max and min value of the arena coordinates
    minx = 1000000
    miny = 1000000
    maxx = -1
    maxy = -1
    for i in range(len(contours)):  # geting max and min x,y coordinates
        # if contours[i].shape=="triangle":
        #     break
        r = contours[i].center
        if r[0] < minx:
            minx = r[0]
        if r[1] < miny:
            miny = r[1]
        if r[0] > maxx:
            maxx = r[0]
        if r[1] > maxy:
            maxy = r[1]

    # print(minx, miny, maxx, maxy)
    arenax = maxx - minx
    arenay = maxy - miny
    nodelen = [arenax / 11, arenay / 11]

    purple = []
    def findmatrix(contours, dim):

        finalmatrix = np.zeros((dim, dim), dtype=int)
        nodelen = [arenax / (dim - 1), arenay / (dim - 1)]
        # errors to ca9librate
        errorx = 0.029 * arenax
        errory = 0.029 * arenay
        st = 1
        t = 0
        for i in contours:

            if i.color in range(6):
                removingextra = 0

                for p in range(dim):

                    if i.center[0] >= minx - errorx + p * nodelen[0] and i.center[0] <= minx + errorx + p * nodelen[0]:
                        x = p
                        break
                    if p == dim - 1:
                        print("cant find x")
                        removingextra = 1

                for q in range(dim):

                    if i.center[1] >= miny - errory + q * nodelen[1] and i.center[1] <= miny + errory + q * nodelen[1]:
                        y = q
                        break
                    if q == dim - 1:
                        print("cant find y")
                        removingextra = 1

                if removingextra == 1:
                    break

                if i.shape == "square":
                    if i.color == 5:
                        if i.shape == "square":
                            finalmatrix[y][x] = 106
                    elif i.color == 4:
                        finalmatrix[y][x] = 105
                        purple.append(t)
                    else:
                        finalmatrix[y][x] = (i.color + 1) + finalmatrix[y][x]
                    i.postion = [y, x]
                if i.shape == "triangle":
                    # print("directions ", str(gettrivector(i)),str(i.color))
                    finalmatrix[y][x] = (10 * gettrivector(i)) + finalmatrix[y][x]
                if i.shape == "circle" and i.color == 5:
                    hos111 = [y, x]
                    st = 0
            t += 1

        if st == 0:
            finalmatrix[hos111[0], hos111[1]] = 116
        finalmatrix[dim - 1, dim - 1] = 102

        return finalmatrix



    def findpath(weight_matrix, startpoint, end_point):
        startpoint = np.array(startpoint)
        dim = list(weight_matrix.shape)
        visited_matrix = np.zeros(dim, dtype=int)
        visited_matrix[startpoint[0], startpoint[1]] = 1
        distancematrix = np.zeros(dim)
        dim.append(2)
        parent_matrix = np.zeros(dim, dtype=int)
        distancematrix = distancematrix + float('inf')
        distancematrix[startpoint[0], startpoint[1]] = 0
        list_of_points = [startpoint]

        r=-10
        # def snake(list_of_points, weight_matrix, distancematrix, parent_matrix, visited_matrix):
        while True:


            row = len(weight_matrix)
            col = len(weight_matrix[0])
            p = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            next_points = []
            for i in list_of_points:
                # oneways outgoing
                if weight_matrix[i[0], i[1]] // 10 == 1:
                    q=np.array([[-1, 0]])
                elif weight_matrix[i[0],i[1]]//10 == 2 :
                    q=np.array([[1, 0]])
                elif weight_matrix[i[0],i[1]]//10 == 3 :
                    q=np.array([[0, 1]])
                elif weight_matrix[i[0],i[1]]//10 == 4 :
                    q=np.array([[0,-1]])

                #corners and sides

                elif i[0] == 0 and i[1] == 0:
                    q = np.array([[1, 0],  [0, 1]])
                elif i[0] == row - 1 and i[1] == col - 1:
                    q = np.array([[-1, 0], [0, -1]])
                elif i[0] == 0 and i[1] == col - 1:
                    q = np.array([[1, 0], [0, -1]])
                elif i[0] == row - 1 and i[1] == 0:
                    q = np.array([[-1, 0], [0, 1]])
                elif i[0]==0 :
                    q=np.array([[1, 0], [0, -1], [0, 1]])
                elif i[0]==(row-1) :
                    q=np.array([[-1, 0], [0, -1], [0, 1]])
                elif i[1]==0 :
                    q=np.array([[-1, 0], [1, 0], [0, 1]])
                elif i[1]==(col-1) :
                    q=np.array([[-1, 0], [1, 0], [0, -1]])
                else:
                    q = p
                for point in q: #new points
                    check = 0
                    newpoint=i+point

                    if newpoint[0] ==12 or newpoint[1] ==12:
                        check=1

                    #cheking oneway incomings
                    if point[0]==1 and weight_matrix[newpoint[0],newpoint[1]]//10 ==1:
                        check=1
                    if point[0]==-1 and weight_matrix[newpoint[0],newpoint[1]]//10 ==2:
                        check=1
                    if point[1] == 1 and weight_matrix[newpoint[0],newpoint[1]]//10 == 4:
                        check=1
                    if point[1] == -1 and weight_matrix[newpoint[0],newpoint[1]]//10 == 3:
                        check=1

                    ab = weight_matrix[newpoint[0], newpoint[1]] % 10
                    if ab != 0 and check == 0 :

                        if (weight_matrix[i[0],i[1]]%10) + distancematrix[i[0],i[1]] < distancematrix[newpoint[0],newpoint[1]]:
                            distancematrix[newpoint[0],newpoint[1]] = (weight_matrix[i[0],i[1]]%10) + distancematrix[i[0],i[1]]
                            visited_matrix[newpoint[0],newpoint[1]] = 1
                            parent_matrix[newpoint[0],newpoint[1]] = i
                            if  ab != 6 and ab!=5:
                                next_points.append(newpoint)
                            # print(distancematrix)
#........................................................................................
            # print(visited_mat)
            # print(np.sum(visited_mat))
            # print(distancematrix)
            summ = np.sum(visited_matrix)
            res = np.count_nonzero(weight_matrix)
            # print(summ,r)
            list_of_points = next_points
            if summ == r:
                break

            if summ == res:
                break
            r = summ

###########################################################
            # if summ != res  :
            #     snake(next_points, weight_matrix, distancematrix, parent_matrix, visited_matrix)


            # return distancematrix, parent_matrix




        # k = [startpoint] #give the start point in a list

        # final_distance, final_parent = snake(k,weight_matrix,distance_mat,parent_mat,visited_mat)

        min_distance = distancematrix[end_point[0],end_point[1]]
        temp = end_point
        #finding path
        pathing=[temp]

        while  True :
            temp = parent_matrix[temp[0],temp[1]]
            pathing.append([temp[0],temp[1]])
            print("r")

            if temp[0]==startpoint[0] and temp[1]==startpoint[1]:
                break

        pathing.reverse()
        return min_distance, pathing

    finalmatrix = findmatrix(contours, 12)
    # distance, path =findpath(finalmatrix,[2,7],[5,11])
    # print(distance)
    # print(path)

    #finding intial path
    starting_point_row,starting_point_col = np.where(finalmatrix == 102)[0][0],np.where(finalmatrix == 102)[1][0]
    p1_row,p2_row,p1_col,p2_col = np.where(finalmatrix == 105)[0][0],np.where(finalmatrix == 105)[0][1],np.where(finalmatrix == 105)[1][0],np.where(finalmatrix == 105)[1][1]
    hospital_corona_row,hospital_corona_col = np.where(finalmatrix == 106)[0][0],np.where(finalmatrix == 106)[1][0]
    hospital_noncorona_row,hospital_noncorona_col = np.where(finalmatrix == 116)[0][0],np.where(finalmatrix == 116)[1][0]

    #making list of all possible paths
    start_p1 = findpath(finalmatrix,[starting_point_row,starting_point_col],[p1_row,p1_col])[1]
    start_p1_time = findpath(finalmatrix,[starting_point_row,starting_point_col],[p1_row,p1_col])[0]
    start_p2 = findpath(finalmatrix,[starting_point_row,starting_point_col],[p2_row,p2_col])[1]
    start_p2_time = findpath(finalmatrix,[starting_point_row,starting_point_col],[p2_row,p2_col])[0]

    p1_corona = findpath(finalmatrix,[p1_row,p1_col],[hospital_corona_row,hospital_corona_col])[1]
    p1_corona_time = findpath(finalmatrix,[p1_row,p1_col],[hospital_corona_row,hospital_corona_col])[0]
    p1_noncorona = findpath(finalmatrix,[p1_row,p1_col],[hospital_noncorona_row,hospital_noncorona_col])[1]
    p1_noncorona_time = findpath(finalmatrix,[p1_row,p1_col],[hospital_noncorona_row,hospital_noncorona_col])[0]
    corona_p1 = findpath(finalmatrix, [hospital_corona_row, hospital_corona_col], [p1_row, p1_col])[1]
    corona_p1_time = findpath(finalmatrix,[hospital_corona_row,hospital_corona_col],[p1_row,p1_col])[0]
    noncorona_p1 = findpath(finalmatrix, [hospital_noncorona_row, hospital_noncorona_col], [p1_row, p1_col])[1]
    noncorona_p1_time = findpath(finalmatrix,[hospital_noncorona_row,hospital_noncorona_col],[p1_row,p1_col])[0]
    # print(noncorona_p1)

    p2_corona = findpath(finalmatrix,[p2_row,p2_col],[hospital_corona_row,hospital_corona_col])[1]
    p2_corona_time = findpath(finalmatrix,[p2_row,p2_col],[hospital_corona_row,hospital_corona_col])[0]
    p2_noncorona = findpath(finalmatrix,[p2_row,p2_col],[hospital_noncorona_row,hospital_noncorona_col])[1]
    p2_noncorona_time = findpath(finalmatrix,[p2_row,p2_col],[hospital_noncorona_row,hospital_noncorona_col])[0]
    corona_p2 = findpath(finalmatrix,[hospital_corona_row,hospital_corona_col],[p2_row,p2_col])[1]
    corona_p2_time = findpath(finalmatrix,[hospital_corona_row,hospital_corona_col],[p2_row,p2_col])[0]
    noncorona_p2 = findpath(finalmatrix,[hospital_noncorona_row,hospital_noncorona_col],[p2_row,p2_col])[1]
    noncorona_p2_time = findpath(finalmatrix,[hospital_noncorona_row,hospital_noncorona_col],[p2_row,p2_col])[0]



    path_p1_corona = [p1_corona, corona_p2, p2_noncorona]
    path_p1_corona_time = start_p1_time + p1_corona_time + corona_p2_time + p2_noncorona_time

    path_p1_noncorona = [p1_noncorona, noncorona_p2, p2_corona]
    path_p1_noncorona_time = start_p2_time+p1_noncorona_time+noncorona_p2_time+p2_corona_time

    path_p1 = [path_p1_corona, path_p1_noncorona]

    path_p2_corona = [p2_corona, corona_p1, p1_noncorona]
    path_p2_corona_time = start_p2_time+p2_corona_time+corona_p1_time+p1_noncorona_time

    path_p2_noncorona = [p2_noncorona, noncorona_p1, p1_corona]
    path_p2_noncorona_time = start_p2_time+p2_noncorona_time+noncorona_p1_time+p1_corona_time

    path_p2 = [path_p2_corona, path_p2_noncorona]

    pt = [path_p1_corona_time, path_p1_noncorona_time, path_p2_corona_time, path_p2_noncorona_time]
    hm = stat.harmonic_mean(pt)

    # print()
    #finding intial path
    if(pt[0] <= pt[3] and pt[1] <= pt[2]):
        pix = 1
        intial_path = start_p1

    elif(pt[0] > pt[3] and pt[1] > pt[2]):
        pix = 2
        intial_path = start_p2

    else:
        if( (path_p1_corona_time+path_p1_noncorona_time - 2*hm) < (path_p2_corona_time + path_p2_noncorona_time - 2*hm) ):
            pix = 1
            intial_path = start_p1

        else:
            pix = 2
            intial_path = start_p2



    # print(finalmatrix)

    def move_my_huskey(start, end, p):

        x = end[0] - start[0]
        y = end[1] - start[1]

        vec_path = np.array([x, y])
        # print(x, y)

        if(vec_path[0] == 0 and vec_path[1] == -1):  # left
            vec_path[0] = -1
            vec_path[1] = 0
            direct = 4
        elif(vec_path[0] == 0 and vec_path[1] == 1):  # right
            vec_path[0] = 1
            vec_path[1] = 0
            direct = 3
        elif(vec_path[0] == -1 and vec_path[1] == 0):  # up
            vec_path[0] = 0
            vec_path[1] = -1
            direct = 1
        elif(vec_path[0] == 1 and vec_path[1] == 0):  # down
            vec_path[0] = 0
            vec_path[1] = 1
            direct = 2

        mag_vec_path = np.linalg.norm(vec_path)

        i = 0
        while True:
            quit = False
            if(i % 10 == 0):
                board = aruco.GridBoard_create(
                    markersX=2,
                    markersY=2,
                    markerLength=0.09,
                    markerSeparation=0.01,
                    dictionary=ARUCO_DICT)

                img = env.camera_feed()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
                k = corners[0]
                k = np.squeeze(k)
                k = np.array(k, dtype=int)
                center_x = (k[0][0] + k[1][0] + k[2][0] + k[3][0]) / 4
                center_y = (k[0][1] + k[1][1] + k[2][1] + k[3][1]) / 4
                centre = np.array([center_x, center_y])

                vec_bot = np.array([(k[0][0] - k[3][0]), (k[0][1] - k[3][1])])
                mag_vec_bot = np.linalg.norm(vec_bot)

                dot_pro = vec_bot @ vec_path
                angle_c = math.acos((dot_pro) / (mag_vec_bot * mag_vec_path)) * 57.2958
                cross_pro = np.cross(vec_bot, vec_path)
                angle_s = math.asin((cross_pro) / (mag_vec_bot * mag_vec_path)) * 57.2958

            # p.stepSimulation()
            i = i + 2

            if (angle_c < 8):  # forward

                if direct == 1 or direct == 2:
                    stop1 = miny + (nodelen[1] * end[0])
                    if direct == 1 and center_y < (stop1 + (0.02 * arenay)):
                        quit = True
                    if direct == 2 and center_y > (stop1 - (0.02 * arenay)):
                        quit = True

                    temp1 = minx + (nodelen[0] * start[1])

                    if center_x > temp1 + (0.007 * arenax):  # align left

                        if direct == 1:
                            p.stepSimulation()
                            env.move_husky(3, 6, 3, 6)
                        else:
                            p.stepSimulation()
                            env.move_husky(6, 3, 6, 3)

                    elif center_x < temp1 - (0.007 * arenax):

                        if direct == 1:
                            p.stepSimulation()
                            env.move_husky(6, 3, 6, 3)
                        else:
                            p.stepSimulation()
                            env.move_husky(3, 6, 3, 6)

                    else:
                        p.stepSimulation()
                        env.move_husky(6, 6, 6, 6)

                if direct == 3 or direct == 4:
                    stop2 = minx + (nodelen[0] * end[1])

                    if direct == 3 and center_x > (stop2 - (0.02 * arenax)):
                        quit = True
                    if direct == 4 and center_x < (stop2 + (0.02 * arenax)):
                        quit = True

                    temp2 = miny + (nodelen[1] * start[0])
                    if center_y > temp2 + (0.01 * arenay):
                        if direct == 3:
                            p.stepSimulation()
                            env.move_husky(3, 6, 3, 6)
                        else:
                            p.stepSimulation()
                            env.move_husky(6, 3, 6, 3)

                    elif center_y < temp2 - (0.01 * arenay):
                        if direct == 3:
                            p.stepSimulation()
                            env.move_husky(6, 3, 6, 3)
                        else:
                            p.stepSimulation()
                            env.move_husky(3, 6, 3, 6)

                    else:
                        p.stepSimulation()
                        env.move_husky(6, 6, 6, 6)

            elif (angle_c >= 8 and angle_s < 0):  # left
                p.stepSimulation()
                env.move_husky(-4, 4, -4, 4)

            elif (angle_c >= 8 and angle_s > 0):  # right
                p.stepSimulation()
                env.move_husky(4, -4, 4, -4)
            else:
                p = 0

            if quit:
                # print(i)
                # print("done")
                break


    def check_patient(img, point):
        for kr in purple:
            if contours[kr].postion[0] == point[0] and contours[kr].postion[1] == point[1]:
                # print(contours[kr].postion,point)
                break

        x, y, w, h = cv2.boundingRect(contours[kr].cont)
        img = img[ y:y + h, x:x + w]
        # cv2.imwrite('abc.png', img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # finding contour of blue colour in purple box
        contour_b = contfind(6, hsv, True)
        # print(contour_b)
        pat = contdetail(contour_b[0], 6)
        if pat.shape == "square":
            return "corona"
        elif pat.shape == "circle":
            return "noncorona"
        else:
            print("can\'t check patient")

    # fimal #
    start_time = time.time()
    print("start_time ---->  ",str(start_time))
    lol = 0
    while True :
        p.stepSimulation()
        env.move_husky(1,1,1,1)
        lol = lol + 2
        if (lol == 100):
            break

    for j in range(len(intial_path) - 1):  # initial path
            if j == len(intial_path) - 2:
                p.stepSimulation()
                env.remove_cover_plate(intial_path[j + 1][0], intial_path[j + 1][1])
                img = env.camera_feed()
                status = check_patient(img, intial_path[j + 1])
                if status == "corona":
                    print("patient tested positive for covid-19")
                    if pix == 1:
                        nextpath = path_p1[0]
                    if pix == 2:
                        nextpath = path_p2[0]
                if status == "noncorona":
                    print("patient tested negative for covid-19")
                    if pix == 1:
                        nextpath = path_p1[1]
                    if pix == 2:
                        nextpath = path_p2[1]

                time.sleep(2)

            move_my_huskey(intial_path[j], intial_path[j + 1], p)

    for qx in range(len(nextpath)):
        time.sleep(2)
        for j in range(len(nextpath[qx]) - 1):  # next path

            if j == len(nextpath[qx]) - 2:
                if qx ==1:
                    p.stepSimulation()
                    env.remove_cover_plate(nextpath[qx][j + 1][0], nextpath[qx][j + 1][1])
                    if status == "noncorona":
                        print("patient tested positive for covid-19")
                    if status == "corona":
                        print("patient tested negative for covid-19")
#
#                    if check_patient(img, nextpath[qx][j + 1]) == "noncorona":
#                        print("patient tested negative covid-19")

                time.sleep(2)

            move_my_huskey(nextpath[qx][j], nextpath[qx][j + 1], p)
        if qx !=1:
            print("......patient dropped sucessfully......")


    final_time = time.time()
    print("stop time ---->  ",str(final_time))
    print("Time taken ---->",str(final_time-start_time))
    print("---------------------------PS completed by team Runtime Error---------------------------")

    while True:
        p.stepSimulation()
        env.move_husky(0, 0, 0, 0)
