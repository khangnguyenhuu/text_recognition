import cv2
import numpy as np 
import scipy.spatial.distance as distance

'''
input: a list coordinate include 4 corner of the bounding box like
coor = [[208,19,299,24,297,56,206,51],
        [54,24,138,24,138,52,54,52],
        [146,29,198,29,198,57,146,57],
        [24,70,80,70,80,98,24,98],
        [86,72,150,72,150,104,86,104],
        [193,72,308,73,307,100,193,98],
        [158,76,181,76,181,98,158,98],
        [235,382,324,387,322,419,233,414],
        [81,386,165,386,165,414,81,414],
        [173,392,225,392,225,420,173,420],
        [208,422,224,422,224,441,208,441],
        [52,433,106,433,106,462,52,462],
        [113,436,177,436,177,468,113,468],
        [206,446,317,444,317,470,206,472],
        [208,480,278,480,278,500,208,500]]
output: sorted list of 4 corner = 8 points [[0,1],[2,3],[4,5],[6,7]] of each bounding box
(0,1)-------(2,3)
  |            |
  |            |
  |            |
(6,7)--------(4,5) 
'''

def sorting_bounding_box(points):
    # lấy ra xmin ymin, xmax ymax của mỗi bbox
    points = list(map(lambda x:[x[0], x[2], x[1], x[3]],points))
    # print(points)
    # lấy ra xmin, ymin, sum của xmin, ymin và ymax
    points_sum = list(map(lambda x:[x[0],sum(x[0]),x[1][1], x[1], x[2], x[3]],points))
    # print (points_sum)
    # print ('-----------------------------')
    # lấy ra xmin, ymin
    x_y_cordinate = list(map(lambda x: x[0],points_sum))
    final_sorted_list = []
    while True:
        try:
            new_sorted_text = []
            initial_value_A  = [i for i in sorted(enumerate(points_sum), key=lambda x:x[1][1])][0]
            threshold_value = abs(initial_value_A[1][0][1] - initial_value_A[1][2])
            threshold_value = (threshold_value/2) + 5
            del points_sum[initial_value_A[0]]  
            del x_y_cordinate[initial_value_A[0]]
    #         print(threshold_value)
            A = [initial_value_A[1][0]]
            K = list(map(lambda x:[x,abs(x[1]-initial_value_A[1][0][1])],x_y_cordinate))
            K = [[count,i]for count,i in enumerate(K)]
            print ('--------------------------------------------', K)
            for i in K: print (i[1][1])
            K = [i for i in K if i[1][1] <= threshold_value]
            sorted_K = list(map(lambda x:[x[0],x[1][0]],sorted(K,key=lambda x:x[1][1])))
            B = []
            points_index = []
            for tmp_K in sorted_K: 
                points_index.append(tmp_K[0])
                B.append(tmp_K[1])
            dist = distance.cdist(A,B)[0]
            d_index = [i for i in sorted(zip(dist,points_index), key=lambda x:x[0])]
            new_sorted_text.append([initial_value_A[1][0], initial_value_A[1][4],  initial_value_A[1][3],  initial_value_A[1][5]])
            index = []
            for j in d_index:
                tmp = [points_sum[j[1]][0], points_sum[j[1]][4], points_sum[j[1]][3], points_sum[j[1]][5]]
                new_sorted_text.append(tmp)
                index.append(j[1])
            for n in sorted(index, reverse=True):
                del points_sum[n]
                del x_y_cordinate[n]
            final_sorted_list.append(new_sorted_text)
            # print(new_sorted_text)
        except Exception as e:
            print(e)
            break

    return final_sorted_list

def visual(img, coordinates):
    #boundingBoxes = [(coor[0], coor[1], coor[4], coor[5]) for coor in coordinates]
    #print (boundingBoxes)
    coordinate = []
    count = 1
    for i, line in enumerate(coordinates):
        for j in line:
        # coordinate.append([line[j][0],line[i][1]])
            cv2.rectangle(img, (int(j[0][0]), int(j[0][1])), (int(j[2][0]), int(j[2][1])), (0,255,0), 1)
            cv2.putText(img, str(count), (int (j[0][0]), int(j[0][1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
            count += 1    
            
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
    cv2.imwrite('img_Re.jpg', img)

coor = [[[33.333332 ,  6.6666665],
        [53.333332 ,  6.6666665],
        [53.333332,  29.333334 ],
        [33.333332 , 29.333334 ]],

        [[ 8.   ,      8.       ],
        [29.333334  , 8.       ],
        [29.333334  ,33.333332 ],
        [ 8.        ,33.333332 ]],

        [[-1.6105617, 34.561058 ],
        [59.64356  , 28.435644 ],
        [62.270626 , 54.70627  ],
        [ 1.0165049, 60.83168  ]]]

def sort_bb(bboxes):
    print ('bboxes: ', len (bboxes))
    lines = []
    dist = 10
    # sort with y coor
    if (len(bboxes) == 2):
        return bboxes
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            if (abs(bboxes[i][0][1] - bboxes[j][0][1]) < dist):
                lines.append([bboxes[i], bboxes[j]])
    if (len(bboxes) % 2 != 0):
        lines.append([bboxes[len(bboxes) - 1]])
    sorted_list = []
    # sort in a line by x coor
    new_lines = []
    for i in lines:
        i = [j for j in sorted(i, key=lambda x:x[0][0])]
        new_lines.append(i)    
    for i in new_lines:
        for j in i:
            sorted_list.append(j)
    del lines
    del new_lines
    return sorted_list
    # final_list = []
    # # merge bb in a line
    # for i in sorted_list:
    #     if (len(i) > 1):
    #         print (i)
    #         final_list.append([i[0][0], i[-1][1], i[-1][2], i[0][3]])
    #     else:
    #         final_list.append([i[0][0], i[0][1], i[0][2], i[0][3]])
    # return final_list               
