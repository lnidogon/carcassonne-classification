

#!/usr/bin/env python
# coding: utf-8

# In[88]:

import os, sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image


# In[89]:


example_name = sys.argv[1]
tile_resolution = 2048
map_resolution = 4096 


# In[90]:


import cv2
# https://www.freedomvc.com/index.php/2022/01/17/basic-background-remover-with-opencv/
def bgremove3(myimage):
    # BG Remover 3
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)

    #Take S and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded

    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask

    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer

    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background # Combine foreground and background

    return finalimage

print("Start")

img = cv.imread('./images/' + example_name, cv.IMREAD_COLOR)

print("Loaded the image")

original_image = img


nbg = bgremove3(img)
kernel = np.ones((20,20),np.uint8)
erosion = cv.morphologyEx(nbg, cv.MORPH_OPEN, kernel)
#plt.imshow(erosion)


print("Removed the background")

# In[91]:


def contourLargest(image):
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 1)
    plt.imshow(thresh)
    contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], -1, (255), 3)
    return mask


# In[92]:


def length(line):
    return math.sqrt((line[0] - line[2])**2 + (line[1] - line[3])**2)
def intersect(line1, line2):
    if line1[2] != line1[0]:
        a1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
    else:
        a1 = 1e9
    b1 = line1[3] - line1[2] * a1
    if line2[2] != line2[0]:
        a2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
    else:
        a2 = 1e9
    if abs(math.atan(a1) - math.atan(a2)) < 1/4 * math.pi:
        return (-1, -1)
    b2 = line2[3] - line2[2] * a2
    if a2 != a1:
        x = -(b2 - b1)/(a2 - a1)
    else:
        x = -1
    y = a1 * x + b1
    return (math.floor(x), math.floor(y))

def lineAngle(line):
    if line[2] != line[0]:
        a = (line[3] - line[1]) / (line[2] - line[0])
    else:
        a = 1e9
    if line[2] != line[0]:
        a = (line[3] - line[1]) / (line[2] - line[0])
    else:
        a = 1e9
    return math.atan(a)

def arePerpendicular(line1, line2):
    if line1[2] != line1[0]:
        a1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
    else:
        a1 = 1e9
    if line2[2] != line2[0]:
        a2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
    else:
        a2 = 1e9
    if abs(math.atan(a1) - math.atan(a2)) < 1/4 * math.pi:
        return False
    return True
def arePointsClose(p1, p2, thresh=4):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < thresh


# In[93]:

dst = cv.cvtColor(contourLargest(erosion), cv.COLOR_BGR2GRAY)
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)
#plt.imshow(cdst, cmap='gray')

print("Found the largest contour")

lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)


linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
x = 0
verticies = []
if linesP is not None:
    for i in range(0, len(linesP)):
        for j in range(i + 1, len(linesP)):
            isc = (-1, -1)
            if arePointsClose(linesP[i][0][0:2], linesP[j][0][0:2]) or arePointsClose(linesP[i][0][0:2], linesP[j][0][2:4]):
                isc = linesP[i][0][0:2]
            if arePointsClose(linesP[i][0][2:4], linesP[j][0][0:2]) or arePointsClose(linesP[i][0][2:4], linesP[j][0][2:4]):
                isc = linesP[i][0][2:4]
            if not arePerpendicular(linesP[i][0], linesP[j][0]) or isc[0] == -1:
                continue
            else:
                x = x + 1
                verticies.append(isc)
                cv.circle(cdstP, isc, radius=5, color=(255, 255, 255), thickness=-1)
#plt.imshow(cdstP)

print("Found contour corners")


# In[94]:


linesV = []
linesH = []
for line in linesP:
    if abs(lineAngle(line[0])) < 1/4 * math.pi:
        linesH.append(line[0])
    else:
        linesV.append(line[0])

maxh = 0
minh = 1e9
maxv = 0
minv = 1e9


for line in linesV: 
    v = (line[0] + line[2])/2
    if v > maxv:
        maxv = v
        maxvL = line
    if v < minv:
        minv = v
        minvL = line
for line in linesH:
    h = (line[1] + line[3])/2
    if h > maxh:
        maxh = h
        maxhL = line
    if h < minh:
        minh = h
        minhL = line

print("Found approximate bounding box")
cv.line(cdstP, maxhL[0:2], maxhL[2:4], (0,255,0), 3, cv.LINE_AA)
cv.line(cdstP, minhL[0:2], minhL[2:4], (0,255,0), 3, cv.LINE_AA)
cv.line(cdstP, maxvL[0:2], maxvL[2:4], (255,0,0), 3, cv.LINE_AA)
cv.line(cdstP, minvL[0:2], minvL[2:4], (255,0,0), 3, cv.LINE_AA)

iscA = intersect(minhL, minvL)
iscB = intersect(minhL, maxvL)
iscC = intersect(maxhL, minvL)
iscD = intersect(maxhL, maxvL)

cv.circle(cdstP, iscA, radius=5, color=(255, 0, 0), thickness=-1)
cv.circle(cdstP, iscB, radius=5, color=(255, 0, 0), thickness=-1)
cv.circle(cdstP, iscC, radius=5, color=(255, 0, 0), thickness=-1)
cv.circle(cdstP, iscD, radius=5, color=(255, 0, 0), thickness=-1)

srcPoints = np.array([iscA, iscB, iscC, iscD], np.float32)
dstPoints = np.array([[0, 0], [map_resolution, 0], [0, map_resolution], [map_resolution, map_resolution]], np.float32)


perspectiveMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
outputSize = (map_resolution, map_resolution)

warped = cv2.warpPerspective(img, perspectiveMatrix, outputSize)
original_image_warped = cv2.warpPerspective(original_image, perspectiveMatrix, outputSize)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

print("Warped the image")

#plt.imshow(warped, cmap='gray')


# In[95]:


vertices_array = np.array(verticies)
homogeneous_vertices = np.hstack((vertices_array, np.ones((vertices_array.shape[0], 1))))
transformed_vertices = homogeneous_vertices @ perspectiveMatrix.T
transformed_vertices_cartesian = transformed_vertices[:, :2] / transformed_vertices[:, 2][:, np.newaxis]
flat_vertices = np.array(transformed_vertices_cartesian)
unique_flat_vertices = []
for i in range(len(flat_vertices)):
    unique = True
    for j in range(len(unique_flat_vertices)):
        dist = length([flat_vertices[i][0], flat_vertices[i][1], unique_flat_vertices[j][0], unique_flat_vertices[j][1]])
        if dist < map_resolution // 100:
            unique = False
            break
    if unique:
        unique_flat_vertices.append(flat_vertices[i])
for v in unique_flat_vertices:
    cv.circle(warped, (math.floor(v[0]), math.floor(v[1])), radius=map_resolution // 100, color=(255, 255, 255), thickness=-1)
#plt.imshow(warped)

print("Warped corner vertices")

# In[96]:


lengthH = []
anglesH = []
lengthV = []
anglesV = []
for i in range(0, len(unique_flat_vertices)):
    for j in range(i + 1, len(unique_flat_vertices)):
        vertex1 = unique_flat_vertices[i]
        vertex2 = unique_flat_vertices[j]
        line = [vertex1[0], vertex1[1], vertex2[0], vertex2[1]]
        ang = lineAngle(line)
        leng = length(line)
        if(abs(ang) < 1/50 * math.pi):
            lengthH.append(leng)
            anglesH.append(ang)
        if(abs(abs(ang) - math.pi/2) < 1/50 * math.pi):
            lengthV.append(leng)
            anglesV.append(ang)
lengthH.sort()
anglesH.sort()
lengthV.sort()
anglesV.sort()
temp = []
lengthHC = []
prevL = None
for l in lengthH:
    if prevL == None or l - prevL > 5:
        if prevL != None:
            lengthHC.append(temp)
        temp = [l]
    else:
        temp.append(l)
    prevL = l
lengthHC.append(temp)
temp = []
lengthVC = []
prevL = None
for l in lengthV:
    if prevL == None or l - prevL > 5:
        if prevL != None:
            lengthVC.append(temp)
        temp = [l]
    else:
        temp.append(l)
    prevL = l
lengthVC.append(temp)
lengthHCA = []
lengthVCA = []
for cluster in lengthHC:
    lengthHCA.append(np.mean(cluster))
for cluster in lengthVC:
    lengthVCA.append(np.mean(cluster))



tilelX = None
tilelY = None
tileaX = np.mean(anglesH)
tileaY = np.mean([x if x > 0 else x + math.pi for x in anglesV])

print("Calculated the average angle and all lengths between points")

for i in range(len(lengthHCA)):
    corr = 0
    incr = 0
    for j in range(i, len(lengthHCA)):
        if abs(abs(lengthHCA[j]/lengthHCA[i]) - abs(round(lengthHCA[j]/lengthHCA[i]))) < 0.15:
            corr = corr + 1
        else:
            incr = incr + 1
    if corr > incr:
        tilelX = lengthHCA[i]
        break
for i in range(len(lengthVCA)):
    corr = 0
    incr = 0
    for j in range(i, len(lengthVCA)):
        if abs(abs(lengthVCA[j]/lengthVCA[i]) - abs(round(lengthVCA[j]/lengthVCA[i]))) < 0.15:
            corr = corr + 1
        else:
            incr = incr + 1
    if corr > incr:
        tilelY = lengthVCA[i]
        break


print("Found the shortest 'possible' tile width and height")

# In[104]:


cornerClusters = [] 
unique_flat_vertices.sort(key=lambda x: x[0])
for v in unique_flat_vertices:
    for i in range(-20, 20):
        for j in range(-20, 20):
            xcor = v[0] + i * tilelX * math.cos(tileaX) + j * math.cos(tileaY) * tilelY
            ycor = v[1] + j * tilelY * math.sin(tileaY) + i * math.sin(tileaX) * tilelX
            if xcor >= -map_resolution / 5 and xcor <= map_resolution * 1.2 and ycor >= -map_resolution / 5 and ycor <= map_resolution * 1.2:
                cornerClusters.append([xcor, ycor])
                cv.circle(warped, (math.floor(xcor), math.floor(ycor)), radius=10, color=(0, 0, 255), thickness=-1)
original_image_warped = cv.cvtColor(original_image_warped, cv.COLOR_BGR2RGB)

print("Expanded each contour corner on the warped image")

#plt.imshow(warped)


# In[1]:


cornerAprox = []
for corner in cornerClusters:
    placed = False
    for i, ca in enumerate(cornerAprox):
        cavg = [np.mean(ca[:,0]), np.mean(ca[:,1])]
        line = [corner[0], corner[1], cavg[0], cavg[1]]
        if length(line) <  map_resolution // 20:
            cornerAprox[i] = np.append(ca, [corner], axis=0)
            placed = True
            break
    if placed == False:
        cornerAprox.append(np.array([corner]))
goodCornersMask = np.array([len(x) > 1 for x in cornerAprox])

print("Grouped corner clusters into groups")


filtered_clusters = [cornerAprox[i] for i in range(len(cornerAprox)) if goodCornersMask[i]]
actualCorners = []
for ca in filtered_clusters:
    cavg = [np.mean(ca[:,0]), np.mean(ca[:,1])]
    actualCorners.append(cavg)
    cv.circle(warped, (math.floor(cavg[0]), math.floor(cavg[1])), radius=20, color=(255, 0, 0), thickness=-1)
#plt.imshow(warped)

print("Averaged the grid into approximate corners")


# In[99]:




# In[100]:


def otherCorners(vertex, vertexList, tileW, tileH, tileAX, tileAY):
    xcor1 = vertex[0]
    ycor1 = vertex[1]
    xcor2 = vertex[0] + tileW * math.cos(tileAX) + math.cos(tileAY) * tileW
    ycor2 = vertex[1] + tileH * math.sin(tileAY) + math.sin(tileAX) * tileH
    cornerA = vertex
    cornerB = min(vertexList, key=lambda x: (x[0] - xcor2)**2 + (x[1] - ycor1)**2)
    cornerC = min(vertexList, key=lambda x: (x[0] - xcor1)**2 + (x[1] - ycor2)**2)
    cornerD = min(vertexList, key=lambda x: (x[0] - xcor2)**2 + (x[1] - ycor2)**2)
    if cornerA == cornerB or cornerA == cornerC or cornerA == cornerD:
        return [-1, -1], None, None, None
    return cornerA, cornerB, cornerC, cornerD


# In[101]:


upperLeftCorner = min(actualCorners, key=lambda x: x[0]**2 + x[1]**2)
cv.circle(warped, (math.floor(upperLeftCorner[0]), math.floor(upperLeftCorner[1])), radius=20, color=(0, 255, 0), thickness=-1)
#plt.imshow(original_image_warped)
cutoutArray = []
rollingCorner = upperLeftCorner
nextStartOfRowCorner = []
startOfRow = True
numberOfTilesY = numberOfTilesX = 0
x = 0
while True:
    x += 1
    if x > 10000:
        break
    ca, cb, cc, cd = otherCorners(rollingCorner, actualCorners, tilelX, tilelY, tileaX, tileaY)
    if ca == [-1, -1]:
        if startOfRow:
            break
        else:
            rollingCorner = nextStartOfRowCorner
            startOfRow = True
            continue
    if startOfRow:
        numberOfTilesY += 1
        numberOfTilesX = 0
        nextStartOfRowCorner = cc
        startOfRow = False
    numberOfTilesX += 1
    cornerPoints = np.array([ca, cb, cc, cd], np.float32)
    dstPoints = np.array([[0, 0], [tile_resolution, 0], [0, tile_resolution], [tile_resolution, tile_resolution]], np.float32)
    cutoutMatrix = cv2.getPerspectiveTransform(cornerPoints, dstPoints)
    outputSizeCutout = (tile_resolution, tile_resolution)
    cutout = cv2.warpPerspective(original_image_warped, cutoutMatrix, outputSizeCutout)
    cutoutArray.append(cutout)
    rollingCorner = cb

print("Created cutouts")

numberOfCutouts = len(cutoutArray)
fig, axs = plt.subplots(numberOfTilesY, numberOfTilesX, figsize=(10, 5))
'''
for i, cutout in enumerate(cutoutArray):
    axs[i//numberOfTilesX][i%numberOfTilesX].imshow(cutout)
    axs[i//numberOfTilesX][i%numberOfTilesX].axis('off') 
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
'''

# In[102]:


sc_filePath= os.path.join(".", os.path.join("cutouts", os.path.splitext(example_name)[0])) + "/"

if not os.path.isdir(sc_filePath):
    os.mkdir(sc_filePath)


# In[103]:

THRES = 50
for i, cutout in enumerate(cutoutArray):
    coloredImage = cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
    w, h, _ = coloredImage.shape
    imageCenter = coloredImage[w // 4:w // 4 * 3,h // 4:h//4 * 3]
    now = np.sum(np.all(imageCenter < THRES, axis=-1))
    nob = np.sum(np.all(imageCenter > 255 - THRES, axis=-1))
    nall = np.sum(np.all(imageCenter > -1, axis=-1))
    var = np.average(np.var(imageCenter, axis=(0,1)))
    bwPart = (now + nob) / nall
    #print(f"{i//numberOfTilesX}, {i%numberOfTilesX}, {bwPart} -- {var}")
    if bwPart > 0.05 or var < 100: 
        continue
    cv2.imwrite(sc_filePath + str(i//numberOfTilesX) + "_" + str(i%numberOfTilesX) + ".jpg", coloredImage)

print("DONE")

