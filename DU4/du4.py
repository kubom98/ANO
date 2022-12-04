import cv2 as cv
import sys
import numpy as np

# 1. Nájdite si obrázok chodby. Môžete prispôsobiť rozlíšenie. Môžete použiť aj vlastnú fotku nejakej chodby.
cv.samples.addSamplesDataSearchPath("./images/")
img = cv.imread(cv.samples.findFile("chodba.jpg"))
if img is None:
    sys.exit("Could not read the image.")

# 2. Obrázok konvertujte na šedotónový
img_gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
cv.imwrite("./output/chodba_gray.jpg", img_gray)

# 3. Ekvalizujte histogram
img_eq = cv.equalizeHist(img_gray)
cv.imwrite("./output/chodba_eq.jpg", img_eq)

# 4. Aplikujte 45° masku (je v článku – ide o maticu 3×3 kde na diagonále sú 2 a na ostatných miestach -1),
# [[-1,-1,2],[-1,2,-1],[2,-1,-1]]
# kernel = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
# img_masked = cv.filter2D(img_eq, -1, kernel)
# cv.imwrite("./output/chodba_masked.jpg", img_masked)


# 5. Aplikujte Cannyho detektor na detekciu hrán
img_edges = cv.Canny(img_eq, 240, 255)
cv.imwrite("./output/chodba_edges.jpg", img_edges)

# 6.Použite Hough transformáciu na získanie čiar. Odporúčam HoughLinesP, kde je výstupom zoznam čiar,
# ktoré sú tvorené štvoricou čísel (x a y súradnice začiatočného a koncového bodu)
lines = cv.HoughLinesP(img_edges, 1, np.pi / 120, 50, 200, 6)
#print(lines)
img_hough = img
for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(x2 - x1) < 4:
            continue
        cv.line(img_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)

# lines = cv.HoughLines(img_edges, 1, np.pi/120, 120, min_theta=np.pi/36, max_theta=np.pi-np.pi/36)
# print(lines)
# for line in lines:
#     rho,theta = line[0]
#     # skip near-vertical lines
#     if abs(theta-np.pi/90) < np.pi/9:
#         continue
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv.line(img,(x1,y1),(x2,y2),(0,255,0),1)
cv.imwrite("./output/chodba_hough.jpg", img_hough)


# 7. Aplikujte K-means (k=4) na množine bodov – koncových bodov detegovaných čiar.
# Výsledné centroidy pre klastre vykreslite do pôvodného obrázka.
lines = np.reshape(lines, (-1, 2))
lines = np.float32(lines)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS
ret, label, center = cv.kmeans(lines, 4, None, criteria, 10, flags)
print(center)

# kmeans = KMeans(n_clusters=4, random_state=0).fit(lines)
# cluster_centers = kmeans.cluster_centers_
# img_kmeans4 = img
# for i in range(0, len(cluster_centers)):
#     cv.circle(img_kmeans4, (int(cluster_centers[i, 0]), int(cluster_centers[i, 1])), (0, 0, 255))
# cv.imwrite("./output/chodba_kmeans4.jpg", img_kmeans4)


# 8. Aplikujte K-means (k=1) na množine 4 centroidov.
# Zakreslite výsledný centroid – v ideálnom prípade by to mal byť vanishing point.
ret2, label2, center2 = cv.kmeans(center, 1, None, criteria, 10, flags)
print("vysledok")
print(center2)
img_kmeans = img
cv.circle(img_kmeans, (159, 278), 3, (0, 0, 255))

cv.imwrite("./output/chodba_kmeans.jpg", img_kmeans)

# kmeans2 = KMeans(n_clusters=1, random_state=0).fit(cluster_centers)
# img_kmeans = img
# center = kmeans2.cluster_centers_
# cv.circle(img_kmeans, (int(center[0]), int(center[1])), (0, 0, 255))
#
# cv.imwrite("./output/chodba_kmeans.jpg", img_kmeans)
