import cv2

img = cv2.imread("/data/pfy/dataset/MVTec-AD/bottle/test/broken_large/000.png")
H, W = img.shape[:2]
# normalized bbox
xmin, ymin, xmax, ymax = 0.315556, 0.266667, 0.912222, 0.942222
# convert to pixel
x1, y1 = int(xmin * W), int(ymin * H)
x2, y2 = int(xmax * W), int(ymax * H)
# draw red box
cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
cv2.imwrite("./bbox_result.png", img)