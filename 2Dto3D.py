import cv2
import numpy as np
# 이미지 로드
image = cv2.imread('sample.jpg')
# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 깊이 맵 생성 (가상의 깊이 적용)
depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
# 결과 출력
cv2.imshow('Original Image', image)
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()