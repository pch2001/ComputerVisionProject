import cv2



cap = cv2.VideoCapture('recorde1.mp4')
while True:
	success, img = cap.read()
	cv2.imshow('video', img)
	if cv2.waitKey(33) & 0xff == ord('q') :
		break;