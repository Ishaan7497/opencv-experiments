import cv2
cap = cv2.VideoCapture(0)

LOWER_BALL = (20, 60, 60) #testing with tennis ball
UPPER_BALL = (50, 255, 255)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed")
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,LOWER_BALL,UPPER_BALL)
    mask = cv2.erode(mask,None,iterations = 2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(largest)

        if not (300 < area < 5000):
            pass
        else:
            (x,y),radius = cv2.minEnclosingCircle(largest)

            if not (5 < radius < 600):
                pass
            else:
                x,y,radius = int(x),int(y),int(radius)
                cv2.circle(frame,(x,y),radius,(255,0),4)


    cv2.imshow("rect", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()