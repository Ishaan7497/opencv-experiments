import cv2
cap = cv2.VideoCapture(0)

LOWER_BALL = (20, 60, 60) #testing with tennis ball
UPPER_BALL = (50, 255, 255)
prev_y = None
direction = None
prev_direction = None
threshold = 5
peak_frames = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed")
        break
    frame_count += 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,LOWER_BALL,UPPER_BALL)
    mask = cv2.erode(mask,None,iterations = 2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(largest)

        if not (300 < area < 6000):
            pass
        else:
            (x,y),radius = cv2.minEnclosingCircle(largest)

            if not (5 < radius < 1000):
                pass
            else:
                x,y,radius = int(x),int(y),int(radius)
                cv2.circle(frame,(x,y),radius,(255,0),4)

                if prev_y is not None:
                    dy = y - prev_y
                    if abs(dy) < threshold:
                        direction = "STILL"
                    elif (dy) < 0:
                        direction = "UP" 
                    else:
                        direction = "DOWN"
                
                cv2.putText(frame,direction,(30,50),cv2.FONT_HERSHEY_COMPLEX,1,(125,125,255),3)
                
                if prev_direction == "UP" and direction == "DOWN":
                    cv2.putText(frame,"PEAK",(30,120),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),3)
                    if len(peak_frames) == 0 and (frame_count - peak_frames[-1]) > 10: # to prevent clip repetition
                        peak_frames.append(frame_count)

                prev_y = y
                if direction != "STILL":
                    prev_direction = direction


    cv2.imshow("rect", frame)
    cv2.imshow("Mask", mask)
    

    if cv2.waitKey(1) == ord('q'):
        break
print(f"Peaks at {peak_frames}")
cap.release()
cv2.destroyAllWindows()