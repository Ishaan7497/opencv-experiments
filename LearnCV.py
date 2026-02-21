import cv2
cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    cv2.imshow("Camera", frame)
    cv2.imshow("video",gray)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()