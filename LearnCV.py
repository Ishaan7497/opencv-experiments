import cv2
cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)


    cv2.imshow("Diff", diff*3)

    prev_gray = gray

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()