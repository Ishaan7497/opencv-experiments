import cv2

cap = cv2.VideoCapture("game.mp4")

LOWER_BALL = (5, 80, 60)
UPPER_BALL = (25, 255, 255)

# tracking variables
prev_gray = None
prev_y = None
direction = "STILL"
prev_direction = None

threshold = 7
peak_frames = []
frame_count = 0
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
shot_height= int(height * 0.35)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # grayscale for motion 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        continue
    
    cv2.line(frame,(0,shot_height),(frame.shape[0],shot_height),(255,255,0),2)
    # motion detection
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    # HSV color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BALL, UPPER_BALL)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    # combine motion + color
    combined = cv2.bitwise_and(mask, motion_mask)

    # find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # process contours
    contours = [c for c in contours if cv2.contourArea(c) > 50]

    if contours:
        for c in contours:
            area = cv2.contourArea(c)

            if not (300 < area < 3000):
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = (4 * 3.14 * area) / (perimeter**2)
            if circularity < 0.6:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)

            if not (8 < radius < 30):
                continue

            x, y, radius = int(x), int(y), int(radius)

            # draw ball
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 3)

            # direction detection
            if prev_y is not None:
                dy = y - prev_y

                if abs(dy) < threshold:
                    direction = "STILL"
                elif dy < 0:
                    direction = "UP"
                else:
                    direction = "DOWN"

            cv2.putText(frame, direction, (30, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (125, 125, 255), 2)

            # peak detection
            if prev_direction == "UP" and direction == "DOWN" and y < shot_height:
                cv2.putText(frame, "PEAK", (30, 120),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 3)

                if not peak_frames or (frame_count - peak_frames[-1]) > 60:
                    peak_frames.append(frame_count)

            prev_y = y
            if direction != "STILL":
                prev_direction = direction

            break

    # show output
    cv2.imshow("Frame", frame)
    cv2.imshow("Combined Mask", combined)

    prev_gray = gray

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Peaks at: {peak_frames}")