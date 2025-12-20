from time import sleep

while True:
    frame = cam.read_frame()
    
    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    cropped = region_of_interest(edges)

    # Check presence
    lane_ok = is_lane_present(cropped)

    if lane_ok:
        go_forward()    # YOUR motor control function
        print("Lane OK ? moving forward")
    else:
        stop()          # YOUR motor control function
        print("Lane LOST ? stopping")

    cv2.imshow("Edges", cropped)
    if cv2.waitKey(1) == 27:
        stop()
        break
