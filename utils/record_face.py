import cv2

def record_video(filename, duration):
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    start_time = cv2.getTickCount()
    while True:
        ret, frame = cap.read()

        out.write(frame)

        cv2.imshow('frame', frame)

        if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage
record_video('my_face_video2min.mp4', 5)
