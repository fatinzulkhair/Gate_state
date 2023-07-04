import cv2
import torch
import time
from datetime import datetime


def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


def scale_value(value, height_f):
    scaled_h = (height_f - 0) / (615 - 0)
    scaled_value = (value - 0) / ((950 * scaled_h) - 0) * 100
    if scaled_value > 100:
        scaled_value = 100
    return round(scaled_value)


def scale_value_h(height_f):
    scaled_h = (height_f - 0) / (615 - 0)
    return round(scaled_h)


def send_warning(text):
    # Specify the file path
    file_path = "Email_warning.txt"
    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write the text to the file
        file.write(text)


def state_and_send(state_percentage):
    global state, triger, start_time
    if state is None:
        if state_percentage > 70:
            state = "close"
        if state_percentage < 30:
            state = "open"
        triger = ""
    elif state != "close":
        if state_percentage > 70:
            triger = "off"
            state = "close"
        else:
            if triger == "on":
                time_now = time.time() - start_time
                if time_now > 30:
                    send_warning(
                        f'[GATE WARNING] the gate had been open for half an hour [{datetime.now().strftime("%H:%M %Y-%m-%d")}]s'
                    )
                    triger = "off"

    elif state != "open":
        if state_percentage < 30:
            state = "open"
            start_time = time.time()
            triger = "on"
    return state


leng = 0
height_f = 615
state = None
triger = None


def plot_boxes(results, frame, classes):
    global leng, height_f, state
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . .")
    print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if (
            row[4] >= 0.60
        ):  ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = (
                int(row[0] * x_shape),
                int(row[1] * y_shape),
                int(row[2] * x_shape),
                int(row[3] * y_shape),
            )  ## BBOx coordniates
            text_d = classes[int(labels[i])]

            if text_d == r"Gate":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  ## BBox
                cv2.rectangle(
                    frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1
                )  ## for text label background
                cv2.putText(
                    frame,
                    text_d,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            leng = x2 - x1
            height_f = y2 - y1

    state = state_and_send(scale_value(leng, height_f))

    return (
        frame,
        scale_value_h(height_f),
        scale_value(leng, height_f),
        height_f,
        state,
    )


cap = cv2.VideoCapture(r"20230623_085531.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2()

model = torch.hub.load(
    "yolov5",
    "custom",
    source="local",
    path="Gate_book.pt",
    force_reload=True,
)

vid_out = "gerbang.mp4"

if vid_out:  ### creating the video writer if video output path is given
    # by default VideoCapture returns float instead of int
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"mp4v")  ##(*'XVID')
    out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

# cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    results = detectx(frame, model=model)
    classes = model.names

    if ret:
        frame = cv2.putText(
            frame,
            f"Scale: {plot_boxes(results, frame, classes=classes)[1]}",
            (850, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frame = cv2.putText(
            frame,
            f"Gerbang tertutup: {plot_boxes(results, frame, classes=classes)[2]}%",
            (850, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frame = cv2.putText(
            frame,
            f"State: {plot_boxes(results, frame, classes=classes)[4]}",
            (850, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    frame = plot_boxes(results, frame, classes=classes)[0]

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)

    if vid_out:
        print(f"[INFO] Saving output video. . . ")
        out.write(frame)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
