import cv2
import torch


def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

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


leng = 0
height_f = 615


def plot_boxes(results, frame, classes):
    global leng, height_f
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

    return frame, scale_value_h(height_f), scale_value(leng, height_f), height_f


cap = cv2.VideoCapture(r"D:\yolo_v5\Gate\20230623_085531.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2()

model = torch.hub.load(
    "D:\yolo_v5\gate_state\yolov5",
    "custom",
    source="local",
    path="D:\yolo_v5\yolov5_deploy\Gate_book.pt",
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
    # height, width, _ = frame.shape

    # print(height)

    # roi = frame[275:810, 805:1350]

    # mask = object_detector.apply(roi)

    # _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for cnt in frame:
    # area = cv2.contourArea(cnt)
    # if area > 100:
    #     # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
    #     # x, y, w, h = cv2.boundingRect(cnt)
    #     # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

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
            f"Tinggi: {plot_boxes(results, frame, classes=classes)[3]}%",
            (850, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    frame = plot_boxes(results, frame, classes=classes)[0]

    cv2.imshow("Frame", frame)

    # cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)

    if vid_out:
        print(f"[INFO] Saving output video. . . ")
        out.write(frame)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
