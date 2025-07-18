import numpy as np
import cv2
import os

def main():
    # Thresholds
    thres = 0.5  # Confidence threshold
    nms_threshold = 0.2  # NMS threshold

    # Open Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

    # Load Class Names
    if not os.path.exists("objects.txt"):
        print("⚠️ ERROR: File 'objects.txt' is not found!")
        exit()

    with open("objects.txt", "r") as f:
        classNames = f.read().splitlines()

    print("✅ Class detected:", classNames)

    # Colors for Bounding Boxes
    Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

    # Load Model
    weightsPath = "frozen_inference_graph.pb"
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    if not os.path.exists(weightsPath) or not os.path.exists(configPath):
        print("⚠️ ERROR: Model is not found!")
        exit()

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Set untuk menyimpan objek yang sudah terdeteksi
    detected_objects = set()

    # Output file
    output_filename = "output.txt"
    output_file = open(output_filename, "w", encoding="utf-8")
    output_file.write("You are currently seeing \n")
    detected_objects = set()

    while True:
        # ✅ Check stop signal
        if os.path.exists("stop_camera.txt"):
            print("🛑 Stop signal received, exiting camera loop.")
            break

        success, img = cap.read()
        if not success:
            print("⚠️ ERROR: Camera can't be read")
            break

        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) == 0:
            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        bbox = list(bbox)
        confs = list(map(float, np.array(confs).reshape(-1)))
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        if len(indices) == 0:
            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for i in indices.flatten():
            x, y, w, h = bbox[i]
            confidence = str(round(confs[i], 2))
            label = classNames[classIds[i] - 1]
            color = Colors[classIds[i] - 1]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence}", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

            # Suppose this runs in a loop
            if label not in detected_objects:
                detected_objects.add(label)

        cv2.imshow("Output", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Convert set to a list and sort or keep order if needed
    object_list = list(detected_objects)

    for i, obj in enumerate(object_list):
        output_file.write(obj)
        if i != len(object_list) - 1:
            output_file.write("\nthen\n")
            print(f"💾 Saving to file: {obj}")

    # Clean up
    output_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()