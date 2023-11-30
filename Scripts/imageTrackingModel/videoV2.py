import cv2 # To use the camera (realtime simulation)
import argparse # To set parameters
import supervision as sv # In this case we are using a deprecared version 0.3.0
import numpy as np
import json
import datetime
from ultralytics import YOLO 

## Zone detection to count detections
ZONE_POLYGON_LEFT= np.array([
    [0, 0],
    [0.35, 0],
    [0.35, 1],
    [0, 1]
])

ZONE_POLYGON_RIGHT = np.array([
    [0.65, 0],
    [1, 0],
    [1, 1],
    [0.65, 1]
])

## Set parameters
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--camera-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def process_video(video_path, yolo_model, frameCheckFrequency):
    args = parse_arguments()
    frame_width, frame_height = args.camera_resolution

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=0.5
    )

    zone_polygon_left = (ZONE_POLYGON_LEFT * np.array([frame_width, frame_height])).astype(int)
    zone_polygon_right = (ZONE_POLYGON_RIGHT * np.array([frame_width, frame_height])).astype(int)

    zone_left = sv.PolygonZone(
        polygon=zone_polygon_left,
        frame_resolution_wh=(frame_width, frame_height)
    )
    zone_right = sv.PolygonZone(
        polygon=zone_polygon_right,
        frame_resolution_wh=(frame_width, frame_height)
    )

    zone_annotator_left = sv.PolygonZoneAnnotator(
        zone=zone_left,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )
    zone_annotator_right = sv.PolygonZoneAnnotator(
        zone=zone_right,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    left = False
    right = False
    exits = 0
    entrances = 0
    frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Break if there are no more frames
        frames += 1

        if frames % frameCheckFrequency == 0:

            result = yolo_model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            detections = detections[detections.class_id == 7]

            labels = [
                f"{yolo_model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            zone_left.trigger(detections=detections)
            zone_right.trigger(detections=detections)
            frame = zone_annotator_left.annotate(scene=frame)
            frame = zone_annotator_right.annotate(scene=frame)

            if (zone_left.current_count > 0 or left):
                left = True
                if (zone_right.current_count > 0):
                    entrances += 1
                    left = False
            if (zone_right.current_count > 0 or right):
                right = True
                if (zone_left.current_count > 0):
                    exits += 1
                    right = False

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

    print(f"Finished with {entrances} entrances and {exits} exits")

    cap.release()
    cv2.destroyAllWindows()

    return entrances, exits


if __name__ == '__main__':
    video_path_1 = './data/Simulador_(3).avi'
    video_path_2 = './data/Simulador_(4).avi'
    yolo_model = YOLO("./../weights/yolov8l.pt")
    entrances_c1, exits_c1 = process_video(video_path_1, yolo_model, 15)
    entrances_c2, exits_c2 = process_video(video_path_2, yolo_model, 15)

    total_entrances = entrances_c1 + entrances_c2
    total_exits = exits_c1 + exits_c2

    print(f"\nTotal entrances: {total_entrances}\nTotal exits: {total_exits}")
    
    result_json = {
        "entrances": total_entrances,
        "exits": total_exits
    }
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_path = f'results/{current_date}.json'

    # Write the result_json dictionary to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(result_json, json_file)

    print(f"Result saved to {file_path}")

