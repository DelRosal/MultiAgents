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
    nFull = 0
    nEmpty = 0
    nTrucks = 0
    frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Break if there are no more frames
        frames += 1

        if frames % frameCheckFrequency == 0:

            result = yolo_model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            detections = detections[detections.class_id != 3]

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
            
            if (zone_right.current_count > 0 or right):
                cv2.imwrite(f"results/img/{datetime.datetime.now().strftime('%H%S')}.jpg", frame)
                right = True
                if (zone_left.current_count > 0):
                    if detections.class_id == 0:
                        nEmpty += 1
                        print("It's a {detections.class_id} ? empty")
                    elif detections.class_id == 1:
                        nFull += 1
                        print(f"The type it's a: {detections.class_id} ? full")
                    nTrucks += 1
                    right = False
            print(detections)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    return nTrucks, nEmpty, nFull


if __name__ == '__main__':
    video_path_1 = './data/et1.avi'
    video_path_2 = './data/ft.avi'
    
    yolo_model = YOLO("./../weights/trucksV11.pt")

    nTrucksE, nEmptyE, nFullE = process_video(video_path_1, yolo_model, 10) # Entrances
    nTrucksX, nEmptyX, nFullX = process_video(video_path_2, yolo_model, 10) # Exits
    
    print(f"\nEntrance finished with {nTrucksE} trucks, {nFullE} fulled and {nEmptyE} empty")
    print(f"\nExit finished with {nTrucksX} trucks, {nFullX} fulled and {nEmptyX} empty")
    
    ## Results to json
    
    result_json = {
        "entranceTrucksEmpty": nEmptyE,
        "entranceTrucksFull": nFullE,
        "exitTrucksEmpty": nEmptyX,
        "exitTrucksFull": nFullX, 
    }
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    file_path = f'results/{current_date}.json'

    # Write the result_json dictionary to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(result_json, json_file)

    print(f"Result saved to {file_path}")

