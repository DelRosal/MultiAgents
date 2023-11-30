import cv2 # To use the camera (realtime simulation)
import argparse # To set parameters
import supervision as sv # In this case we are using a deprecared version 0.3.0
import numpy as np
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
    
## Our main function

def main():
    args = parse_arguments()
    frame_width, frame_height = args.camera_resolution

    cap = cv2.VideoCapture(0) # Get the frames from our local camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("./models/yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=0.5
    )

    ## Zone detections
    zone_polygon_left = (ZONE_POLYGON_LEFT * np.array(args.camera_resolution)).astype(int)
    zone_polygon_right = (ZONE_POLYGON_RIGHT * np.array(args.camera_resolution)).astype(int)

    zone_left = sv.PolygonZone(
        polygon=zone_polygon_left,
        frame_resolution_wh=tuple(args.camera_resolution)
    )
    zone_right = sv.PolygonZone(
        polygon=zone_polygon_right,
        frame_resolution_wh=tuple(args.camera_resolution)
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

    entrances = 0
    exits = 0
    last_detected_zone = None  # This variable tracks the last zone where a truck was detected
    frames = 0
    while True:
        ret, frame = cap.read()
        frames += 1

        if frames % 3 == 0:

            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result) # for version 0.16.0 or > of supervision use from_ultralytics(result)
            detections = detections[detections.class_id == 67] # just don't count ourselves 

            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            ## Add square detections
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )

            ## Detection zones
            zone_left.trigger(detections=detections)
            zone_right.trigger(detections=detections)
            frame = zone_annotator_left.annotate(scene=frame)
            frame = zone_annotator_right.annotate(scene=frame)

            # Update the last detected zone based on current counts
            if zone_left.current_count > 0:
                if last_detected_zone == 'right':
                    # Truck was previously detected on the right, now on the left - count as an exit
                    exits += 1
                last_detected_zone = 'left'

            if zone_right.current_count > 0:
                if last_detected_zone == 'left':
                    # Truck was previously detected on the left, now on the right - count as an entrance
                    entrances += 1
                last_detected_zone = 'right'

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27): # Esc key
            break

    print(f"Finished with {entrances} entrances and {exits} exits")
    print(f"frames: {frames}")

if __name__ == '__main__':
    main()