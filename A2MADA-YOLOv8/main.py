from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8l.yaml')
    model.train(**{'cfg': 'ultralytics/cfg/default.yaml', 'data': 'data/data.yaml'})
