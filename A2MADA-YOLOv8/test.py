from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('best.pt')
    model.val(**{'data': 'data/data.yaml'})
