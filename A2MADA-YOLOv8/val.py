from ultralytics import YOLO

# 安装命令
# python setup.py develop

if __name__ == '__main__':

    model = YOLO('best.pt')
    model.val(**{'data': 'data/data.yaml'})