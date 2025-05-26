from dataloader import roboflow
from ultralytics import YOLO
from multiprocessing import freeze_support
from torch.cuda import is_available

def train_model(dataset , epochs, imgsz, base_model="yolov8m.pt" , device = 'cuda' if is_available() else 'cpu'):
    print("Status : Downloading Datasets")
    
    dataset = roboflow("EyRL1wCn9xaPZrnPnc4b" , "prayag-pawar-w3xwz" , "tanks_f", 16, "yolov8")
    
    model = YOLO(base_model) 
    
    results = model.train(data=dataset, epochs=epochs, imgsz=imgsz , device=device)
    
def main():
    train_model()
    
if __name__ == "__main__":
    freeze_support()
    main()