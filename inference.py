from ultralytics import YOLO
from omegaconf import DictConfig
from pathlib import Path
import hydra  

model = YOLO('runs/detect/eye_detection/weights/best.pt')

@hydra.main(config_path='configs', config_name='config', version_base="1.1")
def main(cfg: DictConfig):
    img = Path(cfg.img_path)

    
    results = model.predict(img) 

   
    for i, r in enumerate(results):
      
        r.save(filename=f"outputs/output_{i}.jpg")

       
        print("Boxes:", r.boxes.xyxy.cpu().numpy())
        print("Confidences:", r.boxes.conf.cpu().numpy()) 
        print("Classes:", r.boxes.cls.cpu().numpy()) 

if __name__ == "__main__":
    main()