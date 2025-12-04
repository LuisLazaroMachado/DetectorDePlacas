from ultralytics import YOLO
import os

def main():
  # Verificamos que estamos en el directorio correcto
  print(f"Directorio de trabajo: {os.getcwd()}")

  # Cargar modelo base (nano)
  model = YOLO('yolov8n.pt')

  # Entrenar
  results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='modelo_placas_linux',
    device=0,  # Usa 0 para GPU Nvidia. Si usas CPU, pon 'cpu'
    workers=4  # En Linux podemos usar workers para cargar datos más rápido
  )

  # Validar
  metrics = model.val()
  print(f"mAP50: {metrics.box.map50}")

if _name_ == '_main_':
  main()