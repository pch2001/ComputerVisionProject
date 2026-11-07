from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # 사전 학습 모델에서 추가 학습 시작
    model = YOLO(str(base_dir / "yolov8n.pt"))

    # 모델 학습
    model.train(
        data=str(base_dir / "datasets" / "data.yaml"),
        epochs=10,               # 학습 데이터 전체를 10번 반복
        imgsz=640,               # 학습 입력 크기
        batch=4,                 # 한 번에 처리할 이미지 수
        device="cpu",            # 현재 설치된 PyTorch에 맞춤
        workers=0,
        project=str(base_dir / "runs" / "train"),
        name="exp",
    )

    print("학습한 모델 저장 위치:", model.trainer.best)