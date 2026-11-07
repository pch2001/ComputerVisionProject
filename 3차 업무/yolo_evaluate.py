from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # 학습이 완료된 모델 불러오기
    model = YOLO(
        str(base_dir / "runs" / "train" / "exp-4" / "weights" / "best.pt")
    )

    # 테스트용 이미지와 정답 라벨로 성능 평가
    metrics = model.val(
        data=str(base_dir / "datasets" / "data.yaml"),
        split="test",
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(base_dir / "runs" / "evaluate"),
        name="test",
    )

    print(f"평균 정밀도: {metrics.box.mp:.1%}")
    print(f"평균 재현율: {metrics.box.mr:.1%}")
    print(f"mAP50: {metrics.box.map50:.1%}")
    print(f"mAP50-95: {metrics.box.map:.1%}")