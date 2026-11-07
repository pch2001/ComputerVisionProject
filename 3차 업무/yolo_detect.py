import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

base_dir = Path(__file__).resolve().parent

# 1. 직접 학습한 모델 불러오기
model = YOLO(
    str(base_dir / "runs" / "train" / "exp-4" / "weights" / "best.pt")
)

# 2. 검증 폴더의 첫 번째 사진 선택
image_dir = base_dir / "datasets" / "valid" / "images"
image_paths = sorted(image_dir.glob("*.jpg"))

if not image_paths:
    raise FileNotFoundError(f"사진이 없습니다: {image_dir}")

image_path = image_paths[0]

# 한글 경로에서도 이미지 읽기
image = cv2.imdecode(
    np.fromfile(str(image_path), dtype=np.uint8),
    cv2.IMREAD_COLOR,
)

if image is None:
    raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

# 3. 객체 탐지
results = model(image, device="cpu", conf=0.25)

# 4. 탐지된 객체에 사각형·이름·신뢰도 표시
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            f"{label} ({confidence:.2f})",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

# 5. 한글 경로에 결과 저장
output_path = base_dir / "clash_detection_result.jpg"
success, data = cv2.imencode(".jpg", image)

if not success:
    raise ValueError("결과 이미지 저장 실패")

data.tofile(str(output_path))

print("사용한 사진:", image_path.name)
print("탐지된 객체 수:", len(results[0].boxes))
print("결과 저장 위치:", output_path)

# 6. 결과 화면 표시
cv2.namedWindow("YOLO Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Object Detection", 900, 900)
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()