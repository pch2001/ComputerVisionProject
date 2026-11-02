import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent
input_dir = base_dir / "original_samples"
output_dir = base_dir / "preprocessed_samples"

# 결과 폴더 생성
for folder in ("크기 조정", "색상 변환", "노이즈 제거", "데이터 증강"):
    (output_dir / folder).mkdir(parents=True, exist_ok=True)


# 한글 경로에 PNG 저장
def save_image(folder, name, image):
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"이미지 저장 실패: {name}")
    data.tofile(str(output_dir / folder / f"{name}.png"))


image_paths = sorted(
    p for p in input_dir.glob("*")
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

if not image_paths:
    raise FileNotFoundError(f"이미지가 없습니다: {input_dir}")

for path in image_paths:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"이미지 읽기 실패: {path.name}")

    # 1. 크기 조정
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    save_image("크기 조정", path.stem, resized)

    # 2. 색상 변화용 이미지 생성
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255)
    color_changed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. 기본 이미지와 증강 이미지 준비
    matrix = cv2.getRotationMatrix2D((111.5, 111.5), 15, 1.0)
    variants = {
        "original": resized,
        "flip": cv2.flip(resized, 1),
        "rotate": cv2.warpAffine(
            resized, matrix, (224, 224),
            borderMode=cv2.BORDER_REFLECT_101,
        ),
        "color": color_changed,
    }

    for kind, picture in variants.items():
        name = path.stem if kind == "original" else f"{path.stem}_{kind}"

        # 4. 회색조 변환 → 노이즈 제거 → 정규화
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        normalized = blurred.astype(np.float32) / 255.0

        if kind == "original":
            save_image("색상 변환", f"{name}_gray", gray)
            save_image("노이즈 제거", f"{name}_blur", blurred)
        else:
            save_image("데이터 증강", name, picture)

        np.save(
            output_dir / "색상 변환" / f"{name}_normalized.npy",
            normalized,
        )

    print(f"완료: {path.name}")