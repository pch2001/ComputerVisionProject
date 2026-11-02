import cv2
import numpy as np
from pathlib import Path

# 이상치 판단 기준: 사진에 맞게 조정
MIN_BRIGHTNESS = 40
MIN_OBJECT_RATIO = 0.10

base_dir = Path(__file__).resolve().parent
input_dir = base_dir / "original_samples"

# 이전 전처리 결과와 구분하기 위한 저장 위치
output_dir = base_dir / "preprocessed_samples_filtered"

for folder in ("크기 조정", "색상 변환", "노이즈 제거", "데이터 증강"):
    (output_dir / folder).mkdir(parents=True, exist_ok=True)


# 한글 경로에 이미지 저장
def save_image(folder, name, image):
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"이미지 저장 실패: {name}")
    data.tofile(str(output_dir / folder / f"{name}.png"))


# 이미지에서 전경이 차지하는 면적 비율 추정
def estimate_object_ratio(image):
    height, width = image.shape[:2]
    margin_x = max(1, int(width * 0.05))
    margin_y = max(1, int(height * 0.05))

    # 사진 가장자리 5%를 배경으로 가정
    rect = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y,
    )

    if rect[2] <= 0 or rect[3] <= 0:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image, mask, rect, background, foreground,
            5, cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None

    object_mask = (
        (mask == cv2.GC_FGD)
        | (mask == cv2.GC_PR_FGD)
    )

    # 전경 검출 실패는 별도 검토 대상으로 처리
    if not object_mask.any():
        return None

    return np.count_nonzero(object_mask) / object_mask.size


image_paths = sorted(
    p for p in input_dir.glob("*")
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

if not image_paths:
    raise FileNotFoundError(f"이미지가 없습니다: {input_dir}")

for path in image_paths:
    image = cv2.imread(str(path))

    if image is None:
        print(f"[검토] 이미지 읽기 실패: {path.name}")
        continue

    # 1. 너무 어두운 이미지 제외
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_original))

    if brightness < MIN_BRIGHTNESS:
        print(f"[제외] {path.name}: 평균 밝기 {brightness:.1f}")
        continue

    # 2. 객체가 너무 작은 이미지 제외
    object_ratio = estimate_object_ratio(image)

    if object_ratio is None:
        print(f"[검토] {path.name}: 객체 영역 추정 실패")
        continue

    if object_ratio < MIN_OBJECT_RATIO:
        print(f"[제외] {path.name}: 추정 객체 비율 {object_ratio:.1%}")
        continue

    # 3. 크기 조정
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    save_image("크기 조정", path.stem, resized)

    # 4. 색상 변화용 이미지 생성
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255)
    color_changed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 5. 데이터 증강
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

        # 6. 회색조 변환 → 노이즈 제거 → 정규화
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

    print(
        f"[완료] {path.name}: "
        f"밝기 {brightness:.1f}, 추정 객체 비율 {object_ratio:.1%}"
    )