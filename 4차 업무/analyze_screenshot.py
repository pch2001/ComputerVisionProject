import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from attack_decision import should_attack

DEFAULT_MODEL = Path(r"D:\AI\Comment\ComputerVisionProject\3차 업무\runs\train\exp-4\weights\best.pt")


def analyze(image_path, model_path, output_path):
    if not image_path.is_file() or not model_path.is_file():
        missing = image_path if not image_path.is_file() else model_path
        raise ValueError(f"파일을 찾을 수 없습니다. {missing}")
    if output_path.resolve() in (image_path.resolve(), model_path.resolve()):
        raise ValueError("출력 파일에는 이미지와 모델의 경로를 사용할 수 없습니다.")

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다. {image_path}")

    result = YOLO(str(model_path)).predict(image, device="cpu", conf=0.25, verbose=False)[0]
    counts = Counter(result.names[int(index)] for index in result.boxes.cls.tolist())
    success, encoded = cv2.imencode(".png", result.plot())
    if not success:
        raise ValueError("결과 이미지를 저장할 수 없습니다.")
    encoded.tofile(output_path)
    return counts


def main():
    parser = argparse.ArgumentParser(description="마을 스크린샷으로 게임 내 공격 조건을 판단합니다.")
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("detection_result.png"))
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("출력 파일 확장자는 .png로 지정하세요.")

    try:
        image_path = args.image
        if image_path is None:
            entered = input("스크린샷 파일을 이 창에 끌어다 놓고 Enter를 누르세요. ").strip().strip('"')
            if not entered:
                parser.error("스크린샷 파일 경로를 입력하세요.")
            image_path = Path(entered)
        counts = analyze(image_path, args.model, args.output)
        building_count = sum(counts.values())
        print("이미지 분석을 마쳤습니다.")
        troop_count = int(input("내 전투 병력 수를 입력하세요. "))
        eligible = should_attack(building_count, troop_count)
        print(f"상대 전력 · 탐지된 건물 {building_count}개")
        print("건물별 개수", ", ".join(f"{name} {count}개" for name, count in sorted(counts.items())) or "없음")
        print(f"결과 이미지 {args.output.resolve()}")
        print(f"비교 결과 · 상대 건물 {building_count}개 / 내 병력 {troop_count}명")
        print("공격 조건 충족" if eligible else "공격 보류")

    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"분석을 마치지 못했습니다. {error}\n")


if __name__ == "__main__":
    main()

