from pathlib import Path
from datasets import load_dataset

# 이 Python 파일이 있는 위치에 원본 이미지 폴더 생성
output_dir = Path(__file__).resolve().parent / "original_samples"
output_dir.mkdir(exist_ok=True)

print("데이터셋에 연결 중입니다...", flush=True)

dataset = load_dataset(
    "ethz/food101",
    split="train",
    streaming=True,
)

# 앞에서부터 이미지 5장을 가져와 저장
for index, sample in enumerate(dataset.take(5), start=1):
    image = sample["image"].convert("RGB")
    output_path = output_dir / f"food_{index:02d}.jpg"

    image.save(output_path, format="JPEG")
    print(f"저장 완료: {output_path}", flush=True)

print("원본 이미지 5장 저장이 완료되었습니다.")