import numpy as np
import pytest
import cv2


# 가짜 깊이 맵 생성
def generate_depth_map(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)


# 테스트 1: 정상 입력의 출력 크기와 자료형 검사
def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(image)

    assert depth_map.shape == image.shape, "출력 크기가 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 자료형이 다릅니다."


# 테스트 2: 이미지가 없을 때 예외가 발생하는지 검사
def test_generate_depth_map_without_image():
    with pytest.raises(ValueError):
        generate_depth_map(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])