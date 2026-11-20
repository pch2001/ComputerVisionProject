def should_attack(building_count: int, troop_count: int) -> bool:
    """Return whether the troop count exceeds the detected building count."""
    if building_count < 0 or troop_count < 0:
        raise ValueError("건물 수와 병력 수는 0 이상의 정수여야 합니다.")
    return troop_count > building_count
