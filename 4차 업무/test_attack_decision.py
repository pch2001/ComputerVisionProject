import pytest

from attack_decision import should_attack


@pytest.mark.parametrize(
    ("buildings", "troops", "expected"),
    [(12, 10, False), (10, 10, False), (8, 10, True), (0, 10, True), (0, 0, False)],
)
def test_attack_decision(buildings, troops, expected):
    assert should_attack(buildings, troops) is expected


@pytest.mark.parametrize(("buildings", "troops"), [(-1, 10), (10, -1)])
def test_negative_counts(buildings, troops):
    with pytest.raises(ValueError):
        should_attack(buildings, troops)
