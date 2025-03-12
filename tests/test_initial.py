import pytest

def test_positive():
    assert 1 == 1
    
def test_negative():
    with pytest.raises(AssertionError):
        assert 1 == 2

@pytest.mark.xfail
def test_fail():
    assert 1 == 2