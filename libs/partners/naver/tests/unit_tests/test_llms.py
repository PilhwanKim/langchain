"""Test Naver Chat API wrapper."""
from langchain_naver import NaverLLM


def test_initialization() -> None:
    """Test integration initialization."""
    NaverLLM()
