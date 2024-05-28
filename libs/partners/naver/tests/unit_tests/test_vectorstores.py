from langchain_naver.vectorstores import NaverVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    NaverVectorStore()
