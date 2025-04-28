from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


def chunk_paragraphs(
    text: str,
    splitter: TextSplitter | None = None,
) -> list[str]:
    if splitter is None:
        splitter = RecursiveCharacterTextSplitter()
    return splitter.split_text(text=text)
