# RecursiveCharacterTextSplitter 递归的将文本分割成指定大小的块。它的核心思想是根据一组分隔符(separators)逐步分割文本，
# 直到每个块的大小都符合预设的 chunk_size。如果某个块仍然过大，它会继续递归地分割，直到满足条件为止。
#   separators：一个字符串列表，表示用于分割文本的分隔符。默认值为 ["\n\n", "\n", " ", ""]，即首先按照段落分割，然后
#              按照行分割，最后按空格分割，如果都不行则按照字符分割。
#   chunk_size：每个块的最大大小(以字符数为单位)。默认值为 100。
#   chunk_overlap：块与块之间的重叠部分的大小。默认值为 20。
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List, Optional

class MarkdownSplitter:
    # chunk_overlap: chunk 间重叠的文本大小
    def __init__(self, markdown_text: str, chunk_size: int = 150, chunk_overlap: int = 50):
        self.markdown_text = markdown_text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 定义标题层级 (支持三级标题)
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]

    def split_by_headers(self):
        # 使用 MarkdownHeaderTextSplitter 进行按标题切分
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=True  # 保留标题
        )
        md_header_splits = markdown_splitter.split_text(self.markdown_text)
        return md_header_splits

    def split_by_characters(self, text_blocks):
        # 使用 RecursiveCharacterTextSplitter 按字符数进一步切分每个块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(text_blocks)
        return splits

    def split(self):
        # 先按标题切分，再按字符切分
        header_splits = self.split_by_headers()
        final_splits = self.split_by_characters(header_splits)
        return final_splits



class Chunker:
    """A modular text chunking class that splits text into smaller, overlapping segments.

    This class provides a flexible way to break down large texts into smaller chunks
    while maintaining context through configurable overlap. It uses RecursiveCharacterTextSplitter
    from langchain under the hood.

    Attributes:
        chunk_size (int): The target size for each text chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        separators (List[str]): List of separators to use for splitting, in order of preference.
        length_function (callable): Function to measure text length (default: len).
    """

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 128,
            separators: Optional[List[str]] = None,
            length_function: callable = len
    ):
        """Initialize the Chunker with specified parameters.

        Args:
            chunk_size (int, optional): Target size for each chunk. Defaults to 512.
            chunk_overlap (int, optional): Number of characters to overlap. Defaults to 128.
            separators (List[str], optional): Custom separators for splitting.
                Defaults to ["\n\n", "\n"].
            length_function (callable, optional): Function to measure text length.
                Defaults to len.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n"]
        self.length_function = length_function

        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function
        )

    def split_text(self, text: str) -> List[str]:
        """Split a single text into chunks.

        Args:
            text (str): The input text to be split into chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        return self.splitter.split_text(text)

    def split_texts(self, texts: List[str]) -> List[List[str]]:
        """Split multiple texts into chunks.

        Args:
            texts (List[str]): A list of input texts to be split into chunks.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains
                the chunks for one input text.
        """
        return [self.split_text(text) for text in texts]


if __name__ == '__main__':

    # 示例的 Markdown 文本（从 HTML 转换而来）
    markdown_document = """
    # Intro
    
    ## History
    Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] 
    
    Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. 
    
    ## Rise and divergence
    As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. 
    
    ### Standardization
    From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. 
    
    ## Implementations
    Implementations of Markdown are available for over a dozen programming languages.
    """

    # 使用 MarkdownSplitter 类
    splitter = MarkdownSplitter(markdown_document)
    final_splits = splitter.split()

    # 打印最终的分割结果
    print("\nFinal Text Splits:")
    for idx, split in enumerate(final_splits):
        print(f"Chunk {idx + 1}:\n{split}\n")
