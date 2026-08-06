import pytest
from utils.text_cleaner import TextCleaner
from utils.pdf_parser import PDFParser
from utils.docx_parser import DocxParser

def test_text_cleaning():
    cleaner = TextCleaner()
    
    # Test lowercase convert
    assert cleaner.clean_text("Hello WORLD") == "hello world"
    
    # Test URL removal
    assert cleaner.clean_text("Visit https://google.com for info") == "visit for info"
    
    # Test Email removal
    assert cleaner.clean_text("Contact me at user@example.com now") == "contact me at now"
    
    # Test special characters removal (keep + and # for tech stacks)
    assert cleaner.clean_text("C++ & C# are cool!") == "c++ c# are cool"

def test_tokenization():
    cleaner = TextCleaner()
    text = "The quick brown fox jumps over the lazy dog"
    tokens = cleaner.tokenize(text, remove_stopwords=True)
    
    # Ensure standard stopwords like "the", "over" are removed
    assert "the" not in tokens
    assert "over" not in tokens
    assert "quick" in tokens

def test_missing_files_handling():
    # Verify parsers return error dictionary instead of crashing
    pdf_res = PDFParser.parse("non_existent_file.pdf")
    assert pdf_res["success"] is False
    assert "does not exist" in pdf_res["error"]
    
    docx_res = DocxParser.parse("non_existent_file.docx")
    assert docx_res["success"] is False
    assert "does not exist" in docx_res["error"]
