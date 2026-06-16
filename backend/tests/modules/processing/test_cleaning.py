from app.modules.processing.cleaning import clean


def test_clean_collapses_horizontal_whitespace():
    assert clean("This  has   extra    spaces.") == "This has extra spaces."


def test_clean_drops_digit_only_noise_lines():
    text = "Invoice 2024\n1234567890\nThank you for your purchase."
    result = clean(text)
    assert "Thank you for your purchase." in result
    assert "1234567890" not in result


def test_clean_drops_symbol_heavy_separator_lines():
    text = "Header\n---+---+---\nBody text here."
    result = clean(text)
    assert "Header" in result
    assert "Body text here." in result
    assert "---+---+---" not in result


def test_clean_keeps_sentences_with_inline_numbers():
    # Contains digits and punctuation but still has enough letters -- must survive.
    sentence = "Invoice #2024-001, due 30.06.2024, total EUR 1,234.56."
    assert sentence in clean(sentence)


def test_clean_caps_excess_blank_lines():
    text = "Para one.\n\n\n\nPara two."
    assert clean(text) == "Para one.\n\nPara two."


def test_clean_handles_german_text():
    text = "Sehr geehrte Damen und Herren, vielen Dank für Ihre Anfrage."
    assert clean(text) == text


def test_clean_handles_chinese_text():
    text = "感謝您的來信，我們將盡快回覆您的問題。"
    assert clean(text) == text


def test_clean_strips_leading_trailing_whitespace():
    assert clean("  Hello world.  ") == "Hello world."
