"""
OCR processing utilities for ablation experiments.
Includes cleaning, classification, and category-aware summarization.
"""

import re


def clean_ocr_text(ocr_tokens, max_tokens=15, min_token_length=2):
    """
    Clean and filter OCR tokens to reduce noise.
    
    Args:
        ocr_tokens: List of OCR token strings
        max_tokens: Maximum number of tokens to include
        min_token_length: Minimum token length to keep (filters single chars, noise)
    
    Returns:
        Cleaned OCR text string
    """
    if not ocr_tokens:
        return ""
    
    # Filter tokens: remove very short tokens (likely noise) and duplicates
    cleaned = []
    seen = set()
    
    for token in ocr_tokens:
        # Normalize: strip whitespace
        token = token.strip()
        
        # Skip empty tokens
        if not token:
            continue
        
        # Skip very short tokens (likely OCR noise)
        if len(token) < min_token_length:
            continue
        
        # Skip tokens that are mostly non-alphanumeric (likely noise)
        alnum_ratio = sum(c.isalnum() for c in token) / len(token) if token else 0
        if alnum_ratio < 0.3:  # Less than 30% alphanumeric
            continue
        
        # Normalize case for deduplication (but keep original)
        token_lower = token.lower()
        if token_lower not in seen:
            seen.add(token_lower)
            cleaned.append(token)
        
        # Limit total tokens
        if len(cleaned) >= max_tokens:
            break
    
    return " ".join(cleaned)


def classify_question(question):
    """
    Classify question type based on keyword heuristics.
    
    Args:
        question: Question string
        
    Returns:
        Question type string: "brand", "text", "number", "date", "time", or "general"
    """
    question_lower = question.lower()
    
    # Brand questions
    brand_keywords = ["brand", "logo", "label", "company", "manufacturer", "maker"]
    if any(keyword in question_lower for keyword in brand_keywords):
        return "brand"
    
    # Number/Price questions
    number_keywords = ["number", "percent", "%", "price", "$", "cost", "amount", "how much", "how many"]
    if any(keyword in question_lower for keyword in number_keywords):
        return "number"
    
    # Date/Year questions
    date_keywords = ["year", "date", "born", "since", "when"]
    if any(keyword in question_lower for keyword in date_keywords):
        return "date"
    
    # Time questions
    time_keywords = ["time", "clock", "what time"]
    if any(keyword in question_lower for keyword in time_keywords):
        return "time"
    
    # Text reading questions
    text_keywords = ["text", "say", "spell", "read", "what does", "what is written"]
    if any(keyword in question_lower for keyword in text_keywords):
        return "text"
    
    # Default fallback
    return "general"


def summarize_ocr_by_type(ocr_tokens, q_type, max_tokens=15, min_token_length=2):
    """
    Summarize OCR tokens based on question category.
    
    Args:
        ocr_tokens: List of OCR token strings
        q_type: Question type string ("brand", "number", "date", "time", "text", "general")
        max_tokens: Maximum number of tokens to include
        min_token_length: Minimum token length to keep
        
    Returns:
        Filtered OCR summary string
    """
    if not ocr_tokens:
        return ""
    
    filtered_tokens = []
    
    for token in ocr_tokens:
        token = token.strip()
        if not token or len(token) < min_token_length:
            continue
        
        # Skip tokens that are mostly non-alphanumeric (likely noise)
        alnum_ratio = sum(c.isalnum() for c in token) / len(token) if token else 0
        if alnum_ratio < 0.3:
            continue
        
        # Category-specific filtering
        if q_type == "brand":
            # Keep capitalized words only (likely brand names)
            if token[0].isupper() and token.isalpha():
                filtered_tokens.append(token)
        
        elif q_type == "number":
            # Keep numbers, $, %, decimals
            if any(char.isdigit() for char in token) or "$" in token or "%" in token:
                filtered_tokens.append(token)
        
        elif q_type == "date":
            # Keep 4-digit numbers or date-like tokens
            if re.match(r'^\d{4}$', token) or re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', token):
                filtered_tokens.append(token)
            elif token.isdigit() and len(token) >= 4:
                filtered_tokens.append(token)
        
        elif q_type == "time":
            # Keep time-like patterns (e.g., "10:30", "5:41")
            if re.match(r'^\d{1,2}[:]\d{2}', token):
                filtered_tokens.append(token)
            elif any(char.isdigit() for char in token) and len(token) <= 5:
                filtered_tokens.append(token)
        
        elif q_type == "text":
            # Keep all cleaned OCR tokens
            filtered_tokens.append(token)
        
        else:  # general fallback
            filtered_tokens.append(token)
        
        if len(filtered_tokens) >= max_tokens:
            break
    
    return " ".join(filtered_tokens) if filtered_tokens else ""


# Predefined prompt templates for OCR ablations
PROMPT_TEMPLATES = {
    # No OCR baseline
    "no_ocr": "Question: {question} Answer:",
    
    # Basic OCR (simple dump)
    "basic_ocr": "The image contains text: {ocr_text}. Question: {question} Answer:",
    
    # Structured OCR (category-aware)
    "structured_ocr": (
        "Relevant text in the image ({q_type}): {ocr_summary}\n"
        "Question: {question}\n"
        "Use ONLY information from the text above when answering.\n"
        "Answer:"
    ),
    
    # Additional prompt engineering variants
    "default": "Question: {question} Answer:",
    "descriptive": "Based on the image, answer the question: {question}",
    "instruction": "Look at the image and answer: {question}",
    "direct": "{question}",
    "text_focus": "What is the text in the image? {question}",
    "ocr_hint_v2": "Question: {question} (Note: The image contains text: {ocr_text}) Answer:",
    "ocr_hint_v3": "Answer this question about the image: {question}\nVisible text in image: {ocr_text}\nAnswer:",
}


def format_prompt(
    question, 
    template_name="no_ocr", 
    ocr_tokens=None, 
    q_type=None
):
    """
    Format a question with the specified prompt template.
    
    Args:
        question: Question string
        template_name: Name of prompt template to use
        ocr_tokens: Optional OCR tokens for OCR-enhanced templates
        q_type: Optional question type for structured OCR
        
    Returns:
        Formatted question string
    """
    if template_name not in PROMPT_TEMPLATES:
        # Default to no_ocr if template not found
        template_name = "no_ocr"
    
    template = PROMPT_TEMPLATES[template_name]
    
    # Handle different template types
    if template_name == "no_ocr" or template_name == "default" or template_name == "direct":
        return template.format(question=question)
    
    elif template_name == "structured_ocr":
        # Need both q_type and ocr_tokens
        if not q_type:
            q_type = classify_question(question)
        
        if ocr_tokens:
            ocr_summary = summarize_ocr_by_type(ocr_tokens, q_type)
        else:
            ocr_summary = ""
        
        if ocr_summary:
            return template.format(
                question=question,
                ocr_summary=ocr_summary,
                q_type=q_type
            )
        else:
            # Fallback to no OCR
            return PROMPT_TEMPLATES["no_ocr"].format(question=question)
    
    elif "{ocr_text}" in template:
        # Basic OCR or other OCR-enhanced templates
        if ocr_tokens:
            ocr_text = clean_ocr_text(ocr_tokens)
        else:
            ocr_text = ""
        
        if ocr_text:
            return template.format(question=question, ocr_text=ocr_text)
        else:
            # Fallback to no OCR
            return PROMPT_TEMPLATES["no_ocr"].format(question=question)
    
    else:
        # Simple templates without OCR
        return template.format(question=question)


if __name__ == "__main__":
    # Test OCR utilities
    print("Testing OCR utilities...")
    
    # Test data
    test_tokens = ["COCA", "COLA", "2.99", "$", "COLD", "123", "noise!!!", "a", "b"]
    test_question = "What brand is this drink?"
    
    print(f"\nTest tokens: {test_tokens}")
    print(f"Test question: {test_question}")
    
    # Test cleaning
    cleaned = clean_ocr_text(test_tokens)
    print(f"\nCleaned OCR: {cleaned}")
    
    # Test classification
    q_type = classify_question(test_question)
    print(f"Question type: {q_type}")
    
    # Test category-aware summarization
    summary = summarize_ocr_by_type(test_tokens, q_type)
    print(f"Category-aware summary: {summary}")
    
    # Test prompt formatting
    print("\n--- Prompt Template Tests ---")
    for template_name in ["no_ocr", "basic_ocr", "structured_ocr"]:
        formatted = format_prompt(test_question, template_name, test_tokens, q_type)
        print(f"\n{template_name}:")
        print(formatted)
    
    print("\n✓ OCR utilities test passed!")
