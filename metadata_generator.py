import os
import PyPDF2
import docx
import pytesseract
from PIL import Image
from datetime import datetime
from collections import Counter
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
from pdf2image import convert_from_path
import fitz  # PyMuPDF - better alternative to PyPDF2
from langdetect import detect, LangDetectException
from textblob import TextBlob
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

# Configure Tesseract path (adjust this to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Check OCR availability
try:
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

def extract_text_from_pdf_pymupdf(file_path):
    text = ""
    extraction_log = []
    try:
        doc = fitz.open(file_path)
        extraction_log.append(f"Opened PDF with {len(doc)} pages using PyMuPDF")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n[Page {page_num + 1} Text]\n{page_text}"
                extraction_log.append(f"Extracted text from page {page_num + 1}")
            else:
                extraction_log.append(f"No text found in page {page_num + 1}")
        doc.close()
    except Exception as e:
        extraction_log.append(f"PyMuPDF error: {str(e)}")
        return "", extraction_log
    return text, extraction_log

def extract_text_from_pdf_pypdf2(file_path):
    text = ""
    extraction_log = []
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            extraction_log.append(f"Opened PDF with {len(reader.pages)} pages using PyPDF2")
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                    extraction_log.append("Decrypted PDF with empty password")
                except Exception as e:
                    extraction_log.append(f"Could not decrypt PDF: {str(e)}")
                    return "", extraction_log
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"\n[Page {i+1} Text]\n{page_text}"
                        extraction_log.append(f"Extracted text from page {i+1}")
                    else:
                        extraction_log.append(f"No text found in page {i+1}")
                except Exception as e:
                    extraction_log.append(f"Page {i+1} error: {str(e)}")
    except Exception as e:
        extraction_log.append(f"PyPDF2 error: {str(e)}")
        return "", extraction_log
    return text, extraction_log

def extract_text_from_pdf_ocr(file_path):
    text = ""
    extraction_log = []
    if not OCR_AVAILABLE:
        extraction_log.append("OCR not available - Tesseract not configured")
        return "", extraction_log
    try:
        images = convert_from_path(file_path, dpi=200, thread_count=2)
        extraction_log.append(f"Converted {len(images)} pages to images for OCR")
        for i, img in enumerate(images):
            try:
                custom_config = r'--oem 3 --psm 6'
                page_text = pytesseract.image_to_string(img, config=custom_config)
                if page_text.strip():
                    text += f"\n[Page {i+1} OCR Text]\n{page_text}"
                    extraction_log.append(f"OCR processed page {i+1}")
                else:
                    extraction_log.append(f"No OCR text found on page {i+1}")
            except Exception as e:
                extraction_log.append(f"OCR failed for page {i+1}: {str(e)}")
    except Exception as e:
        extraction_log.append(f"PDF to image conversion failed: {str(e)}")
        return "", extraction_log
    return text, extraction_log

def extract_text_from_pdf(file_path):
    extraction_log = []
    try:
        text, log = extract_text_from_pdf_pymupdf(file_path)
        extraction_log.extend(log)
        if text.strip():
            extraction_log.append("Successfully extracted text using PyMuPDF")
            return text, extraction_log
    except Exception as e:
        extraction_log.append(f"PyMuPDF not available: {str(e)}")
    text, log = extract_text_from_pdf_pypdf2(file_path)
    extraction_log.extend(log)
    if text.strip():
        extraction_log.append("Successfully extracted text using PyPDF2")
        return text, extraction_log
    extraction_log.append("No text found with standard methods, trying OCR...")
    text, log = extract_text_from_pdf_ocr(file_path)
    extraction_log.extend(log)
    if text.strip():
        extraction_log.append("Successfully extracted text using OCR")
        return text, extraction_log
    else:
        extraction_log.append("All extraction methods failed")
        return "No text could be extracted from this PDF", extraction_log

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text, ["DOCX extracted successfully"]
    except Exception as e:
        return f"DOCX extraction error: {e}", [f"DOCX error: {str(e)}"]

def extract_text_from_image(file_path):
    try:
        if not OCR_AVAILABLE:
            return "", ["OCR not available - Tesseract not configured"]
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        if text.strip():
            return text, ["Image OCR successful"]
        else:
            return "", ["Image OCR completed but no text found"]
    except Exception as e:
        return f"Image extraction error: {e}", [f"Image error: {str(e)}"]

def extract_keywords(text, num_keywords=10):
    try:
        if not text or not isinstance(text, str):
            return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalpha() and len(word) > 2 and word not in stop_words]
        if not words:
            return []
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(num_keywords)]
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return []

def extract_summary(text, num_sentences=3):
    try:
        if not text or not isinstance(text, str):
            return "No text available for summary"
        text = re.sub(r'\[Page \d+ .*?\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return "No meaningful sentences found"
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        summary = [sentences[0]]
        if len(sentences) > 2:
            summary.append(sentences[len(sentences)//2])
        if len(sentences) > 1:
            summary.append(sentences[-1])
        return ' '.join(summary[:num_sentences])
    except Exception as e:
        print(f"Summary extraction error: {e}")
        return "Summary generation failed"

def classify_content(text):
    try:
        if not text or not isinstance(text, str):
            return "Unknown"
        text_lower = text.lower()
        legal_keywords = ['contract', 'agreement', 'clause', 'terms', 'conditions', 'legal', 'whereas']
        report_keywords = ['report', 'analysis', 'findings', 'conclusion', 'executive summary', 'methodology']
        resume_keywords = ['resume', 'cv', 'curriculum vitae', 'experience', 'education', 'skills']
        financial_keywords = ['invoice', 'receipt', 'payment', 'amount', 'total', 'tax', 'billing']
        academic_keywords = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'references']
        if any(word in text_lower for word in legal_keywords):
            return "Legal Document"
        elif any(word in text_lower for word in academic_keywords):
            return "Academic Document"
        elif any(word in text_lower for word in report_keywords):
            return "Report"
        elif any(word in text_lower for word in resume_keywords):
            return "Resume/CV"
        elif any(word in text_lower for word in financial_keywords):
            return "Financial Document"
        else:
            return "General Document"
    except Exception as e:
        print(f"Content classification error: {e}")
        return "Unknown"

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'Unknown'

def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception:
        return 'Unknown'

def extract_key_information(text):
    """Extract entities, keywords, and important sections"""
    entities = {
        'PERSON': [],
        'ORG': [],
        'PLACES': []
    }
    key_phrases = []
    
    try:
        if 'nlp' in globals() and nlp is not None:
            doc = nlp(text)
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['PERSON'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['ORG'].append(ent.text)
                elif ent.label_ == 'GPE':
                    entities['PLACES'].append(ent.text)
            # Extract key phrases using noun chunks
            key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
        else:
            # Fallback: simple regex-based extraction
            import re
            # Extract potential names (capitalized words)
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            entities['PERSON'] = list(set(names))[:10]
            # Extract organizations (simple pattern: capitalized words followed by Inc, Ltd, etc.)
            orgs = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(Inc|Ltd|Corporation|Corp|LLC)\b', text)
            entities['ORG'] = list(set(orgs))[:10]
            # Extract places (simple pattern: common city/country names or capitalized words)
            places = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
            entities['PLACES'] = list(set(places))[:10]
            # Simple key phrase extraction (common noun phrases)
            words = text.split()
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            key_phrases = list(set(bigrams))[:10]
    except Exception as e:
        print(f"Entity extraction failed: {e}")
    return entities, key_phrases

def improved_classify_content(text):
    categories = {
        'Finance': ['invoice', 'payment', 'tax', 'amount', 'total', 'billing', 'account', 'balance', 'statement', 'bank'],
        'Education': ['university', 'school', 'student', 'teacher', 'course', 'curriculum', 'exam', 'degree', 'academic'],
        'Legal': ['contract', 'agreement', 'clause', 'terms', 'conditions', 'legal', 'whereas', 'law', 'court'],
        'Medical': ['patient', 'doctor', 'medicine', 'treatment', 'diagnosis', 'hospital', 'clinical'],
        'Technology': ['software', 'hardware', 'computer', 'technology', 'system', 'application', 'device'],
        'Report': ['report', 'analysis', 'findings', 'conclusion', 'summary', 'methodology'],
        'Resume/CV': ['resume', 'cv', 'curriculum vitae', 'experience', 'education', 'skills'],
        'Academic': ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'references'],
    }
    text_lower = text.lower() if text else ''
    for category, keywords in categories.items():
        if any(word in text_lower for word in keywords):
            return category
    return 'General'

def generate_metadata(file_path, original_filename=None):
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    extraction_log = []
    if ext == '.pdf':
        text, log = extract_text_from_pdf(file_path)
        extraction_log.extend(log)
    elif ext == '.docx':
        text, log = extract_text_from_docx(file_path)
        extraction_log.extend(log)
    elif ext in ('.png', '.jpg', '.jpeg'):
        text, log = extract_text_from_image(file_path)
        extraction_log.extend(log)
    elif ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            extraction_log.append("TXT file read successfully")
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                extraction_log.append("TXT file read with latin-1 encoding")
            except Exception as e:
                text = ""
                extraction_log.append(f"TXT read error: {str(e)}")
        except Exception as e:
            text = ""
            extraction_log.append(f"TXT read error: {str(e)}")
    else:
        extraction_log.append(f"Unsupported file type: {ext}")
    try:
        file_stats = os.stat(file_path)
        basic_info = {
            "filename": original_filename or os.path.basename(file_path),
            "file_type": ext[1:].upper() if ext else 'Unknown',
            "file_size": f"{file_stats.st_size / 1024:.2f} KB",
            "processing_date": datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
        }
        content_analysis = {
            "language": detect_language(text) if text else 'Unknown',
            "document_type": improved_classify_content(text) if text else 'Unknown',
            "word_count": len(text.split()) if text and isinstance(text, str) else 0,
            "character_count": len(text) if text and isinstance(text, str) else 0,
            "line_count": text.count('\n') + 1 if text and isinstance(text, str) else 0,
            "readability_score": None,
            "sentiment": analyze_sentiment(text) if text else 'Unknown',
            "category": improved_classify_content(text) if text else 'Unknown',
            "content type": classify_content(text) if text else "Unknown",
        }
        entities, key_phrases = extract_key_information(text) if text else ({}, [])
        semantic_data = {
            "summary": extract_summary(text) if text else "No text extracted",
            "key_topics": extract_keywords(text) if text else [],
            "key_phrases": key_phrases,
            "entities": entities,
            "entraction log": extraction_log,
            "text preview": text[:500] + "..." if text and len(text) > 500 else text or "No text extracted"
        }
        return {
            "basic_info": basic_info,
            "content_analysis": content_analysis,
            "semantic_data": semantic_data
        }
    except Exception as e:
        return {
            'error': f"Metadata generation failed: {str(e)}",
            'extraction_log': extraction_log
        }