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
import textstat
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

        import re
        text_lower = text.lower()
        categories = {
            "Legal Document": ['contract', 'agreement', 'clause', 'terms', 'conditions', 'legal', 'whereas'],
            "Academic Document": ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'references'],
            "Report": ['report', 'analysis', 'findings', 'conclusion', 'executive summary', 'methodology'],
            "Resume/CV": ['resume', 'cv', 'curriculum vitae', 'experience', 'skills'],
            "Financial Document": ['invoice', 'receipt', 'payment', 'amount', 'total', 'tax', 'billing'],
        }

        scores = {
            category: sum(1 for word in keywords if re.search(r'\b' + re.escape(word) + r'\b', text_lower))
            for category, keywords in categories.items()
        }


        best_match = max(scores, key=scores.get)
        if scores[best_match] > 1:
            return best_match
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
    """Extract unique entities (PERSON, ORG, PLACES) with improved filtering for academic documents."""
    import string
    entities = {
        'PERSON': [],
        'ORG': [],
        'PLACES': []
    }
    # Custom ignore lists and stopwords for academic/section headers
    ignore_words = set([
        'Types', 'Meditation', 'Prayer', 'al', 'al.', 'the Jesus Prayer', 'Walk', 'Page', 'Text', 'the', 'and', 'or', 'of', 'in', 'on', 'for', 'with', 'to', 'by', 'at', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'this', 'that', 'these', 'those', 'a', 'an', 'but', 'if', 'because', 'while', 'where', 'when', 'which', 'who', 'whom', 'whose', 'how', 'what', 'why', 'can', 'may', 'might', 'should', 'would', 'could', 'will', 'shall', 'do', 'does', 'did', 'done', 'has', 'have', 'had', 'having', 'not', 'no', 'yes', 'etc', 'et', 'al', 'al.'
    ])
    min_len = 2
    max_len = 5  # max words in an entity
    def is_valid_entity(ent):
        # Remove if entity is in ignore list, is all lowercase, or is a single common word
        ent_clean = ent.strip(string.punctuation + ' ')
        if not ent_clean:
            return False
        if ent_clean in ignore_words:
            return False
        if ent_clean.lower() in ignore_words:
            return False
        if ent_clean.islower() or ent_clean.isupper():
            return False
        if any(char.isdigit() for char in ent_clean):
            return False
        if len(ent_clean.split()) > max_len or len(ent_clean.split()) < min_len:
            return False
        if ent_clean.lower() in set([w.lower() for w in ignore_words]):
            return False
        return True
    try:
        if 'nlp' in globals() and nlp is not None:
            doc = nlp(text)
            persons = set()
            orgs = set()
            places = set()
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if ent.label_ == 'PERSON' and is_valid_entity(ent_text):
                    persons.add(ent_text)
                elif ent.label_ == 'ORG' and is_valid_entity(ent_text):
                    orgs.add(ent_text)
                elif ent.label_ == 'GPE' and is_valid_entity(ent_text):
                    places.add(ent_text)
            entities['PERSON'] = sorted(persons)
            entities['ORG'] = sorted(orgs)
            entities['PLACES'] = sorted(places)
        else:
            import re
            names = set(n.strip() for n in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text) if is_valid_entity(n))
            orgs = set(o.strip() for o in re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(Inc|Ltd|Corporation|Corp|LLC)\b', text) if is_valid_entity(o))
            places = set(p.strip() for p in re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b', text) if is_valid_entity(p))
            entities['PERSON'] = sorted(names)
            entities['ORG'] = sorted(orgs)
            entities['PLACES'] = sorted(places)
    except Exception as e:
        print(f"Entity extraction failed: {e}")
    return entities

def improved_classify_content(text):
    if not text or not isinstance(text, str):
        return "Unknown"

    categories = {
        'Finance': ['invoice', 'payment', 'tax','total', 'billing', 'account', 'balance', 'bank'],
        'Education': ['university', 'school', 'student', 'teacher', 'course', 'curriculum', 'exam', 'degree', 'academic'],
        'Legal': ['contract', 'agreement', 'clause', 'terms', 'conditions', 'legal', 'whereas', 'law', 'court'],
        'Medical': ['patient', 'doctor', 'medicine', 'treatment', 'diagnosis', 'hospital', 'clinical'],
        'Technology': ['software', 'hardware', 'computer', 'technology', 'system', 'application', 'device'],
        'Report': ['research', 'report', 'analysis', 'findings', 'conclusion', 'summary', 'methodology'],
        'Resume/CV': ['resume', 'cv', 'curriculum vitae', 'experience', 'skills'],
        'Academic': ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'references'],
    }

    text_lower = text.lower()
    scores = {}

    for category, keywords in categories.items():
        scores[category] = sum(1 for word in keywords if word in text_lower)

    best_match = max(scores, key=scores.get)
    if scores[best_match] > 2:
        return best_match
    else:
        return 'General'


def generate_metadata(file_path, original_filename=None):
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    extraction_log = []
    page_count = None
    if ext == '.pdf':
        text, log = extract_text_from_pdf(file_path)
        extraction_log.extend(log)
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
        except Exception:
            page_count = None
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
        readability_score = None
        if text and isinstance(text, str) and len(text.split()) > 0:
            try:
                readability_score = textstat.flesch_reading_ease(text)
                if readability_score is not None:
                    readability_score = round(readability_score, 2)
            except Exception:
                readability_score = None
        basic_info = {
            "filename": original_filename or os.path.basename(file_path),
            "file_type": ext[1:].upper() if ext else 'Unknown',
            "file_size": f"{file_stats.st_size / 1024:.2f} KB",
            "processing_date": datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
        }
        content_analysis = {
            "language": detect_language(text) if text else 'Unknown',
            "word_count": len(text.split()) if text and isinstance(text, str) else 0,
            "character_count": len(text) if text and isinstance(text, str) else 0,
            "line_count": text.count('\n') + 1 if text and isinstance(text, str) else 0,
            "page_count": page_count,
            "readability_score": readability_score,
            "sentiment": analyze_sentiment(text) if text else 'Unknown',
            "category": improved_classify_content(text) if text else 'Unknown',
            "content type": classify_content(text) if text else "Unknown",
        }
        entities = extract_key_information(text) if text else {}
        semantic_data = {
            "summary": extract_summary(text) if text else "No text extracted",
            "key_topics": extract_keywords(text) if text else [],
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
