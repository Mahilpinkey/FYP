import fitz  # PyMuPDF for PDF reading
import re
import pandas as pd
import spacy
from nltk.stem import WordNetLemmatizer

# ------------------- NLP Setup -------------------
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# ------------------- Rule Dictionaries -------------------
PRONOUN_MAP = {
    "i": "I", "you": "YOU", "he": "HE", "she": "SHE",
    "we": "WE", "they": "THEY", "it": "IT",
    "me": "I", "him": "HE", "her": "SHE", "us": "WE", "them": "THEY",
    "my": "I", "your": "YOU", "his": "HE", "our": "WE", "their": "THEY", "its": "IT",
    "mine": "I", "yours": "YOU", "hers": "SHE", "ours": "WE", "theirs": "THEY",
    "this": "THIS", "that": "THAT", "these": "THESE", "those": "THOSE",
    "who": "WHO", "whom": "WHO", "whose": "WHO", "which": "WHICH", "that": "THAT"
}

FAMILY_RELATIONS = {
    "brother": ("MALE", "SIBLING"), "sister": ("FEMALE", "SIBLING"),
    "mother": ("FEMALE", "PARENT"), "father": ("MALE", "PARENT"),
    "son": ("MALE", "CHILD"), "daughter": ("FEMALE", "CHILD"),
    "grandmother": ("FEMALE", "GRANDPARENT"), "grandfather": ("MALE", "GRANDPARENT"),
    "uncle": ("MALE", "PARENT-SIBLING"), "aunt": ("FEMALE", "PARENT-SIBLING"),
    "cousin": ("NEUTRAL", "COUSIN"), "husband": ("MALE", "SPOUSE"), "wife": ("FEMALE", "SPOUSE")
}

TIME_ASPECT_MARKERS = {
    "now": "PRESENT", "today": "PRESENT",
    "yesterday": "PAST", "ago": "PAST",
    "tomorrow": "FUTURE", "soon": "FUTURE", "later": "FUTURE",
    "always": "HABITUAL", "often": "HABITUAL", "usually": "HABITUAL",
    "sometimes": "OCCASIONAL", "rarely": "OCCASIONAL", "never": "NEGATION"
}

NEGATIONS = {"not","never","no","none","nothing","nobody","cannot","don't","doesn't","didn't","won't"}

QUESTION_WORDS = {"who","what","where","when","why","how","which"}

AUXILIARIES = {"is","am","are","was","were","be","being","been",
               "do","does","did","done","has","have","had",
               "will","would","shall","should","can","could","may","might","must"}

ARTICLES = {"a","an","the"}
PREPOSITIONS = {"in","on","at","to","with","from","into","onto","over","under","by","for","about","of"}
CONJUNCTIONS = {"and","but","or","nor","for","so","yet","although","because","since","unless","while"}

# ------------------- Helper Functions -------------------
def finger_spell(word):
    if len(word) <= 2 or word.upper() in {"I","YOU","HE","SHE","WE","THEY","IT"}:
        return word.upper()
    return "-".join(list(word.upper()))

def detect_tense_and_aspect(doc):
    markers = []
    for token in doc:
        if token.text.lower() in TIME_ASPECT_MARKERS:
            markers.append(TIME_ASPECT_MARKERS[token.text.lower()])
        if token.pos_ == "VERB":
            if token.tag_ in {"VBD","VBN"}:
                markers.append("PAST")
            elif token.tag_ in {"VBZ","VBP"}:
                markers.append("PRESENT")
            elif token.text.lower() in {"will","shall"}:
                markers.append("FUTURE")
    return list(dict.fromkeys(markers))  # unique

def extract_svo(sentence):
    doc = nlp(sentence)
    subj, verb, obj = None, None, None
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verb = token.lemma_.lower()
        if "subj" in token.dep_:
            subj = token.lemma_.lower()
        if token.dep_ in {"dobj","pobj","obj"}:
            obj = token.lemma_.lower()
    return subj, verb, obj, doc

def handle_questions(sentence, doc, tokens):
    markers = []
    if sentence.strip().endswith("?"):
        if any(t in QUESTION_WORDS for t in tokens):
            qword = [t for t in tokens if t in QUESTION_WORDS][0]
            markers.append(qword.upper())
        else:
            markers.append("QUESTION")
    return markers

# ------------------- ISL Rule Application -------------------
def apply_isl_rules(sentence):
    subj, verb, obj, doc = extract_svo(sentence)
    content_tokens = [
        token.text.lower() for token in doc
        if token.is_alpha and token.pos_ in {"NOUN","VERB","ADJ","ADV","PROPN","NUM"}
        and token.text.lower() not in AUXILIARIES | ARTICLES | PREPOSITIONS | CONJUNCTIONS
    ]
    mapped_tokens = [PRONOUN_MAP.get(t,t) for t in content_tokens]
    processed_tokens = []
    for t in mapped_tokens:
        if t in FAMILY_RELATIONS:
            gender, relation = FAMILY_RELATIONS[t]
            processed_tokens.extend([gender, relation])
        else:
            processed_tokens.append(t)
    if subj and verb and obj:
        mapped_subj = PRONOUN_MAP.get(subj,subj)
        mapped_obj = PRONOUN_MAP.get(obj,obj)
        reordered = [mapped_subj, mapped_obj, verb]
        other_words = [t for t in processed_tokens if t not in {mapped_subj, mapped_obj, verb}]
        reordered.extend(other_words)
    else:
        reordered = processed_tokens
    reordered.extend(detect_tense_and_aspect(doc))
    if any(tok in NEGATIONS for tok in [t.text.lower() for t in doc]):
        reordered.append("NOT")
    question_markers = handle_questions(sentence, doc, [t.text.lower() for t in doc])
    reordered.extend(question_markers)
    simplified = []
    for tok in reordered:
        if isinstance(tok,str) and "-" not in tok:
            lemma = lemmatizer.lemmatize(tok,pos="v")
            simplified.append(lemma.upper() if lemma.isalpha() else finger_spell(lemma))
        else:
            simplified.append(tok)
    seen=set();final=[]
    for t in simplified:
        if t not in seen:
            seen.add(t)
            final.append(t)
    return " ".join(final)

# ------------------- PDF Processing -------------------
def extract_sentences_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    clean_text = re.sub(r"\s+"," ",text).strip()
    sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z])', clean_text)
    return sentences

# ------------------- Dataset Creator -------------------
def create_dataset(pdf_path, csv_output):
    sentences = extract_sentences_from_pdf(pdf_path)
    data = []
    for sent in sentences:
        gloss = apply_isl_rules(sent)
        if gloss.strip():
            data.append({"english_sentence": sent.strip(), "isl_gloss": gloss})
    pd.DataFrame(data).to_csv(csv_output,index=False,encoding="utf-8")
    print(f"✅ Dataset saved as {csv_output} with {len(data)} pairs")

# ------------------- MAIN -------------------
if __name__ == "__main__":
    pdf_path = "freecompress-Class_10_Science_English_Medium-2024_Edition-www.tntextbooks.in (2).pdf"
    csv_output = "english_to_isl_gloss_dataset.csv"
    create_dataset(pdf_path, csv_output)
