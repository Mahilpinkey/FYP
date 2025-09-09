import os
import yt_dlp
from pydub import AudioSegment
from datetime import datetime

# ✅ NLP imports
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import spacy
import whisper  # ✅ NEW

# ✅ Punctuation restoration model
from deepmultilingualpunctuation import PunctuationModel

# Download nltk resources (only first run)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_folder")

# Load punctuation model once at startup
punct_model = PunctuationModel()

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# ------------------- YOUTUBE AUDIO -------------------
def download_youtube_audio(url, output_audio_name="extracted_audio.mp3"):
    ensure_folder(OUTPUT_FOLDER)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = os.path.splitext(output_audio_name)[0]
    audio_path = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}_{timestamp}.mp3"
    )

    print("Downloading audio only...")
    ydl_opts = {
        'outtmpl': os.path.join(OUTPUT_FOLDER, f"{base_name}_{timestamp}"),
        'verbose': True,
        'format': 'bestaudio/best',
        "cookies": os.path.join(BASE_DIR, "cookies.txt"),
        'quiet': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"✅ Audio saved to {audio_path}")
    return audio_path

# ------------------- ENHANCED ISL RULES -------------------
# Expanded pronoun mapping
PRONOUN_MAP = {
    # Subject pronouns
    "i": "I", "you": "YOU", "he": "HE", "she": "SHE", 
    "we": "WE", "they": "THEY", "it": "IT",
    
    # Object pronouns
    "me": "I", "him": "HE", "her": "SHE", "us": "WE", "them": "THEY",
    
    # Possessive pronouns
    "my": "I", "your": "YOU", "his": "HE", "her": "SHE", 
    "our": "WE", "their": "THEY", "its": "IT",
    
    # Possessive adjectives
    "mine": "I", "yours": "YOU", "hers": "SHE", "ours": "WE", "theirs": "THEY",
    
    # Demonstrative pronouns
    "this": "THIS", "that": "THAT", "these": "THESE", "those": "THOSE",
    
    # Relative pronouns
    "who": "WHO", "whom": "WHO", "whose": "WHO", "which": "WHICH", "that": "THAT"
}

# Expanded family and relationship terms
FAMILY_RELATIONS = {
    "brother": ("MALE", "SIBLING"), "sister": ("FEMALE", "SIBLING"),
    "mother": ("FEMALE", "PARENT"), "father": ("MALE", "PARENT"),
    "son": ("MALE", "CHILD"), "daughter": ("FEMALE", "CHILD"),
    "grandmother": ("FEMALE", "GRANDPARENT"), "grandfather": ("MALE", "GRANDPARENT"),
    "grandson": ("MALE", "GRANDCHILD"), "granddaughter": ("FEMALE", "GRANDCHILD"),
    "uncle": ("MALE", "PARENT-SIBLING"), "aunt": ("FEMALE", "PARENT-SIBLING"),
    "nephew": ("MALE", "SIBLING-CHILD"), "niece": ("FEMALE", "SIBLING-CHILD"),
    "cousin": ("NEUTRAL", "COUSIN"), "husband": ("MALE", "SPOUSE"), "wife": ("FEMALE", "SPOUSE"),
    "partner": ("NEUTRAL", "PARTNER"), "spouse": ("NEUTRAL", "SPOUSE")
}

# Expanded time and aspect markers
TIME_ASPECT_MARKERS = {
    "now": "PRESENT", "today": "PRESENT", 
    "yesterday": "PAST", "ago": "PAST", 
    "tomorrow": "FUTURE", "soon": "FUTURE", "later": "FUTURE",
    "always": "HABITUAL", "often": "HABITUAL", "usually": "HABITUAL",
    "sometimes": "OCCASIONAL", "rarely": "OCCASIONAL", "never": "OCCASIONAL-NEGATION"
}

# Enhanced negation list with contractions
NEGATIONS = {
    "not", "never", "no", "none", "nothing", "nobody", "nowhere", "neither", 
    "cannot", "can't", "couldn't", "couldnot",
    "don't", "doesn't", "didn't", "dont", "doesnt", "didnt",
    "won't", "wouldn't", "wont", "wouldnt",
    "shouldn't", "mustn't", "mightn't", "shant",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"
}

# Expanded question words
QUESTION_WORDS = {
    "who", "what", "where", "when", "why", "how", "which", "whom", 
    "whose", "whatever", "whenever", "wherever", "whoever", "whichever", "howcome"
}

# Expanded auxiliaries and modals
AUXILIARIES = {
    "is", "am", "are", "was", "were", "be", "being", "been",
    "do", "does", "did", "done",
    "has", "have", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must", "ought"
}

# Expanded articles
ARTICLES = {"a", "an", "the"}

# Expanded prepositions
PREPOSITIONS = {
    "in", "on", "at", "to", "with", "from", "into", "onto", "over",
    "under", "above", "below", "by", "for", "about", "around", "through", "of",
    "across", "after", "against", "along", "among", "before", "behind", "beneath",
    "beside", "between", "beyond", "down", "during", "except", "inside", "near",
    "off", "out", "outside", "past", "since", "throughout", "toward", "underneath",
    "until", "upon", "within", "without"
}

# Conjunctions to potentially remove or handle specially
CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "so", "yet", 
    "although", "because", "since", "unless", "until", "while"
}

def finger_spell(word):
    """Convert a word to finger-spelled representation"""
    # Don't finger-spell common short words
    if len(word) <= 2 or word.upper() in {"I", "YOU", "HE", "SHE", "WE", "THEY", "IT"}:
        return word.upper()
    return "-".join(list(word.upper()))

def detect_tense_and_aspect(doc):
    """
    Enhanced tense and aspect detection using spaCy's linguistic features
    Returns a list of tense/aspect markers
    """
    markers = []
    
    # Check for explicit time words first
    for token in doc:
        if token.text.lower() in TIME_ASPECT_MARKERS:
            markers.append(TIME_ASPECT_MARKERS[token.text.lower()])
    
    # Check verb tenses
    for token in doc:
        if token.pos_ == "VERB":
            # Past tense
            if token.tag_ in {"VBD", "VBN"}:
                markers.append("PAST")
            # Present tense
            elif token.tag_ in {"VBZ", "VBP"}:
                markers.append("PRESENT")
            # Future indicated by "will" or "shall"
            elif token.text.lower() in {"will", "shall"}:
                markers.append("FUTURE")
            
            # Aspect markers
            if token.tag_ == "VBG":  # -ing form
                markers.append("PROGRESSIVE")
            if token.tag_ == "VBN" and token.dep_ != "ROOT":  # Past participle (passive)
                markers.append("COMPLETED")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_markers = []
    for marker in markers:
        if marker not in seen:
            seen.add(marker)
            unique_markers.append(marker)
    
    return unique_markers

def extract_svo_enhanced(sentence):
    """
    Enhanced subject-verb-object extraction that handles more complex sentences
    """
    doc = nlp(sentence)
    subjects, verbs, objects = [], [], []
    
    for token in doc:
        # Find main verbs
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verbs.append(token.lemma_.lower())
        
        # Find subjects
        if "subj" in token.dep_:
            subjects.append(token.lemma_.lower())
        
        # Find objects
        if token.dep_ in {"dobj", "pobj", "obj"}:
            objects.append(token.lemma_.lower())
    
    # Return the first of each found, or None if not found
    subj = subjects[0] if subjects else None
    verb = verbs[0] if verbs else None
    obj = objects[0] if objects else None
    
    return subj, verb, obj, doc

def handle_questions(sentence, doc, tokens):
    """
    Enhanced question handling for different question types
    """
    markers = []
    
    # Check for question mark
    is_question = sentence.strip().endswith('?')
    
    # Find question words
    question_words = [t for t in tokens if t in QUESTION_WORDS]
    
    if question_words:
        # For WH-questions, the question word typically comes first in ISL
        markers.insert(0, question_words[0].upper())
    elif is_question:
        # For yes/no questions, add a question marker
        markers.append("QUESTION")
    
    # Check for question inversion (auxiliary before subject)
    has_inversion = False
    for i, token in enumerate(doc):
        if token.dep_ == "aux" and i > 0 and doc[i-1].dep_ in {"nsubj", "nsubjpass"}:
            has_inversion = True
            break
    
    if has_inversion and not question_words:
        markers.append("QUESTION")
    
    return markers

def apply_isl_rules(sentence):
    """
    Enhanced ISL rules application with better linguistic processing
    """
    # Parse the sentence with spaCy
    doc = nlp(sentence)
    
    # Extract SVO
    subj, verb, obj, doc = extract_svo_enhanced(sentence)
    
    # Get all content words (nouns, verbs, adjectives, adverbs)
    content_tokens = [
        token.text.lower() for token in doc 
        if token.is_alpha and 
           token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"} and
           token.text.lower() not in AUXILIARIES | ARTICLES | PREPOSITIONS | CONJUNCTIONS
    ]
    
    # Apply pronoun mapping
    mapped_tokens = [PRONOUN_MAP.get(t, t) for t in content_tokens]
    
    # Handle family/gender terms
    processed_tokens = []
    for t in mapped_tokens:
        if t in FAMILY_RELATIONS:
            gender, relation = FAMILY_RELATIONS[t]
            processed_tokens.extend([gender, relation])
        else:
            processed_tokens.append(t)
    
    # Reorder to SOV structure if we have all components
    if subj and verb and obj:
        # Map subjects and objects
        mapped_subj = PRONOUN_MAP.get(subj, subj)
        mapped_obj = PRONOUN_MAP.get(obj, obj)
        
        # Handle family terms in subject/object position
        if mapped_subj in FAMILY_RELATIONS:
            gender, relation = FAMILY_RELATIONS[mapped_subj]
            mapped_subj = f"{gender}-{relation}"
        if mapped_obj in FAMILY_RELATIONS:
            gender, relation = FAMILY_RELATIONS[mapped_obj]
            mapped_obj = f"{gender}-{relation}"
            
        reordered = [mapped_subj, mapped_obj, verb]
        
        # Add other content words that weren't part of the core SVO
        other_words = [t for t in processed_tokens if t not in {mapped_subj, mapped_obj, verb}]
        reordered.extend(other_words)
    else:
        reordered = processed_tokens
    
    # Detect and add tense/aspect markers
    tense_markers = detect_tense_and_aspect(doc)
    reordered.extend(tense_markers)
    
    # Handle negation
    if any(tok in NEGATIONS for tok in [t.text.lower() for t in doc]):
        reordered.append("NOT")
    
    # Handle questions
    question_markers = handle_questions(sentence, doc, [t.text.lower() for t in doc])
    reordered.extend(question_markers)
    
    # Simplify verb forms (lemmatize)
    simplified = []
    for tok in reordered:
        if isinstance(tok, str) and "-" not in tok:  # Don't process already composed terms
            # Lemmatize verbs
            if tok in {verb, subj, obj} and verb:  # Only process if it was identified as a verb
                lemma = lemmatizer.lemmatize(tok, pos='v')
                simplified.append(lemma.upper() if lemma.isalpha() else finger_spell(lemma))
            else:
                simplified.append(tok.upper() if tok.isalpha() else finger_spell(tok))
        else:
            simplified.append(tok)
    
    # Final cleanup - remove duplicates while preserving order
    seen = set()
    final_gloss = []
    for token in simplified:
        if token not in seen:
            seen.add(token)
            final_gloss.append(token)
    
    return final_gloss

# ------------------- AUDIO TO TEXT + GLOSS -------------------
def audio_to_text(audio_file, output_text_file="transcription.txt"):
    ensure_folder(OUTPUT_FOLDER)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert MP3 → WAV (mono, 16kHz)
    wav_path = os.path.join(OUTPUT_FOLDER, f"converted_audio_{timestamp}.wav")
    sound = AudioSegment.from_file(audio_file, format="mp3")
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    sound.export(wav_path, format="wav")

    # ✅ Use Whisper instead of Google recognizer
    print("Converting audio to text with Whisper...")
    model = whisper.load_model("base")  # "tiny", "base", "small", "medium", "large"
    result = model.transcribe(wav_path, language="en")
    full_text = result["text"]

    # ✅ Restore punctuation
    print("⏳ Restoring punctuation...")
    punctuated_text = punct_model.restore_punctuation(full_text)
    print("✅ Punctuation restored")

    # Save transcript
    transcript_file = os.path.join(
        OUTPUT_FOLDER,
        f"{os.path.splitext(output_text_file)[0]}_{timestamp}.txt"
    )
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(punctuated_text)
    print(f"✅ Transcript saved: {transcript_file}")

    # Process sentences directly to ISL gloss (no intermediate token step)
    sentences = sent_tokenize(punctuated_text)
    all_gloss = []

    for sentence in sentences:
        # Directly convert each sentence to ISL gloss
        gloss = apply_isl_rules(sentence)
        if gloss:  # Only add non-empty glosses
            all_gloss.append(" ".join(gloss))

    # Save gloss (no token file generated)
    gloss_file = os.path.join(OUTPUT_FOLDER, f"all_gloss_{timestamp}.txt")
    with open(gloss_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_gloss))

    print(f"✅ All gloss saved: {gloss_file}")

    return punctuated_text, all_gloss

# ------------------- MAIN -------------------
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=GygBY01Qbnk"
    audio_path = download_youtube_audio(youtube_url)
    transcript, glosses = audio_to_text(audio_path, "video_transcription.txt")
    