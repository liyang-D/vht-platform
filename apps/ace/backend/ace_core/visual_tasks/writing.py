import json
import os

import spacy
from dotenv import load_dotenv
from LLM.vlm import build_client, describe_images, save_vlm_response
import language_tool_python
load_dotenv()

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")
tool = None


def sentence_has_grammer_issues(sentence_text):
        global tool
        if tool is None:
            tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(sentence_text)
        critical_errors = [
            m for m in matches
        if m.category in ('GRAMMAR', 'TYPOS', 'CASING')
        ]
        # Returns True if grammar errors exist, False if none are found
        return len(critical_errors) > 0


def classify_sentences(text):
    """Split text into sentences and, for each, report whether it has a
    subject+verb (i.e. counts as a sentence rather than a fragment) and
    whether it contains any grammar/spelling errors."""
    doc = nlp(text)

    classified = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in sent)
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)

        classified.append({
            "text": sent_text,
            "is_valid_sentence": has_subject and has_verb,
            "has_errors": sentence_has_grammer_issues(sent_text),
        })

    return classified

def score_sentence_writing(text):
    global tool
    sentences = classify_sentences(text)
    # Shut the Java server down again now every sentence has been checked
    if tool is not None:
        tool.close()
        tool = None

    valid_sentences = [s for s in sentences if s["is_valid_sentence"]]
    clean_sentences = [s for s in valid_sentences if not s["has_errors"]]

    if len(valid_sentences) >= 2 and len(clean_sentences) == len(valid_sentences):
        return 2
    if len(valid_sentences) >= 2:
        return 1
    if len(valid_sentences) == 1 and len(clean_sentences) == 1:
        return 1
    return 0


transcribe_prompt = """You are transcribing handwritten text from a photo, for a cognitive assessment.

Transcribe exactly what is written, as plain text, preserving sentence boundaries and
punctuation. If a word is illegible, write [illegible] in its place. Do not correct
spelling or grammar — transcribe it as written.

Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "transcription": "<the transcribed text>",
  "sentence_count": "<integer count of distinct attempted sentences visible on the page>"
}
"""

transcribe_llm = build_client(0.0, 4000)

def inspect_writing_image(image_path: str) -> dict:
    """Return the transcription and visible attempted-sentence count."""
    data = describe_images(transcribe_llm, transcribe_prompt, [image_path])
    text = data.get("transcription", "")
    try:
        sentence_count = max(0, int(data.get("sentence_count", 0)))
    except (TypeError, ValueError):
        sentence_count = len([sentence for sentence in nlp(text).sents if sentence.text.strip()])
    return {"transcription": text, "sentence_count": sentence_count}


def transcribe_writing(image_path: str) -> str:
    """Send the photographed writing sample to the VLM and return its plain-text
    transcription. Returns "" on any connection failure or malformed response."""
    return inspect_writing_image(image_path)["transcription"]

def score_writing_image(image_path: str) -> dict:
    """Transcribe the photographed writing sample via VLM, then score it against
    the ACE-III sentence-writing criteria."""
    inspection = None
    if isinstance(image_path, str) and image_path.lstrip().startswith("{"):
        try:
            payload = json.loads(image_path)
            inspection = payload.get("inspection")
            image_path = payload.get("final_image", image_path)
        except (ValueError, TypeError):
            inspection = None
    text = (
        inspection.get("transcription", "")
        if isinstance(inspection, dict)
        else transcribe_writing(image_path) if os.path.exists(image_path) else image_path
    )
    total = score_sentence_writing(text) if text else 0
    result = {"total": total, "sentence_count": (inspection or {}).get("sentence_count")}
    save_vlm_response("writing", image_path, inspection or {"transcription": text}, result)
    return result
