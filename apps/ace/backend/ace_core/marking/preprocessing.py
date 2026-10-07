import re
from word2number import w2n

ordinals = {
    "first": "one", "second": "two", "third": "three", "fourth": "four",
    "fifth": "five", "sixth": "six", "seventh": "seven", "eighth": "eight",
    "ninth": "nine", "tenth": "ten", "eleventh": "eleven", "twelfth": "twelve",
    "thirteenth": "thirteen", "fourteenth": "fourteen", "fifteenth": "fifteen",
    "sixteenth": "sixteen", "seventeenth": "seventeen", "eighteenth": "eighteen",
    "nineteenth": "nineteen", "twentieth": "twenty", "thirtieth": "thirty",
}

def clean_response(text):
    text = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

number_words = set(w2n.american_number_system.keys())

# e.g Converts 85 into eighty-five
def normalise_number(text):
    text = " ".join(ordinals.get(w, w) for w in text.split())
    # Drop fillers said mid-number ("two thousand and... uh... twenty six")
    words = [w for w in text.split() if w not in {"um", "uh", "erm", "er", "ah", "eh"}]
    if not words or not all(w.isdigit() or w in number_words or w == "and" for w in words):
        return text
    if not any(w.isdigit() or w in number_words for w in words):
        return text
    # A spoken year is two halves: "twenty twenty six"
    for i in range(1, len(words)):
        try:
            hi, lo = w2n.word_to_num(" ".join(words[:i])), w2n.word_to_num(" ".join(words[i:]))
        except (ValueError, IndexError):
            continue
        if 10 <= hi <= 99 and 10 <= lo <= 99:
            return str(hi * 100 + lo)
    try:
        return str(w2n.word_to_num(" ".join(words)))
    except (ValueError, IndexError):
        return text
