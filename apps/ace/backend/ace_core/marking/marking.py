from metaphone import doublemetaphone
import re
from nltk.corpus import wordnet as wn
import rapidfuzz
from marking.preprocessing import *
from marking.preprocessing import number_words
from visual_tasks.clock_scorer import score_clock_image
from visual_tasks.cube_scorer import score_cube_image
from visual_tasks.infinity_scorer import score_infinity_image
from visual_tasks.pen_paper_scorer import score_pen_paper_video
from visual_tasks.writing import score_writing_image


letter_fluency_bands = [
    (0, 1, 0),
    (2, 3, 1),
    (4, 5, 2),
    (6, 7, 3),
    (8, 10, 4),
    (11, 13, 5),
    (14, 17, 6),
    (18, float("inf"), 7)
]

animal_fluency_band = [
    (0, 4, 0),
    (5, 6, 1),
    (7, 8, 2),
    (9, 10, 3),
    (11, 13, 4),
    (14, 16, 5),
    (17, 21, 6),
    (22, float("inf"), 7)
]

fuzzy_threshold = 82
fuzzy_window = 6 # Longest possible awnser someone could give is 6 words long "a stich in time saves nine" or a spelled out year "two thousand and twenty six"

def phonetic_equal(a, b):
    # Remove very short words produce noisy phonetic codes
    if len(a) < 3 or len(b) < 3:
        return False
    # Filters out phonetically similar but distinct words
    # Prevents false matches like "bell" vs "ball" (if fuzzy ratio < 80)
    if rapidfuzz.fuzz.ratio(a, b) < 80:
        return False
    # Compute Double Metaphone codes (returns primary and secondary phonetic representations)
    pa, sa = doublemetaphone(a)
    pb, sb = doublemetaphone(b)
    # If either string produces no valid phonetic encoding, fail early
    if not pa or not pb:
        return False
    # Collect non-empty primary (p) and secondary (s) phonetic keys for both words
    codes_a = {c for c in (pa, sa) if c}
    codes_b = {c for c in (pb, sb) if c}
    # Match if there is ANY overlap between the phonetic keys of word A and word B
    return bool(codes_a & codes_b)

def score_exact(response, answers):
    """Single-character exact match (fragmented letters).
      Checks if the expected letter appears as a word in the response."""
    words = clean_response(response).split()
    return 1 if any(clean_response(a) in words for a in answers) else 0

def score_integer(response, answers):
    """Parsed integer match via sliding window (dot counting). Credits a
    match only if no other number is said afterwards"""
    response_words = clean_response(response).split()
    expected = [normalise_number(clean_response(a)) for a in answers]

    match_end = -1  # Tracks the ending index of the LAST valid match found
    # Locate the furthest matching expected number (window size 1 to 3 words)
    for n in range(1, min(3, len(response_words)) + 1):
        for i in range(len(response_words) - n + 1):
            if normalise_number(" ".join(response_words[i:i + n])) in expected:
                match_end = max(match_end, i + n)
    # If none of the expected numbers were found anywhere in the response
    if match_end == -1:
        return 0
    # Check for trailing numbers spoken after the last valid match
    # (e.g. "it was 5 no wait 6" -> match_end is after '5', but '6' follows)
    for n in range(1, min(3, len(response_words) - match_end) + 1):
        for i in range(match_end, len(response_words) - n + 1):
            val = normalise_number(" ".join(response_words[i:i + n]))
            try:
                int(val)
                return 0  # a different number was said afterwards
            except ValueError:
                pass
    return 1

def score_serial_sevens(response):
    """Five steps down from 100, one mark per step exactly 7 below the number
    said before it. Takes the numbers the patient offered as answers -- graph.py
    runs extract_serial_sevens over the spoken turn first, so no word parsing
    happens here. Values of 100+ (the starting point, echoed back) and a number
    repeated twice in a row are dropped before scoring."""
    if isinstance(response, list):
        response = " ".join(map(str, response))
    spoken = [int(n) for n in re.findall(r"\d+", str(response)) if int(n) < 100]
    score = 0
    prev = 100
    steps = 0
    for i, num in enumerate(spoken):
        # "ninety... ninety-three" leaves a bare 90 in front of the real answer that the LLM does not remove
        following = next((n for n in spoken[i + 1:] if n != num), None)
        if num % 10 == 0 and following is not None and following // 10 == num // 10:
            continue
        if num == prev:  # same number said twice
            continue
        if prev - num == 7:
            score += 1
        prev = num
        steps += 1
        if steps == 5:
            break
    return score

tens = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}

def _is_number_fragment(response_words, i, n):
    """A lone ones/ordinal word (e.g. "ninth") immediately preceded by a
    tens-word (e.g. "twenty ninth", from hyphen-split "twenty-ninth") is the
    tail of a compound number -- only the pair together counts as a match,
    not the ones-word alone."""
    return n == 1 and i > 0 and response_words[i - 1] in tens

def score_fuzzy(response, answers):
    """Any answer from = 1 point."""
    response_words = clean_response(response).split()
    for answer in answers:
        # Pre-process target answer
        expected = normalise_number(clean_response(answer))
        for n in range(1, min(fuzzy_window, len(response_words)) + 1):
            for i in range(len(response_words) - n + 1):
                if _is_number_fragment(response_words, i, n):
                    continue
                span = response_words[i:i + n]
                window = normalise_number(" ".join(span))
                window_joined = normalise_number("".join(span))
                # do not exept close calls that match 88 fuzzy window
                # e.g unintelligi (88% match but is not a point)
                cut_off = expected.startswith(window) and window != expected
                # Check match criteria: fuzzy similarity exact match OR phonetic sound
                if ((rapidfuzz.fuzz.ratio(window, expected) >= fuzzy_threshold and not cut_off)
                        or window_joined == expected
                        or phonetic_equal(window, expected)):
                    return 1
    return 0

def is_same(a, b):
    return rapidfuzz.fuzz.ratio(a, b) >= fuzzy_threshold or phonetic_equal(a, b)

name_fillers = {
    "his", "her", "him", "its", "it's", "name", "is", "was", "the", "a", "an",
    "um", "uh", "erm", "i", "think", "that's", "that", "mr", "mrs", "ms", "dr",
    "president", "minister", "prime", "hold", "on", "yes", "sure", "right", "okay", "ok",
}


def score_person_name(response, answers):
    """
    - Allows surnames (e.g., "Obama").
    - If full name is incorrect (e.g., "June Thatcher" instead of "Margaret Thatcher"), score is 0.
    """
    # Isolate multi-word accepted answers (e.g., ["Margaret", "Thatcher"])
    full = [a for a in answers if len(a.split()) > 1]
    # Isolate target surnames (e.g., ["Thatcher"])
    surnames = [clean_response(a) for a in answers if len(a.split()) == 1]
    if not surnames:
        return 0
    full_tokens = [clean_response(a).split() for a in full]
    words = clean_response(response).split()
    for surname in surnames:
        # Get valid given-name sequences for this surname (e.g., ["Margaret"])
        givens = [t[:-1] for t in full_tokens if is_same(t[-1], surname)]
        for i, w in enumerate(words):
            # Locate the position of the surname in the user's response
            if not is_same(w, surname):
                continue
            # Gather preceding non-filler words (the given names provided by the user)
            claimed, j = [], i - 1
            while j >= 0 and words[j] not in name_fillers:
                claimed.insert(0, words[j])
                j -= 1
            # Rule: Bare surname provided (e.g., "Thatcher" or "Mrs. Thatcher") -> Score 1
            if not claimed:
                return 1
            # Rule: Full name provided -> Score 1 ONLY if given name matches ("Margaret Thatcher"),
            # otherwise Score 0 if given name is incorrect ("June Thatcher").
            if any(len(g) == len(claimed) and all(is_same(x, y) for x, y in zip(claimed, g))
                   for g in givens):
                return 1

    return 0

answer_variants = {
    # reading: phonetically similar
    "sew": ["sou", "so", "soh"],
    "soot": ["sut", "sutt"],
    "dough": ["doe", "doh", "dou"],
    "height": ["hite"],
    # naming: accepted alternate names for the same picture
    "kangaroo": ["wallaby"],
    "camel": ["dromedary"],
    "rhinoceros": ["rhino"],
    "barrel": ["keg", "tub"],
    "crocodile": ["alligator"],
    "accordion": ["piano accordion", "squeeze box"],
}

def score_fuzzy_list(response, answers):
    """Each answer in list scored separately via sliding window. Response words
    already used by an earlier match can't be reused by a later one."""
    score = 0
    matched = set() # Tracks distinct target answers already found to avoid duplicate scoring
    used_words = set() # Tracks word indices in the response already consumed by a match
    response_words = clean_response(response).split()
    # Pre-process and normalize expected target answers
    expected = []
    for a in answers:
        base = normalise_number(clean_response(a))
        expected.append((base, answer_variants.get(base, [])))

    for y, aliases in expected:
        # Skip this target answer if an identical target was already matched
        if y in matched:
            continue
        # Try n-gram window sizes from 1 up to fuzzy threshold (or total response length)
        for n in range(1, min(fuzzy_threshold, len(response_words)) + 1):
            # Slide an n-word window across the response text
            for i in range(len(response_words) - n + 1):
                # Skip candidate windows containing word indices already claimed by a prior match
                if used_words & set(range(i, i + n)):
                    continue
                # combine words into normalized phrase
                window = normalise_number(" ".join(response_words[i:i + n]))

                # Check match criteria: high fuzzy string similarity OR matching phonetic sound
                if (rapidfuzz.fuzz.ratio(window, y) >= 88  or phonetic_equal(window, y)
                        or any(rapidfuzz.fuzz.ratio(window, v) >= 88 for v in aliases)):
                    matched.add(y)
                    used_words.update(range(i, i + n))
                    score += 1
                    break # Match found for size 'n'; exit the window-position loop
            else:
                continue
            break
    return score


def score_all_correct_list(response, answers):
    """1 point only if every item in the list was matched,
    else 0. Used where the guide gives no partial credit (Reading):
    'Score 1 point if all five words are read correctly')."""
    return 1 if score_fuzzy_list(response, answers) == len(answers) else 0

def score_sentence_repetition(response, answers):
    """Every word of the target sentence must be found (fuzzy/phonetic) somewhere
    in the response so hesitation filler ("um", "let me see") around the
    sentence doesn't tank the score, but a dropped or wrong word does."""
    words = clean_response(answers[0]).split()
    return score_all_correct_list(response, words)

def is_animal(word):
    """Checks if a word belongs to the animal hierarchy in WordNet.
    "All types of animals are accepted, including insects, humans,
    prehistoric, extinct as well as mythical creatures (e.g., unicorn)."
    """
    formatted = word.replace(" ", "_")
    if formatted.lower() == "pegasus": # pegasus not in wordnet
        return True
    for ss in wn.synsets(formatted, pos=wn.NOUN):
        if ss == wn.synset("animal.n.01"):
            return True
        parents = set(ss.closure(lambda s: s.hypernyms()))
        if wn.synset("animal.n.01") in parents:
            return True
        if wn.synset("imaginary_being.n.01") in parents: # Mythical Creatures are accepted
            return True
        if wn.synset("dinosaur.n.01") in parents: # dinousours are accepted
            return True
    return False



animal_types = {
    "fish", "bird", "insect", "reptile", "mammal", "rodent",
    "amphibian", "primate", "bug", "animal", "shellfish", "arachnid",
}

# examples like Doe, Deer , faawn and stag are worth 1 point
gender_map = {
    "doe": "deer", "stag": "deer", "fawn": "deer", "buck": "deer",
    "bull": "cow", "calf": "cow", "heifer": "cow", "steer": "cow",
    "stallion": "horse", "mare": "horse", "foal": "horse", "colt": "horse",
    "rooster": "chicken", "hen": "chicken", "chick": "chicken",
    "ram": "sheep", "ewe": "sheep", "lamb": "sheep",
    "sow": "pig", "boar": "pig", "piglet": "pig",
    "cub": "bear", "lioness": "lion", "puppy": "dog", "kitten": "cat",
    "man":"women"
}

def score_animal_fluency(response):
    words = clean_response(response).split()
    unique_animals = set()
    used_indices = set()
    # Catch 2-word animals in WordNet (e.g., "polar bear")
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if is_animal(bigram):
            canonical = gender_map.get(bigram, bigram) # checks against bigram to see if gender-specific name is present
            unique_animals.add(canonical)
            used_indices.update({i, i + 1})
    # Catch 1-word animals in WordNet + lemmatize plurals ("cats" -> "cat")
    for i, word in enumerate(words):
        if i in used_indices:
            continue
        root = wn.morphy(word, wn.NOUN) or word
        matched_term = None

        if is_animal(root):
            matched_term = root
        elif is_animal(word):
            matched_term = word

        if matched_term:
            canonical = gender_map.get(matched_term, matched_term)
            unique_animals.add(canonical)
    # Map unique animal names to their primary WordNet noun synset
    synsets = {}
    for animal_name in unique_animals:
        formatted_name = animal_name.replace(" ", "_")
        # Query WordNet for noun synsets matching the formatted name
        found_synsets = wn.synsets(formatted_name, pos=wn.NOUN)
        # If matches exist, store the primary (first) synset using the original name
        if found_synsets:
            synsets[animal_name] = found_synsets[0]

    # Drop a category word (e.g. "fish") if a specific exemplar under it was
    # also said (e.g. "salmon") -- only the specific exemplars should count.
    to_drop = set()
    for category in unique_animals:
        if category in animal_types and category in synsets:
            cat_ss = synsets[category]
            for item_name, item_ss in synsets.items():
                if item_name != category:
                    parents = set(item_ss.closure(lambda s: s.hypernyms()))
                    if cat_ss in parents:
                        to_drop.add(category)
                        break

    final_unique = unique_animals - to_drop
    return scaled_count(len(final_unique), animal_fluency_band)

def scaled_count(count, bands):
    """Returns score from the first band where count falls between min and max.
    if no band matches return 0 score"""
    for min_count, max_count, score in bands:
        if min_count <= count <= max_count:
            return score
    return 0

# Common P- first names excluded from letter fluency even when they also happen
common_names = {
    "peter", "paul", "patricia", "pamela", "paula", "penny", "penelope",
    "philip", "phillip", "phoebe", "priscilla", "patrick", "percy", "piper",
}

def p_word_root(word):
    """Return the dictionary root of `word` if it's a valid common P-word, else None.
    Normalizing to the WordNet root (via morphy) merges perseverations and plurals
    (pay/paid/pays -> pay, pot/pots -> pot) into a single countable word."""
    word = word.lower().strip()
    # Rejects the word if less than 2 or starts with P return None
    if len(word) < 3 or not word.startswith("p"):
        return None
    # rejects word if in common names
    if word in common_names:
        return None
    # Gets the base dictonary root (e.g paid -> pay)
    root = wn.morphy(word)

    if root is None: # rejects madeup words or typos
        return None
    # Check if root exists as a common word in WordNet
    for synset in wn.synsets(root):
        for lemma in synset.lemmas():
            if lemma.name().lower() == root and lemma.name()[0].islower():
                return root
    return None

def score_letter_fluency(response):
    """Scores valid P words a person produces and converts into fluency score 0-7"""
    words = clean_response(response).split()

    # Gests unique dictionary roots {} drops duplicate values
    roots = {p_word_root(w) for w in words}

    roots.discard(None) # Removes None from the list
    return scaled_count(len(roots), letter_fluency_bands) # Returns score 0-7

def parse_spoken_prompts(question: dict) -> list[str]:
    """Extract every spoken prompt from instructions, one entry per Wait-for-response pause."""
    instructions = question.get("instructions", "")
    chunks = instructions.split("Wait for a response.")
    prompts = []
    for chunk in chunks[:-1]:
        speaks = re.findall(r"[Ss]peak[^']*'(.*?)'(?=[^a-zA-Z]|$)", chunk, re.DOTALL)
        if speaks:
            prompts.append(" ".join(speaks))
        else:
            # Non-Speak-prefixed prompts (e.g. word repetition continuations)
            first_q = chunk.find("'")
            last_q = chunk.rfind("'")
            if first_q != -1 and last_q > first_q:
                prompts.append(chunk[first_q + 1:last_q])
    return prompts

def get_sub_prompts(question: dict) -> list[str]:
    """Returns prompts only when each answer has its own spoken prompt (multi-prompt tracking)."""
    prompts = parse_spoken_prompts(question) # Gets "instruction section from the json"
    n_answers = len(question.get("answers", []))
    if len(prompts) == n_answers and n_answers > 1:
        return prompts
    return []

def score_mixed_list_detailed(response, answers):
    """Like score_mixed_list but reports which answers matched, in answer order.
    Used to carry per-element recall results into a later recognition task."""
    matched_text = set()
    matched = [False] * len(answers)
    # Pre-process and tokenise the participant's full response
    response_words = clean_response(response).split()
    # Check each target answer against sliding n-gram windows in the response
    for idx, answer in enumerate(answers):
        cleaned = clean_response(answer)
        normed = normalise_number(cleaned)
        is_number = normed.isdigit()
        # Skip if this specific answer text was already matched earlier in the list
        if cleaned in matched_text:
            continue
        # Search using 1 to 3 word sliding windows across response_words
        for n in range(1, min(3, len(response_words)) + 1):
            for i in range(len(response_words) - n + 1):
                window_raw = " ".join(response_words[i:i + n])
                window = normalise_number(window_raw)
                hit = (window == normed) if is_number else (rapidfuzz.fuzz.ratio(window_raw, cleaned) >= fuzzy_threshold)
                # Exact match for numerical answers, fuzzy ratio match for text answers
                if hit:
                    matched_text.add(cleaned)
                    matched[idx] = True
                    break
            else:
                continue
            break
    return matched

def score_mixed_list(response, answers):
    """Each answer scored separately; numeric answers use integer match, strings use fuzzy."""
    return sum(score_mixed_list_detailed(response, answers))

def score_question(response, question, sub_index=None):
    """Main dispatch/handler. Returns the integer score for `response`."""
    match_type = question.get("match_type", "fuzzy_list")
    answers = question.get("answers", [])

    if match_type == "fluency_letter":
        return score_letter_fluency(response)
    if match_type == "fluency_animal":
        return score_animal_fluency(response)
    if match_type == "clock":
        return score_clock_image(response)['total']
    if match_type == "pen_paper":
        return score_pen_paper_video(response)["total"]
    if match_type == "cube":
        return score_cube_image(response)["total"]
    if match_type == "sentances":
        return score_writing_image(response)["total"]
    if match_type == "infinity":
        return score_infinity_image(response)["total"]
    if not answers:
        return 0

    if sub_index is not None:
        if sub_index >= len(answers):
            return 0
        alt = answers[sub_index]
        # Some dynamic sub-answers (e.g. date +/- tolerance) resolve to a list
        # of acceptable alternatives rather than a single string.
        return score_fuzzy(response, alt if isinstance(alt, list) else [alt])

    dispatch = {
        "exact":        score_exact,
        "integer":      score_integer,
        "fuzzy":        score_fuzzy,
        "fuzzy_list":   score_fuzzy_list,
        "person_name":  score_person_name,
        "mixed_list":   score_mixed_list,
        "all_correct_list": score_all_correct_list,
        "serial_sevens": lambda r, a: score_serial_sevens(r),
        "sentence_repetition": score_sentence_repetition,
    }
    fn = dispatch.get(match_type)
    return fn(response, answers) if fn else 0
