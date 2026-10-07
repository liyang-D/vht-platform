from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage
from rapidfuzz import fuzz
from PIL import Image
import json
import os
import re
import time
from LLM.dialogue import *
from marking.marking import *
from datetime import datetime
from data_loader import resolve_dynamic_answers, get_season_transition, ace_json
def is_click_point_question(question: dict) -> bool:
    return question.get("question_text", "").startswith("Comprehension: Which picture")


def task_type(question: dict) -> str:
    match_type = question.get("match_type", "")
    if match_type in ("clock", "cube", "infinity", "sentances"):
        return "draw"
    if match_type == "pen_paper":
        return "video"
    if is_click_point_question(question):
        return "click"
    return "spoken"


def run_visual_task(*_args, **_kwargs):
    raise RuntimeError("A session I/O adapter has not been configured.")


def run_click_task(*_args, **_kwargs):
    raise RuntimeError("A session I/O adapter has not been configured.")
def furhat_nod(_furhat):
    return None


def furhat_repeat(_furhat):
    return None

class ACEState(MessagesState):
    current_domain: str
    question_index: int
    sub_question_index: int
    question_score: int
    scores: dict
    domain_queue: list
    complete: bool
    needs_repeat: bool
    repeat_count: int
    reprompt_kind: str   # None | "season" | "name" | "leader" | "trial"
    turn_progress: int   # generic per-question counter; meaning depends on the active handler
    recall_matches: dict  # recall_key -> per-answer bool list, for recognition-task skip logic
    question_log: list   # one record per finished question, for the end-of-test review JSON
    question_turn_start: int  # index into messages where the in-progress question's turns began
    previous_task_signature: tuple  # (domain, modality) of the most recently finished question, or None

session_config_ = None
tts = None
audio = None
audio_f = None
gui_ = None
latest_state = None
save_ace_file = None
season_actual, season_adjacent = get_season_transition(datetime.now())

def configure(session_config, tts_engine, audio, audio_fluency, gui):
    """Gets Session variables, and injects them into the scoring pipeline.
     Location, Current/Last president, pm and name for user
    """
    global session_config_, tts, audio_, audio_f, gui_
    session_config_ = session_config
    tts = tts_engine
    audio_ = audio
    audio_f = audio_fluency
    gui_ = gui
    gui_.set_on_close(save_on_close) # when gui closes save the current run

    for domain in ace_json.values():
        for question in domain["questions"]:
            original_answers = question.setdefault("_source_answers", question["answers"])
            if "DYNAMIC:season" in original_answers:
                question["season_sub_index"] = original_answers.index("DYNAMIC:season")

            if "DYNAMIC:uk_prime_minister" in original_answers and session_config.get("previous_uk_pm"):
                question["outgoing_leader"] = session_config["previous_uk_pm"]
            if "DYNAMIC:us_president" in original_answers and session_config.get("previous_us_president"):
                question["outgoing_leader"] = session_config["previous_us_president"]
            question["answers"] = resolve_dynamic_answers(original_answers, session_config)


def finalize_score(state, question, domain, q_index, score, new_sub) -> dict:
    """Apply question/domain caps, add to the running score, and clear all
    reprompt/progress state. Every handler and the default scoring path end
    here"""
    question_cap = question["score_cap"]
    question_score_so_far = state.get("question_score", 0)
    score = min(score, question_cap - question_score_so_far)

    domain_cap = ace_json[domain]["score_cap"]
    current_scores = dict(state["scores"])
    domain_score_so_far = current_scores.get(domain, 0)
    score = min(score, domain_cap - domain_score_so_far)

    current_scores[domain] = domain_score_so_far + score
    print(f"Scored {score} points for question {q_index + 1} in domain '{domain}' (total: {current_scores[domain]})")
    return {
        "scores": current_scores,
        "sub_question_index": new_sub,
        "question_score": question_score_so_far + score,
        "needs_repeat": False,
        "repeat_count": 0,
        "reprompt_kind": None,
        "turn_progress": 0,
    }


def reprompt(kind: str) -> dict:
    # Triggers a reprompt when the user asks vague follow-ups.
    # Used in Attention Season & Memory Name questions like:
    # - "What was their last name?"
    # - "Who was the previous one?"
    # - "Could it be another season?"
    return {"needs_repeat": True, "repeat_count": 0, "reprompt_kind": kind}


def handle_person_name(state, question, response, domain, q_index, sub_index):
    """
    scores normally, else reprompt for surname or outgoing leader as fallback
    if reprompt_kind is either leader or name then score instead.
    """
    outgoing = question.get("outgoing_leader")


    # Check if they named the outgoing leader
    if state.get("reprompt_kind") == "leader":
        score = score_person_name(response, [outgoing, outgoing.split()[-1]]) if outgoing else 0
        return finalize_score(state, question, domain, q_index, score, 0)
    if state.get("reprompt_kind") == "name": #Re-score after we asked for their surname
        return finalize_score(state, question, domain, q_index, score_question(response, question), 0)
    score = score_question(response, question)
    if score:
        return finalize_score(state, question, domain, q_index, score, 0)

    # Standard scorer accepts bare surname but fails on wrong first name
    if len(clean_response(response).split()) == 1:
        return reprompt("name")

    # Wrong (and not just incomplete) — if there's been a recent change of
    # leader, probe for the outgoing politician's name as an alternate point.
    if outgoing:
        return reprompt("leader")
    # Out of reprompts & wrong answer -> 0 points
    return finalize_score(state, question, domain, q_index, 0, 0)


def reprompt_text_person_name(state, question, text):
    """
    Picks the spoken follow-up text based on which reprompt_kind was just set
    by handle_person_name: "name" asks for the surname, "leader" asks for
    the outgoing politician. Returns None on a normal (non-reprompt) turn.
    """
    if state.get("reprompt_kind") == "name":
        return "And what was their surname?"
    if state.get("reprompt_kind") == "leader":
        return "Who was the previous one, before them?"
    return None


def handle_registration(state, question, response, domain, q_index, sub_index):
    """
    Repeats a word list so the patient can learn it.
    Only trial 1st counts toward the real score—later trials are just practice.
    Stops early if they get 100% right on any trial.
    """
    max_trials = question.get("max_attempts", 3)
    trial = state.get("turn_progress", 0)
    # Score current trial, but lock in the trial 1 score for the final result
    attempt_score = score_question(response, question)
    first_attempt_score = attempt_score if trial == 0 else state.get("question_score", 0)
    all_correct = attempt_score == question["score_cap"]
    trial += 1
    print(f"Registration trial {trial}/{max_trials}: {attempt_score}/{question['score_cap']} this trial "
          f"(first-attempt score: {first_attempt_score})")
    # Not perfect yet and still have tries left? Reprompt for another trial.
    if not all_correct and trial < max_trials:
        return {
            "needs_repeat": True, "repeat_count": 0,
            "turn_progress": trial, "reprompt_kind": "registration",
            "question_score": first_attempt_score,
        }
    # Done (either got max score or ran out of tries) then end with 1st try's score
    return finalize_score({**state, "question_score": 0}, question, domain, q_index, first_attempt_score, 0)


def reprompt_text_registration(state, question, text):
    """
    Repeat the same question for registration question.
    """
    return text if state.get("reprompt_kind") == "registration" else None


def handle_season(state, question, response, domain, q_index, sub_index):
    """
    Adjacent-season check: e.g. "spring" said during a summer/spring boundary
    window — reprompt once ("could it be another season?") instead of scoring
    wrong outright. sub_index scoring only fires for this one sub-answer slot
    set +-7 days.
    """
    expected_answers = question["answers"]
    if (
        state.get("reprompt_kind") != "season" and season_adjacent
        and score_fuzzy(response, [season_adjacent]) and not score_fuzzy(response, [season_actual])
    ):
        return reprompt("season")

    score = score_question(response, question, sub_index=sub_index) if sub_index < len(expected_answers) else 0
    return finalize_score(state, question, domain, q_index, score, sub_index + 1)

def reprompt_text_season(state, question, text):
    """
    Reprompt for season slightly off by patient.
    """
    return "Could it be another season?" if state.get("reprompt_kind") == "season" else None

def handle_sub_score_bands(state, question, response, domain, q_index, sub_index):
    """
    Handles multi-word repetition tests (e.g., caterpillar, eccentricity, etc.).
    Steps through words one by one, tracks correct reps, then maps the final total
    to a score band.
    """
    expected_answers = question["answers"]
    # Score the current word if we haven't run past the answer list
    word_score = score_question(response, question, sub_index=sub_index) if sub_index < len(expected_answers) else 0
    correct_so_far = state.get("turn_progress", 0) + (1 if word_score else 0)
    new_sub = sub_index + 1
    # Move to the next sub-question
    if new_sub < len(expected_answers):
        return {
            "needs_repeat": False, "repeat_count": 0,
            "sub_question_index": new_sub, "turn_progress": correct_so_far,
        }
    # All words done -> convert total correct words into the final band score
    score = scaled_count(correct_so_far, question["sub_score_bands"])
    return finalize_score(state, question, domain, q_index, score, new_sub)

def handle_name_address_trials(state, question, response, domain, q_index, sub_index):
    """
    Repeats name & address (up to 3 times) for learning.
    Only the final trial's score counts—earlier tries are just practice and get thrown out.
    """
    total_trials = question.get("trials", 3)
    # Score the current attempt
    trial = state.get("turn_progress", 0) + 1
    attempt_score = score_question(response, question)
    print(f"Name & address trial {trial}/{total_trials}: {attempt_score}/{question['score_cap']} this trial")
    # if remaining trails ask again
    if trial < total_trials:
        return {"needs_repeat": True, "repeat_count": 0, "turn_progress": trial, "reprompt_kind": "trial"}
    # Score and move on
    return finalize_score(state, question, domain, q_index, attempt_score, 0)

def reprompt_text_name_address(state, question, text):
    """
    Re-speaks the name/address stimulus on practice trials (2 & 3).
    Ditches the 1st-trial preamble ("We will do this three times...")
    and returns None once we hit the final scoring turn.
    """
    # Only run this if in the middle of learning trials
    if state.get("reprompt_kind") != "trial":
        return None
    # get the target phrase from instructions, or fall back to original text
    quotes = re.findall(r"Speak:\s*'(.*?)'", question.get("instructions", ""))
    stimulus = quotes[-1] if quotes else text
    return f"Let's do that again. {stimulus}"

def recognition_recalled(question, sub_index, state):
    """True if this recognition element's tokens were already fully credited
    in the linked delayed-recall question, so it doesn't need to be re-asked."""
    element_indices = question.get("element_recall_indices")
    if not element_indices or sub_index >= len(element_indices):
        return False
    recalled = state.get("recall_matches", {}).get(question.get("recall_key"), [])
    return all(idx < len(recalled) and recalled[idx] for idx in element_indices[sub_index])


def handle_recognition(state, question, response, domain, q_index, sub_index):
    """
    Auto-awards 1 point if the patient already recalled this item earlier.
    Otherwise, scores their multiple-choice answer normally.
    """
    new_sub = sub_index + 1
    # if they got it right during delayed recall then score and move on
    if recognition_recalled(question, sub_index, state):
        return finalize_score(state, question, domain, q_index, 1, new_sub)
    # Otherwise, score the multiple-choice response
    score = score_question(response, question, sub_index=sub_index)
    return finalize_score(state, question, domain, q_index, score, new_sub)


"""
Handlers for unique questions that dont fit generic structure.
like non linear scoring for repetition, names, recall address/name trial.
"""
_HANDLERS = {
    "person_name": (handle_person_name, reprompt_text_person_name),
    "registration": (handle_registration, reprompt_text_registration),
    "season": (handle_season, reprompt_text_season),
    "word_rep": (handle_sub_score_bands, None),
    "name_address": (handle_name_address_trials, reprompt_text_name_address),
    "recognition": (handle_recognition, None),
}


def match_kind(question, sub_index):
    """
    Reads JSON fields for specific question HANDLERS
    """
    # Memory Q3-Q6: current/former UK PM, current US president, JFK
    if question.get("match_type") == "person_name":
        return "person_name"
    # Memory Q1: 3-word learning trial (lemon, key, ball)
    if question.get("score_first_attempt_only"):
        # a short/partial recall (e.g. "lemon") is a complete, valid attempt
        return "registration"
    # Attention Q1: the season sub-answer within day/date/month/year/season
    if question.get("season_sub_index") is not None and sub_index == question["season_sub_index"]:
        return "season"
    # Language Q3: word repetition (caterpillar, eccentricity, unintelligible, statistician)
    if question.get("sub_score_bands"):
        return "word_rep"
    # Memory Q2 (name/address learning, 3 trials) and Visuospatial's delayed
    # recall question (same name/address, asked again later)
    if question.get("score_final_trial_only"):
        # a partial recall mid-trial (e.g. only 4 of 7 elements) is a genuine,
        # complete turn — classify_turn tends to read it as "incomplete" since
        # it isn't the full phrase, which would wrongly trigger a generic
        # re-ask instead of just tallying this trial's score.
        return "name_address"
    # Visuospatial recognition question: name/number/street/town/county multiple-choice
    if question.get("element_recall_indices"):
        return "recognition"
    return None

def save_on_close():
    if save_ace_file is not None:
        save_progress(save_ace_file)

def conversation_node(state: ACEState) -> dict:
    """
    Speaks the current question and captures the patient's response.
    Handles skip logic, visual/click tasks, and custom reprompt phrasing.
    """
    global save_ace_file
    save_ace_file = state
    domain = state["current_domain"]
    q_index = state["question_index"]
    sub_index = state.get("sub_question_index", 0)
    question = ace_json[domain]["questions"][q_index]
    # Skip asking if the patient already got this right during delayed recall
    if match_kind(question, sub_index) == "recognition" and recognition_recalled(question, sub_index, state):
        return {"messages": [AIMessage(content=""), HumanMessage(content="")]}

    # Route drawing/visual tasks to their own handler
    if question.get("match_type") in ("clock", "pen_paper", "cube", "infinity", "sentances"):
        return run_visual_task(state, question, tts, audio_, session_config_, gui_)

    # Route interactive click-on-screen pointing task
    if is_click_point_question(question):
        total_questions = len(ace_json[domain]["questions"])
        next_question = ace_json[domain]["questions"][q_index + 1] if q_index + 1 < total_questions else None
        return run_click_task(state, question, tts, session_config_, next_question, gui_)
    # Display stimulus image if the question calls for one (and isn't a repeat turn)
    if question.get("image") and not state.get("needs_repeat"):
        gui_.show_stimulus_image(question["image"])
    # Otherwise wipe the stage so the previous question's image doesn't linger
    elif not question.get("image"):
        gui_.get_stage_frame()
    # Grab the text prompt for the current sub-question
    prompts = get_sub_prompts(question)
    if prompts and sub_index < len(prompts):
        text = prompts[sub_index]
    else:
        spoken = parse_spoken_prompts(question)
        text = spoken[0] if spoken else question["question_text"]
    # Check if a custom reprompt function exists for this question type
    kind = match_kind(question, sub_index)
    reprompt_fn = _HANDLERS[kind][1] if kind else None
    reprompt = reprompt_fn(state, question, text) if reprompt_fn else None
    # Determine what to say out loud
    if reprompt:
        spoken_text = reprompt
    elif state.get("needs_repeat"):
        if tts.furhat:
            furhat_repeat(tts.furhat)
        spoken_text = rephrase_question(text)
    else:
        wrapper = transition_que(
            state, session_config_["patient"]["name"], domain, task_type(question), sub_index,
        )
        if wrapper and tts.furhat:
            furhat_nod(tts.furhat)
        spoken_text = f"{wrapper} {text}".strip() if wrapper else text

    # Output text to console, GUI, and Text-to-Speech
    gui_.add_message("assessor", spoken_text)
    tts.speak(spoken_text)
    print("Assessor:", spoken_text)
    # Listen for the patient's answer (up to 2 x 60 seconds retries if audio comes back empty)
    question_key = (domain, q_index, sub_index)
    for _ in range(2):
        user_input = (
            audio_f.capture_response(on_tick=gui_.pump, question_key=question_key) if domain == "Fluency"
            else audio_.capture_response(on_tick=gui_.pump, question_key=question_key)
        )
        if not user_input:
            print("[no response detected]")
            continue
        break
    else:
        user_input = ""
    # No answer after retries -> trigger a repeat
    if not user_input:
        return {
            "needs_repeat": True,
            "messages": [AIMessage(content=spoken_text), HumanMessage(content="")],
        }

    print("Patient:", user_input)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=user_input)]}


def scoring_node(state: ACEState) -> dict:
    """
    Scores the patient's last response.
    Routes to specialized handlers for complex question types, or runs standard
    classification and fuzzy matching for generic ones.
    """
    domain = state['current_domain']
    q_index = state['question_index']
    sub_index = state.get("sub_question_index", 0)
    question = ace_json[domain]['questions'][q_index]
    score_domain = question.get("score_domain", domain)

    last_message = state['messages'][-1].content

    expected_answers = question['answers']
    prompts = get_sub_prompts(question)
    is_multi = bool(prompts)

    # Grab the text that was asked for this specific sub-question turn
    if prompts and sub_index < len(prompts):
        asked_text = prompts[sub_index]
    else:
        spoken = parse_spoken_prompts(question)
        asked_text = spoken[0] if spoken else question["question_text"]

    # Non-answer + under the retry cap -> re-prompt without scoring.
    # Non-answer + cap hit -> fall through and score whatever was given.
    repeats = state.get("repeat_count", 0)
    max_repeats = question.get("max_attempts", 1)
    match_type = question.get("match_type", "")
    is_fluency = match_type in ("fluency_letter", "fluency_animal")

    kind = match_kind(question, sub_index)

    is_visual = question.get("match_type") in ("clock", "pen_paper", "cube", "infinity", "sentances")

    skip_repeat = (
        is_fluency or is_visual or match_type == "serial_sevens"
        or is_click_point_question(question)
        or (kind is not None and kind != "season")
    )

    # Re-ask (without scoring) when the LLM says the patient didn't actually answer
    # doesn't apply (drawings, clicks, fluency, serial sevens, custom handlers).
    if not skip_repeat and classify_turn(last_message, asked_text) != "answer" and repeats < max_repeats:

        return {"needs_repeat": True, "repeat_count": repeats + 1}

    uses_score_fuzzy = kind in ("season", "recognition", "person_name") or (kind is None and (is_multi or match_type in ("fuzzy", "exact", "integer")))
    if uses_score_fuzzy:
        # uses LLM to classify users final awnser - introduced to prevent spam awnsers
        # was only introduced on questions were users can spam there awnsers.
        last_message = extract_final_answer(last_message, asked_text)
    elif match_type == "serial_sevens":
        last_message = extract_serial_sevens(last_message) or last_message
    # Route to specialized scoring handler if one exists
    if kind:
        return _HANDLERS[kind][0](state, question, last_message, score_domain, q_index, sub_index)

    # Standard question scoring
    if is_multi:
        score = score_question(last_message, question, sub_index=sub_index) if sub_index < len(expected_answers) else 0
        new_sub = sub_index + 1
    else:
        score = score_question(last_message, question)
        new_sub = 0

    result = finalize_score(state, question, score_domain, q_index, score, new_sub)
    recall_key = question.get("recall_key")
    print(f"Scoring node: domain={domain}, question={q_index + 1}, sub={sub_index + 1}, "
          f"score={score}, new_sub={new_sub}, recall_key={recall_key}, is_multi={is_multi}")
    if recall_key and not is_multi: # Very last question if user recalls half of name,adress then prompt multichoice
        # Stash per-element results so a later recognition question can skip
        # elements already credited here.
        matches = score_mixed_list_detailed(last_message, expected_answers)
        result["recall_matches"] = {**state.get("recall_matches", {}), recall_key: matches}
    return result

def question_record(state: ACEState) -> dict:
    """
    Builds a review-log entry for a completed question.
    Gathers all patient turns since the question started and pairs them with the final score.
    """
    domain = state["current_domain"]
    q_index = state["question_index"]
    question = ace_json[domain]["questions"][q_index]

    # Grab all patient responses given during this question's attempts
    start = state.get("question_turn_start", 0)
    responses = [m.content for m in state["messages"][start:] if isinstance(m, HumanMessage)]

    return {
        "domain": domain,
        "question_index": q_index,
        "question_text": question.get("question_text"),
        "match_type": question.get("match_type", "fuzzy_list"),
        "response": responses,
        "score": state.get("question_score", 0),
        "score_cap": question["score_cap"],
    }

def advance_node(state: ACEState) -> dict:
    """
    Logs the finished question, resets scoring, and routes to the next question.
    Handles standard domain progression plus custom detour logic for the ACE-III
    (interleaving Fluency & Visuospatial between Memory recall steps).
    """
    domain = state["current_domain"]
    q_index = state["question_index"]
    question = ace_json[domain]["questions"][q_index]
    total_questions = len(ace_json[domain]["questions"])
    # Update log & tracking metadata
    question_log = state.get("question_log", []) + [question_record(state)]
    question_turn_start = len(state["messages"])
    previous_task_signature = (domain, task_type(question))

    if domain == "Memory" and q_index == 1:
        # left memory in the json so now i have to detour to fluency before going back to memory for the retrograde questions.
        result = {
            "current_domain": "Fluency", "question_index": 0, "sub_question_index": 0, "question_score": 0,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    elif domain == "Memory" and q_index == 5:
        # Delayed recall/recognition (Memory Q6-7)
        # asking them right after the retrograde questions; detour back
        # once Visuospatial finishes.
        q = state["domain_queue"].copy()
        next_domain = q.pop(0)
        result = {
            "current_domain": next_domain, "question_index": 0, "sub_question_index": 0, "question_score": 0,
            "domain_queue": q,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    elif domain == "Visuospatial" and q_index + 1 >= total_questions:
        # End of that detour: back to Memory for delayed recall/recognition.
        result = {
            "current_domain": "Memory", "question_index": 6, "sub_question_index": 0, "question_score": 0,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    elif domain == "Fluency" and q_index + 1 >= total_questions:
        # End of the detour:
        result = {
            "current_domain": "Memory", "question_index": 2, "sub_question_index": 0, "question_score": 0,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    elif q_index + 1 < total_questions:
        result = {
            "question_index": q_index + 1, "sub_question_index": 0, "question_score": 0,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    else:
        q = state["domain_queue"].copy()
        next_domain = q.pop(0)
        result = {
            "current_domain": next_domain,
            "question_index": 0,
            "sub_question_index": 0,
            "question_score": 0,
            "domain_queue": q,
            "question_log": question_log,
            "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }

    result = {**result, "needs_repeat": False, "repeat_count": 0,
              "reprompt_kind": None, "turn_progress": 0}

    # Auto-save progress pointing at the upcoming state
    save_progress({**state, **result})
    return result

results_dir = os.getenv("ACE_SESSION_DATA_DIR", os.path.join(os.path.dirname(__file__), "results"))


def report_node(state: ACEState) -> dict:
    """
    Finishes the ACE-III test reports the scores. Generates a JSON file and ends the session
    """
    # Log the final question since advance_node gets skipped on the last turn
    question_log = state.get("question_log", []) + [question_record(state)]
    scores = {domain: 0 for domain in ace_json}
    for record in question_log:
        scores[record["domain"]] += record["score"]
    total = sum(scores.values())
    # Console summary output
    print("\n--- ACE-III Complete ---")
    for domain, score in scores.items():
        print(f"{domain}: {score}/{ace_json[domain]['score_cap']}")
    print(f"Total: {total}/100")
    # Generate timestamped JSON report in the results directory
    os.makedirs(results_dir, exist_ok=True)
    patient_name = session_config_.get("patient", {}).get("name", "unknown")
    safe_name = "".join(c if c.isalnum() else "_" for c in str(patient_name))
    out_path = os.path.join(results_dir, f"ACE-III_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump({
            "patient": session_config_.get("patient"),
            "assessor": session_config_.get("assessor"),
            "date": datetime.now().isoformat(),
            "domain_scores": scores,
            "domain_caps": {d: ace_json[d]["score_cap"] for d in ace_json},
            "total_score": total,
            "questions": question_log,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Clean up checkpoint file so this completed run can't be resumed by accident
    progress_path = os.path.join(progress_dir, f"progress_{safe_name}.json")
    if os.path.exists(progress_path):
        os.remove(progress_path)
    global _latest_state
    _latest_state = None
    # Final wrap-up UI prompt & shutdown delay
    gui_.add_message("assessor", "Thank you — that concludes the assessment.")
    time.sleep(3)
    gui_.close()

    return {"complete": True}

# Routes Flow of StateMachine
def router(state: ACEState) -> str:
    """
    Decides the next edge to transition to in the state graph.
    Returns one of: 'repeat_question', 'next_sub_question', 'next_question', 'next_domain', or 'report'.
    """
    # Needs a reprompt/repeat turn? Hold state and ask again.
    if state.get("needs_repeat"):
        return "repeat_question"
    domain = state['current_domain']
    q_index = state['question_index']
    sub_index = state.get('sub_question_index', 0)
    question = ace_json[domain]["questions"][q_index]
    prompts = get_sub_prompts(question)
    # Sub-question step remaining?
    if prompts and sub_index < len(prompts):
        return "next_sub_question"
    total_questions = len(ace_json[domain]["questions"])
    # More questions left in current domain?
    if q_index + 1 < total_questions:
        return "next_question"
    # if Visuospatial finished Force detour back to Memory delayed recall
    elif domain == "Visuospatial":
        return "next_domain"
    # More domains left in queue? Move to next domain
    elif state['domain_queue']:
        return "next_domain"
    # Everything done -> final report node
    else:
        return "report"

progress_dir = os.path.join(results_dir, "progress")

def save_progress(state: ACEState) -> str:
    """
    Saves the full in-progress state to disk so an interrupted assessment can be resumed.
    Overwrites the same patient checkpoint file on every call.
    """
    os.makedirs(progress_dir, exist_ok=True)
    patient_name = session_config_.get("patient", {}).get("name", "unknown")
    safe_name = "".join(c if c.isalnum() else "_" for c in str(patient_name))
    out_path = os.path.join(progress_dir, f"progress_{safe_name}.json")
    # Serializes state attributes and converts LangChain messages into simple dicts
    with open(out_path, "w") as f:
        json.dump({
            "current_domain": state["current_domain"],
            "question_index": state["question_index"],
            "sub_question_index": state.get("sub_question_index", 0),
            "question_score": state.get("question_score", 0),
            "scores": state["scores"],
            "domain_queue": state["domain_queue"],
            "complete": state.get("complete", False),
            "needs_repeat": state.get("needs_repeat", False),
            "repeat_count": state.get("repeat_count", 0),
            "reprompt_kind": state.get("reprompt_kind"),
            "turn_progress": state.get("turn_progress", 0),
            "recall_matches": state.get("recall_matches", {}),
            "question_log": state.get("question_log", []),
            "question_turn_start": state.get("question_turn_start", 0),
            "previous_task_signature": state.get("previous_task_signature"),
            "messages": [{"role": m.type, "content": m.content} for m in state["messages"]],
        }, f, indent=2)
    return out_path

builder = StateGraph(ACEState)
builder.add_node("conversation", conversation_node)
builder.add_node("scoring", scoring_node)
builder.add_node("advance", advance_node)
builder.add_node("report", report_node)

builder.add_edge(START, "conversation")
builder.add_edge("conversation", "scoring")
builder.add_conditional_edges("scoring", router, {
    "repeat_question": "conversation",
    "next_sub_question": "conversation",
    "next_question": "advance",
    "next_domain": "advance",
    "report": "report",
})
builder.add_edge("advance", "conversation")
builder.add_edge("report", END)

graph = builder.compile()
