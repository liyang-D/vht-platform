import os
import threading
import time
from rapidfuzz import fuzz
from langchain_core.messages import AIMessage, HumanMessage
from camera.capture import capture_drawing, record_video
from LLM.dialogue import transition_que, rephrase_question, check_in, is_finished_drawing
from marking.marking import parse_spoken_prompts
from voice.capture import max_response

draw_timer = 180 #3 mins
video_timer = 60 #1 min, hard cap on the recording

grid_columns, grid_rows = 3, 4
grid_items = [
    "spoon", "book", "kangaroo",
    "penguin", "anchor", "camel",
    "harp", "rhinoceros", "barrel",
    "crown", "crocodile", "accordion",
]

def is_draw_task(question: dict) -> bool:
    """ Helper function to check if its a drawing task """
    text = question.get("match_type", "")
    target_types = ("clock", "cube", "infinity", "sentances")
    return text in target_types

def is_video_task(question: dict) -> bool:
    """Checks to see if its a video task (pen paper task)"""
    return question.get("question_text", "").startswith("Comprehension: Follow three-stage commands")

def is_click_point_question(question: dict) -> bool:
    """Checks to see if the question is pointing at pictures"""
    return question.get("question_text", "").startswith("Comprehension: Which picture")

def task_type(question: dict) -> str:
    """Categorizes a question into a task-type tag ('draw', 'video', 'click', or 'spoken').
    Used to detect transition between tasks"""
    if is_draw_task(question):
        return "draw"
    if is_video_task(question):
        return "video"
    if is_click_point_question(question):
        return "click"
    return "spoken"

def run_click_task(state, question: dict, tts, session_config: dict, next_question: dict | None, gui) -> dict:
    """
    Handles click selection tasks.
    """
    # Extract spoken prompts or fallback to standard question text
    spoken = parse_spoken_prompts(question)
    text = spoken[0] if spoken else question["question_text"]

    # Determine assessor vocal output based on system state
    if state.get("needs_repeat"):
        spoken_text = rephrase_question(text)
    elif state.get("previous_task_signature") != (state["current_domain"], "click"):
        # Add a transition instruction if switching into a click-based task
        spoken_text = f"You will need to click on the screen now. {text}"
    else:
        spoken_text = text

    # Log and speak the instruction to the patient
    print("Assessor:", spoken_text)
    gui.add_message("assessor", spoken_text)
    tts.speak(spoken_text)

    # Determine if the click UI canvas should stay open after this question finishes
    image_path = question.get("image")
    keep_open = bool(
        next_question and is_click_point_question(next_question) and next_question.get("image") == image_path
    )
    # Display the click grid UI and wait for interaction
    clicked_index = gui.launch_click_canvas(image_path, keep_open, grid_columns, grid_rows) if image_path else None

    # Handle scenario where the user fails to click or times out
    if clicked_index is None:
        return {"needs_repeat": True}

    # Map grid coordinate index to item name
    clicked_item = grid_items[clicked_index]
    print("Patient (pointed to):", clicked_item)
    gui.add_message("patient", clicked_item)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=clicked_item)]}


def run_visual_task(state, question: dict, tts, audio, session_config: dict, gui) -> dict:
    """
    Handles standard visual tasks, drawing tasks (camera capture),
    and multi-stage command tasks (video capture).
    """

    spoken = parse_spoken_prompts(question)
    text = spoken[0] if spoken else question["question_text"]
    # Display reference stimulus image in GUI unless it's a drawing task
    # (drawing tasks render the stimulus inside their own specialized window layout)
    if question.get("image") and not is_draw_task(question):
        gui.show_stimulus_image(question["image"])

    # Prepare speech text and dynamic conversational transitions
    wrapper = ""
    if state.get("needs_repeat"):
        spoken_text = rephrase_question(text)
    else:
        wrapper = transition_que( # Transitions ques
            state, session_config["patient"]["name"], state["current_domain"],
            task_type(question), state.get("sub_question_index", 0),
        )
        spoken_text = f"{wrapper} {text}".strip() if wrapper else text
    print("Assessor:", spoken_text)
    gui.add_message("assessor", spoken_text)

    if is_draw_task(question): # Responsible for generating conversation during the drawing tasks
        if wrapper:
            tts.speak(wrapper)
        for prompt in spoken:
            tts.speak(prompt)
        time.sleep(0.45)

        task_name = question["question_text"].split(":")[0].lower().replace(" ", "_")
        output_path = os.path.join(os.path.dirname(__file__), f"{task_name}.png")

        gui.start_draw_task(draw_timer, question.get("image"))
        deadline = time.time() + draw_timer
        last_spoken = ""
        finished = False
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            heard = audio.capture_response(
                on_tick=gui.tick_draw_panel,
                max_duration=max(min(remaining, max_response), 3),
            )
            if (heard and fuzz.partial_ratio(heard.lower(), last_spoken.lower()) < 75
                    and is_finished_drawing(heard)):
                finished = True
                break
            if deadline - time.time() > 15:  # dont check in 15 seconds before the end
                last_spoken = check_in()
                tts.speak(last_spoken)
        gui.end_draw_panel(finished)

        tts.speak("Okay, now show me.")
        capture_drawing(output_path)
        # Clear "Finished!" and the hold-to-camera prompt so they don't linger into the next question
        gui.get_stage_frame()
        gui.pump()
        return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=output_path)]}

    if is_video_task(question):
        if wrapper:
            tts.speak(wrapper)

        practice, scored_prompts = (spoken[0], spoken[1:]) if len(spoken) > 1 else (None, spoken)
        if practice:
            tts.speak(practice)
            audio.capture_response(on_tick=gui.pump)  # practice trial, not scored did not enfore the
            # first question being a practice run
        output_path = os.path.join(os.path.dirname(__file__), "..", "data", "videos", "pen_paper.mp4")

        stop_event = threading.Event()
        recorder = threading.Thread(
            target=record_video, args=(output_path, stop_event),
            kwargs={"max_duration": video_timer}, daemon=True,
        )
        recorder.start()
        gui.start_video_panel()
        try:
            for prompt in scored_prompts:
                gui.set_video_prompt(prompt)
                tts.speak(prompt)
                gui.wait(8)
        finally:
            stop_event.set()
            recorder.join(timeout=5)
        gui.end_video_panel()
        return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=output_path)]}

    tts.speak(spoken_text)

    # Retry audio capture up to 5 times to ignore silence or TTS echo feedback
    for _ in range(5):
        user_input = audio.capture_response(on_tick=gui.pump)
        if not user_input:
            print("[no response detected]")
            continue
        if fuzz.partial_ratio(user_input.lower(), spoken_text.lower()) > 75:
            print("[echo detected, ignoring]")
            continue
        break
    else:
        return {"needs_repeat": True}

    print("Patient:", user_input)
    gui.add_message("patient", user_input)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=user_input)]}
