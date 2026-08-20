import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys
import threading
import queue

# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import your actual RAG backend
from scripts import grounded_generation


# ============================================================
# COLORS
# ============================================================

BG = "#F7F4EF"
CARD = "#FFFFFF"
DARK = "#4A2E1C"
BROWN = "#6B4226"
LIGHT_BROWN = "#C7AD91"
GOLD = "#9A7045"
TEXT = "#3D3027"
MUTED = "#81766C"
BORDER = "#DED5CA"
SUCCESS = "#557A55"
WARNING = "#9A7045"


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()
root.title("Osteoporosis RAG - Clinical Guideline Assistant")
root.geometry("1500x850")
root.minsize(1100, 700)
root.configure(bg=BG)


# ============================================================
# BACKGROUND DECORATION
# ============================================================

canvas = tk.Canvas(
    root,
    bg=BG,
    highlightthickness=0
)
canvas.place(relx=0, rely=0, relwidth=1, relheight=1)


def draw_bone(canvas, x, y, scale=1.0, angle=0):
    """
    Draw a simple decorative bone using Canvas.
    This is only a background decoration.
    """

    # Bone shaft
    length = 120 * scale
    width = 18 * scale

    canvas.create_line(
        x,
        y,
        x + length,
        y,
        fill="#D6C4AD",
        width=int(width),
        capstyle=tk.ROUND
    )

    # Left joint
    r = 24 * scale
    canvas.create_oval(
        x - r,
        y - r,
        x + r,
        y + r,
        fill="#E5D8C8",
        outline="#B99D7D",
        width=2
    )

    canvas.create_oval(
        x - r * 0.45,
        y - r * 0.45,
        x + r * 0.45,
        y + r * 0.45,
        fill=BG,
        outline=""
    )

    # Right joint
    canvas.create_oval(
        x + length - r,
        y - r,
        x + length + r,
        y + r,
        fill="#E5D8C8",
        outline="#B99D7D",
        width=2
    )

    canvas.create_oval(
        x + length - r * 0.45,
        y - r * 0.45,
        x + length + r * 0.45,
        y + r * 0.45,
        fill=BG,
        outline=""
    )


def draw_spine(canvas, x, y, scale=1.0):
    """
    Decorative spine.
    """

    vertebra_w = 42 * scale
    vertebra_h = 15 * scale

    for i in range(8):
        yy = y + i * 30 * scale

        canvas.create_oval(
            x - vertebra_w / 2,
            yy - vertebra_h / 2,
            x + vertebra_w / 2,
            yy + vertebra_h / 2,
            fill="#DCCAB4",
            outline="#B99D7D",
            width=1
        )

        # Small side bones
        canvas.create_line(
            x - vertebra_w / 2,
            yy,
            x - vertebra_w,
            yy - 8 * scale,
            fill="#B99D7D",
            width=2
        )

        canvas.create_line(
            x + vertebra_w / 2,
            yy,
            x + vertebra_w,
            yy - 8 * scale,
            fill="#B99D7D",
            width=2
        )


def draw_background():
    canvas.delete("all")

    w = root.winfo_width()
    h = root.winfo_height()

    # Soft network lines
    nodes = [
        (70, 150),
        (180, 90),
        (290, 180),
        (130, 310),
        (260, 370),
        (120, 560),
        (300, 630),
        (w - 120, 140),
        (w - 240, 220),
        (w - 100, 360),
        (w - 250, 470),
        (w - 150, 650),
    ]

    connections = [
        (0, 1),
        (1, 2),
        (0, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (7, 8),
        (8, 9),
        (8, 10),
        (10, 11),
    ]

    for a, b in connections:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]

        canvas.create_line(
            x1, y1, x2, y2,
            fill="#E5DDD3",
            width=2
        )

    # Nodes
    for x, y in nodes:
        canvas.create_oval(
            x - 6,
            y - 6,
            x + 6,
            y + 6,
            fill=LIGHT_BROWN,
            outline=""
        )

    # Decorative bones
    draw_bone(canvas, 70, 210, 0.8)
    draw_bone(canvas, w - 330, 250, 0.75)
    draw_bone(canvas, 80, h - 170, 0.65)
    draw_bone(canvas, w - 350, h - 180, 0.7)

    # Spine decorations
    draw_spine(canvas, 180, 430, 0.65)
    draw_spine(canvas, w - 170, 450, 0.65)


root.after(100, draw_background)
root.bind("<Configure>", lambda event: draw_background())


# ============================================================
# MAIN CONTENT FRAME
# ============================================================

main_frame = tk.Frame(
    root,
    bg=BG
)
main_frame.place(
    relx=0.5,
    rely=0.5,
    anchor="center",
    relwidth=0.72,
    relheight=0.88
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    main_frame,
    bg=BG
)
header.pack(
    fill="x",
    pady=(10, 20)
)


title = tk.Label(
    header,
    text="OSTEOPOROSIS RAG",
    font=("Segoe UI", 28, "bold"),
    fg=DARK,
    bg=BG
)
title.pack()


subtitle = tk.Label(
    header,
    text="Clinical Practice Guideline Assistant",
    font=("Segoe UI", 14),
    fg=TEXT,
    bg=BG
)
subtitle.pack(pady=(4, 10))


# Decorative separator
separator = tk.Frame(
    header,
    bg=GOLD,
    height=2
)
separator.pack(fill="x", padx=100)


# ============================================================
# MAIN CARD
# ============================================================

card = tk.Frame(
    main_frame,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)

card.pack(
    fill="both",
    expand=True,
    padx=25
)


# ============================================================
# CLINICAL QUESTION
# ============================================================

question_label = tk.Label(
    card,
    text="Clinical Question",
    font=("Segoe UI", 13, "bold"),
    fg=TEXT,
    bg=CARD
)
question_label.pack(
    anchor="w",
    padx=30,
    pady=(25, 7)
)


question_entry = tk.Entry(
    card,
    font=("Segoe UI", 12),
    fg=TEXT,
    bg="#FCFBF9",
    relief="flat",
    highlightthickness=1,
    highlightbackground=BORDER,
    highlightcolor=GOLD
)

question_entry.pack(
    fill="x",
    padx=30,
    ipady=10
)

question_entry.insert(
    0,
    "Enter your clinical question here..."
)


def clear_placeholder(event):
    if question_entry.get() == "Enter your clinical question here...":
        question_entry.delete(0, tk.END)


question_entry.bind("<FocusIn>", clear_placeholder)


# ============================================================
# OPTIONS ROW
# ============================================================

options_frame = tk.Frame(
    card,
    bg=CARD
)
options_frame.pack(
    fill="x",
    padx=30,
    pady=(18, 5)
)


# ---------------- TOP K ----------------

topk_frame = tk.Frame(
    options_frame,
    bg=CARD
)
topk_frame.pack(
    side="left",
    padx=(0, 35)
)


tk.Label(
    topk_frame,
    text="Top-K Evidence",
    font=("Segoe UI", 11, "bold"),
    fg=TEXT,
    bg=CARD
).pack(
    side="left",
    padx=(0, 10)
)


topk_var = tk.IntVar(value=3)

topk_spinbox = tk.Spinbox(
    topk_frame,
    from_=1,
    to=10,
    textvariable=topk_var,
    width=4,
    font=("Segoe UI", 11),
    justify="center",
    relief="solid",
    bd=1
)

topk_spinbox.pack(side="left")


# ---------------- SEARCH MODE ----------------

mode_frame = tk.Frame(
    options_frame,
    bg=CARD
)
mode_frame.pack(
    side="left"
)


tk.Label(
    mode_frame,
    text="Search Mode",
    font=("Segoe UI", 11, "bold"),
    fg=TEXT,
    bg=CARD
).pack(
    side="left",
    padx=(0, 10)
)


mode_var = tk.StringVar(value="hybrid")

mode_combo = ttk.Combobox(
    mode_frame,
    textvariable=mode_var,
    values=[
        "hybrid",
        "semantic",
        "keyword"
    ],
    state="readonly",
    width=12
)

mode_combo.pack(side="left")


# ============================================================
# STATUS
# ============================================================

status_frame = tk.Frame(
    card,
    bg="#F7F2EC"
)
status_frame.pack(
    fill="x",
    padx=30,
    pady=(8, 15)
)


status_label = tk.Label(
    status_frame,
    text="● Ready — Clinical assistant is ready",
    font=("Segoe UI", 10),
    fg=SUCCESS,
    bg="#F7F2EC"
)

status_label.pack(
    anchor="w",
    padx=12,
    pady=8
)


# ============================================================
# ASK BUTTON
# ============================================================

button_frame = tk.Frame(
    card,
    bg=CARD
)
button_frame.pack(
    fill="x",
    padx=30,
    pady=(0, 15)
)


ask_button = tk.Button(
    button_frame,
    text="ASK",
    font=("Segoe UI", 12, "bold"),
    fg="white",
    bg=BROWN,
    activebackground=DARK,
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    bd=0
)

ask_button.pack(
    ipadx=55,
    ipady=9
)


# ============================================================
# OUTPUT AREA
# ============================================================

output_title_frame = tk.Frame(
    card,
    bg=CARD
)
output_title_frame.pack(
    fill="x",
    padx=30
)


tk.Label(
    output_title_frame,
    text="Grounded Clinical Answer",
    font=("Segoe UI", 13, "bold"),
    fg=TEXT,
    bg=CARD
).pack(
    side="left"
)


# Confidence indicator
confidence_label = tk.Label(
    output_title_frame,
    text="Confidence: —",
    font=("Segoe UI", 9, "bold"),
    fg=MUTED,
    bg=CARD
)
confidence_label.pack(
    side="right"
)


# ============================================================
# ANSWER TEXT
# ============================================================

answer_frame = tk.Frame(
    card,
    bg=CARD
)

answer_frame.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=(7, 25)
)


answer_text = tk.Text(
    answer_frame,
    font=("Segoe UI", 11),
    fg=TEXT,
    bg="#FCFBF9",
    relief="flat",
    wrap="word",
    padx=15,
    pady=12,
    highlightthickness=1,
    highlightbackground=BORDER,
    state="disabled"
)

answer_text.pack(
    side="left",
    fill="both",
    expand=True
)


scrollbar = ttk.Scrollbar(
    answer_frame,
    orient="vertical",
    command=answer_text.yview
)

scrollbar.pack(
    side="right",
    fill="y"
)

answer_text.configure(
    yscrollcommand=scrollbar.set
)


# ============================================================
# BOTTOM INFORMATION
# ============================================================

info_frame = tk.Frame(
    main_frame,
    bg=BG
)

info_frame.pack(
    fill="x",
    pady=(10, 0)
)


info_text = tk.Label(
    info_frame,
    text=(
        "Retrieval: Keyword + Semantic + Hybrid   •   "
        "Grounded Generation   •   Claim Verification"
    ),
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=BG
)

info_text.pack()


# ============================================================
# QUEUE FOR THREAD RESULTS
# ============================================================

result_queue = queue.Queue()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_answer(text):
    answer_text.configure(state="normal")
    answer_text.delete("1.0", tk.END)
    answer_text.insert("1.0", text)
    answer_text.configure(state="disabled")


def set_status(text, color=SUCCESS):
    status_label.configure(
        text=f"● {text}",
        fg=color
    )


def extract_confidence(response):
    """
    Try to extract confidence information from the response
    without assuming a specific response structure.
    """

    if not isinstance(response, dict):
        return None

    possible_keys = [
        "confidence",
        "confidence_score",
        "confidence_level",
        "retrieval_confidence"
    ]

    for key in possible_keys:
        if key in response:
            return response[key]

    return None


def process_queue():
    try:
        while True:

            result = result_queue.get_nowait()

            if result["type"] == "success":

                response = result["response"]

                output = response.get(
                    "output_text",
                    "No grounded answer was returned."
                )

                set_answer(output)

                confidence = extract_confidence(response)

                if confidence is not None:
                    confidence_label.configure(
                        text=f"Confidence: {confidence}",
                        fg=SUCCESS
                    )
                else:
                    confidence_label.configure(
                        text="Confidence: Available in backend",
                        fg=SUCCESS
                    )

                set_status(
                    "Grounded response generated successfully",
                    SUCCESS
                )

                ask_button.configure(
                    state="normal",
                    text="ASK"
                )

            elif result["type"] == "error":

                set_answer(
                    "An error occurred while processing the clinical question:\n\n"
                    + str(result["error"])
                )

                confidence_label.configure(
                    text="Confidence: —",
                    fg=MUTED
                )

                set_status(
                    "Error while running RAG pipeline",
                    "#9B4B45"
                )

                ask_button.configure(
                    state="normal",
                    text="ASK"
                )

    except queue.Empty:
        pass

    root.after(100, process_queue)


# ============================================================
# RUN RAG
# ============================================================

def run_rag():

    query = question_entry.get().strip()

    if not query or query == "Enter your clinical question here...":
        messagebox.showwarning(
            "Missing Question",
            "Please enter a clinical question first."
        )
        return

    try:
        top_k = int(topk_var.get())
    except ValueError:
        messagebox.showwarning(
            "Invalid Top-K",
            "Top-K must be a number between 1 and 10."
        )
        return

    mode = mode_var.get()

    ask_button.configure(
        state="disabled",
        text="PROCESSING..."
    )

    set_status(
        f"Running {mode.upper()} retrieval and grounded generation...",
        WARNING
    )

    confidence_label.configure(
        text="Confidence: Processing...",
        fg=WARNING
    )

    set_answer(
        "Retrieving clinical evidence...\n\n"
        "Please wait while the RAG pipeline generates a grounded answer."
    )

    def worker():

        try:

            response = grounded_generation.run(
                query,
                top_k=top_k,
                mode=mode
            )

            result_queue.put({
                "type": "success",
                "response": response
            })

        except Exception as e:

            result_queue.put({
                "type": "error",
                "error": e
            })

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


# ============================================================
# BUTTON COMMAND
# ============================================================

ask_button.configure(
    command=run_rag
)


# ============================================================
# ENTER KEY
# ============================================================

question_entry.bind(
    "<Return>",
    lambda event: run_rag()
)


# ============================================================
# START QUEUE PROCESSOR
# ============================================================

root.after(100, process_queue)


# ============================================================
# RUN GUI
# ============================================================

root.mainloop()
