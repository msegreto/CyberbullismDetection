import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from joblib import load
from PIL import Image, ImageTk
from treeinterpreter import treeinterpreter as ti
import numpy as np

import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.text_preprocessing import preprocess_text


label_map = {
    0: "age",
    1: "ethnicity",
    2: "gender",
    3: "other_cyberbullying",
    4: "religion"
}

# vocabulary loading
bynary_voc = load("../model/tfidf_vocabulary.pkl")
multiclass_voc = load("../model/tfidf_vocabulary_multiclass.pkl")
# binary model loading
binary_clf = load("../model/grid_search_binary_f1/RandomForest_TF-IDF.pkl")
# Carica classificatore multiclass
multiclass_clf = load("../model/grid_search_multiclass/RandomForest_TF-IDF_multiclass.pkl")

# check botton call
def on_check():
    text = message_entry.get()
    if not text.strip():
        messagebox.showwarning("Warning", "The system cannot analyze an empty message")
        return

    # Preprocessing
    preprocessed = preprocess_text(text)

    try:
        #vectorization
        X = bynary_voc.transform([preprocessed])

        # binary prediction
        prediction = binary_clf.predict(X)[0]

        if prediction == 1:

            X = multiclass_voc.transform([preprocessed])
            # multiclass prediction
            multiclass_prediction = multiclass_clf.predict(X)[0]

            predicted_label = label_map.get(multiclass_prediction, "Unknown")
            
            result_label.config(
                text=f"Cyberbullying Message: {predicted_label}",
                fg="red"
            )
            result_label.after(1000, explanation_window(predicted_label, preprocessed ))

        else:
            result_label.config(text="Not Cyberbullying", fg="green")

    except Exception as e:
        result_label.config(text="Error during classification", fg="orange")
        print("Errore:", e)



def explanation_window(predicted_label, preprocessed):

    new_win = tk.Toplevel()
    new_win.title("Explanation")

    screen_width = new_win.winfo_screenwidth()
    screen_height = new_win.winfo_screenheight()
    win_width = screen_width // 2
    win_height = screen_height // 2
    new_win.geometry(f"{win_width}x{win_height}+100+100")
    new_win.update_idletasks()

    # ------------------------
    # MAIN FRAME
    # ------------------------
    main_frame = tk.Frame(new_win)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # LEFT
    left_frame = tk.Frame(main_frame, width=win_width // 2, bg="#f0f0f0")
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

    # RIGHT
    right_frame = tk.Frame(main_frame, width=win_width // 2)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ------------------------
    # PICTURE RIGHT+BOTTOM
    # ------------------------
    image_frame = tk.Frame(right_frame)
    image_frame.pack(side=tk.TOP, pady=10)

    image_filename = f"top_25_{predicted_label}.png"
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "word_class_distribution", image_filename))

    if os.path.isfile(image_path):
        try:
            img = Image.open(image_path)
            img = img.resize((
                int(win_width * 0.5),
                int(win_height * 0.5)
            ), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            img_label = tk.Label(image_frame, image=img_tk)
            img_label.image = img_tk
            img_label.pack()
        except Exception as e:
            tk.Label(image_frame, text="Error loading image").pack()
            print("Image error:", e)
    else:
        tk.Label(image_frame, text="Image not found").pack()
        print(f"Not found: {image_path}")

    # ------------------------
    # RULES RIGHT+BOTTOM
    # ------------------------
    try:
        rules_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "association_rules", "association_rules.csv"))
        filtered_rules = rules_df[rules_df['Class'] == predicted_label]['Rules'].tolist()
    except Exception as e:
        filtered_rules = [f"Could not load rules: {e}"]

    rules_container = tk.Frame(right_frame)
    rules_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    title_label = tk.Label(rules_container, text="Association Rules", font=("Arial", 12, "bold"), anchor="w")
    title_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

    canvas = tk.Canvas(rules_container)
    scrollbar = ttk.Scrollbar(rules_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def resize_scrollable(event):
        canvas.itemconfig(window_id, width=event.width)

    canvas.bind("<Configure>", resize_scrollable)

    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    if filtered_rules:
        for i, rule in enumerate(filtered_rules):
            bg_color = "#ffffff" if i % 2 == 0 else "#e6e6e6"
            tk.Label(
                scrollable_frame,
                text=rule,
                anchor="w",
                justify="left",
                wraplength=int(win_width * 0.5 - 20),
                bg=bg_color,
                fg="#222222",
                padx=5,
                pady=4,
                font=("Arial", 10)
            ).pack(fill="x")
    else:
        tk.Label(scrollable_frame, text="No rules found for this class.").pack()
    
    # ------------------------
    # LEFT TREE INTERPRETER
    # ------------------------
    add_treeinterpreter_table(left_frame, preprocessed, multiclass_clf.named_steps["model"] , multiclass_voc)

def add_treeinterpreter_table(left_frame, preprocessed, model, vectorizer):

    # Vectorization
    x = vectorizer.transform([preprocessed]).toarray().astype("float32")
    feature_names = vectorizer.get_feature_names_out()

    # prediction
    prediction, bias, contributions = ti.predict(model, x)
    predicted_label_id = np.argmax(prediction)

    # dataframe construction
    data = []
    for i in range(len(feature_names)):
        data.append({
            "feature": feature_names[i],
            "contribution": contributions[0][i][predicted_label_id],
            "tfidf_value": x[0][i]
        })

    df_interp = pd.DataFrame(data)
    df_interp_sorted = df_interp.reindex(df_interp.contribution.abs().sort_values(ascending=False).index)
    df_top_50 = df_interp_sorted.head(50)

    container = tk.Frame(left_frame)
    container.pack(fill="both", expand=True, padx=10, pady=10)

    title_label = tk.Label(container, text="Top 50 Features (TreeInterpreter)", font=("Arial", 12, "bold"))
    title_label.pack(pady=(0, 5), anchor="w")

    tree_scroll = ttk.Scrollbar(container)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    tree = ttk.Treeview(container, yscrollcommand=tree_scroll.set)
    tree.pack(fill="both", expand=True)
    tree_scroll.config(command=tree.yview)

    tree["columns"] = ("feature", "contribution", "tfidf")
    tree["show"] = "headings"

    tree.heading("feature", text="Feature")
    tree.heading("contribution", text="Contribution")
    tree.heading("tfidf", text="TF-IDF")

    tree.column("feature", anchor="w", width=120)
    tree.column("contribution", anchor="center", width=100)
    tree.column("tfidf", anchor="center", width=100)

    for _, row in df_top_50.iterrows():
        tree.insert("", tk.END, values=(
            row["feature"],
            f"{row['contribution']:.4f}",
            f"{row['tfidf_value']:.4f}"
        ))

# Loading CSV example previously selected
def load_examples():
    try:
        df = pd.read_csv("../dataset/selected_explanable_example.csv")
        return [f"[{row['label_name']}] {row['original_text']}" for _, row in df.iterrows()]
    except Exception as e:
        print("Error during example loading:", e)
        return []

# Here the selected example is dragged to the input box ready to be checked
def on_select_example(*args):
    selected = example_var.get()
    if selected != placeholder:
        message_entry.delete(0, tk.END)
        if "]" in selected:
            text_only = selected.split("]", 1)[1].strip()
        else:
            text_only = selected
        message_entry.insert(0, text_only)

# Here follows the window element creation
root = tk.Tk()
root.title("Cyberbullism Detection")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = screen_width // 2
window_height = screen_height // 2
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.5, anchor="center")

title = tk.Label(frame, text="Check your message🕵🏼‍♂️", font=("Arial", 18))
title.pack(pady=10)

message_entry = tk.Entry(frame, width=50, font=("Arial", 14))
message_entry.pack(pady=5)

examples = load_examples()
placeholder = "Harmful example"
example_var = tk.StringVar(value=placeholder)

button_frame = tk.Frame(frame)
button_frame.pack(pady=5)

check_button = tk.Button(button_frame, text="Check", command=on_check, width=18)
check_button.pack(side="left", padx=(0, 5))

example_menu = tk.OptionMenu(button_frame, example_var, *examples)
example_menu.config(width=18)
example_menu.pack(side="left", padx=(5, 0))

example_var.trace_add("write", on_select_example)

result_label = tk.Label(frame, text="", font=("Arial", 16))
result_label.pack(pady=10)

# App Start
root.mainloop()

