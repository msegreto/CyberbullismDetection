import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from joblib import load
from utils.text_preprocessing import preprocess_text
from PIL import Image, ImageTk
import os


label_map = {
    0: "age",
    1: "ethnicity",
    2: "gender",
    3: "other_cyberbullying",
    4: "religion"
}

# vocabulary loading
bynary_voc = load("model/tfidf_vocabulary.pkl")
multiclass_voc = load("model/tfidf_vocabulary_multiclass.pkl")
# binary model loading
binary_clf = load("model/grid_search_binary_f1/RandomForest_TF-IDF.pkl")
# Carica classificatore multiclass
multiclass_clf = load("model/grid_search_multiclass/RandomForest_TF-IDF_multiclass.pkl")

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
            result_label.after(1000, explanation_window(predicted_label))

        else:
            result_label.config(text="Not Cyberbullying", fg="green")

    except Exception as e:
        result_label.config(text="Errore nella classificazione", fg="orange")
        print("Errore:", e)



def explanation_window(predicted_label):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    import pandas as pd
    import os

    # new window creation
    new_win = tk.Toplevel()
    new_win.title("Explanation")

    screen_width = new_win.winfo_screenwidth()
    screen_height = new_win.winfo_screenheight()
    win_width = screen_width // 2
    win_height = screen_height // 2
    new_win.geometry(f"{win_width}x{win_height}+100+100")  
    new_win.update_idletasks()

    # ------------------------
    # CONTAINER: top 25 word + association rules
    # ------------------------
    image_and_rules_frame = tk.Frame(new_win)
    image_and_rules_frame.pack(fill=tk.BOTH, expand=True)

    # ------------------------
    # IMMAGINE (up)
    # ------------------------
    image_frame = tk.Frame(image_and_rules_frame)
    image_frame.pack(side=tk.TOP, anchor="n", pady=10)

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
    # REGOLE (bottom, scrollable)
    # ------------------------
    try:
        rules_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "association_rules", "association_rules.csv"))
        filtered_rules = rules_df[rules_df['Class'] == predicted_label]['Rules'].tolist()
    except Exception as e:
        filtered_rules = [f"Could not load rules: {e}"]

    rules_container = tk.Frame(image_and_rules_frame)
    rules_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    canvas = tk.Canvas(rules_container)
    scrollbar = ttk.Scrollbar(rules_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    half_width = int(win_width * 0.5)
    if filtered_rules:
        for i, rule in enumerate(filtered_rules):
            bg_color = "#ffffff" if i % 2 == 0 else "#e6e6e6"
            tk.Label(
                scrollable_frame,
                text=rule,
                anchor="w",
                justify="left",
                wraplength=half_width,
                bg=bg_color,
                fg="#222222",
                padx=5,
                pady=4,
                font=("Arial", 10)
            ).pack(fill="x")
    else:
        tk.Label(scrollable_frame, text="No rules found for this class.").pack()



# Loading CSV example previously selected
def load_examples():
    try:
        df = pd.read_csv("dataset/selected_explanable_example.csv")
        return [f"[{row['label_name']}] {row['text']}" for _, row in df.iterrows()]
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

