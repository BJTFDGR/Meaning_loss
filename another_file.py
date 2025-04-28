import os
PATH = '/home/chenboc1/localscratch2/chenboc1/Meaning_loss/cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
# --- Imports ---
import torch
import re
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset # Import Dataset class
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json # For saving/loading preprocessed data list
import random

# --- Configuration ---
# --- Choose your model ---
MODEL_NAME = "bert-base-uncased"
# Examples:
# MODEL_NAME = "openai-community/gpt2-large"
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# --- Choose your dataset ---
DATASET_NAME = "openwebtext"
DATASET_SPLIT = "train"

# --- Experiment Parameters ---
TARGET_WORD = "sure"
CONTEXT_LENGTHS_TO_TEST = [100, 200, 300, 500, 1000]
NUM_SAMPLES_PER_LENGTH = 500
# --- Preprocessing Parameters ---
NUM_PREFILTERED_EXAMPLES = 2000 # Target number of examples containing the word
MODEL_MAX_LENGTH = 512
PREFILTERED_DATA_DIR = "./prefiltered_data" # Directory to save/load filtered data
FORCE_REFILTER = True # Set to True to re-run filtering even if file exists
# --- Create directory for filtered data ---
os.makedirs(PREFILTERED_DATA_DIR, exist_ok=True)
# Define filename based on dataset and target word
filtered_data_file = os.path.join(PREFILTERED_DATA_DIR, f"{DATASET_NAME.replace('/','_')}_{TARGET_WORD}_{NUM_PREFILTERED_EXAMPLES}.jsonl")

# --- <<<< UPDATED Preprocessing Section >>>> ---
filtered_examples = []

if os.path.exists(filtered_data_file) and not FORCE_REFILTER:
    print(f"Loading pre-filtered data from: {filtered_data_file}")
    try:
        with open(filtered_data_file, 'r') as f:
            for line in f:
                filtered_examples.append(json.loads(line))
        print(f"Loaded {len(filtered_examples)} examples.")
        if len(filtered_examples) < NUM_PREFILTERED_EXAMPLES:
             print(f"Warning: Loaded file contains fewer examples ({len(filtered_examples)}) than requested ({NUM_PREFILTERED_EXAMPLES}).")
    except Exception as e:
        print(f"Error loading pre-filtered data: {e}. Will attempt to re-filter.")
        filtered_examples = []

if not filtered_examples:
    print("-" * 50)
    print(f"Preprocessing: Filtering dataset '{DATASET_NAME}' using split() to find {NUM_PREFILTERED_EXAMPLES} examples containing '{TARGET_WORD}'...")
    print("-" * 50)
    start_filter_time = time.time()
    try:
        # Load the streaming dataset
        raw_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True, trust_remote_code=True)

        count = 0
        processed_count = 0
        # Iterate through the stream and collect examples
        for example in raw_dataset:
            processed_count += 1
            text = example.get('text', '')
            if text:
                # --- Use text.split() for filtering ---
                # This check is case-sensitive and requires exact word match after splitting by whitespace.
                # It will miss words attached to punctuation (e.g., "sorry.")
                try:
                    if TARGET_WORD in text.split():
                        filtered_examples.append({'text': text}) # Store only the text
                        count += 1
                        if count % 100 == 0:
                            print(f"  Found {count}/{NUM_PREFILTERED_EXAMPLES} examples... (Processed {processed_count})")
                        if count >= NUM_PREFILTERED_EXAMPLES:
                            print(f"Reached target of {NUM_PREFILTERED_EXAMPLES} filtered examples.")
                            break # Stop after finding enough examples
                except Exception as split_error:
                    # Handle potential errors if text is unusual for split()
                    # print(f"Warning: Could not split text sample: {split_error}") # Debug
                    pass

        end_filter_time = time.time()
        print(f"Filtering finished in {end_filter_time - start_filter_time:.2f} seconds. Found {len(filtered_examples)} examples.")

        # Save the filtered data to disk as JSON Lines
        if filtered_examples:
            print(f"Saving filtered data to: {filtered_data_file}")
            try:
                with open(filtered_data_file, 'w') as f:
                    for item in filtered_examples:
                        f.write(json.dumps(item) + '\n')
            except Exception as e:
                print(f"Error saving filtered data: {e}")
        else:
             print("No examples containing the target word were found using split().")
             exit() # Exit if no relevant data found

    except Exception as e:
        print(f"Error during dataset filtering: {e}")
        exit() # Exit if filtering fails

# --- Load the preprocessed data into a datasets.Dataset object for easier handling ---
if filtered_examples:
     print(f"Using {len(filtered_examples)} pre-filtered examples.")
else:
     print("No pre-filtered examples available. Exiting.")
     exit()
# --- <<<< END OF PREPROCESSING SECTION >>>> ---


# --- 1. Load Model and Tokenizer (Same as before) ---
print(f"\nLoading model: {MODEL_NAME}...")
# ... (Keep the model loading code from the previous version here) ...
# --- (Paste Model Loading Block Here) ---
if "llama" in MODEL_NAME.lower():
     try:
         from huggingface_hub import HfFolder
         if HfFolder.get_token() is None:
             print("="*80); print(" Llama model detected. Please login to Hugging Face Hub."); print(" Run: huggingface-cli login"); print("="*80)
     except ImportError: print("Warning: huggingface_hub not installed. Login might be required for Llama models.")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    print(f"Model loaded successfully on device: {device}")
    MODEL_MAX_LENGTH = tokenizer.model_max_length
    if MODEL_MAX_LENGTH is None or MODEL_MAX_LENGTH > 100000: 
        MODEL_MAX_LENGTH = 2048; print(f"Warning: Model max length not found, falling back to {MODEL_MAX_LENGTH}.")
    print(f"Model max sequence length: {MODEL_MAX_LENGTH}")
    valid_context_lengths = []
    for length in CONTEXT_LENGTHS_TO_TEST:
        if length > MODEL_MAX_LENGTH: print(f"Warning: CONTEXT_LENGTH_TARGET {length} exceeds model max length {MODEL_MAX_LENGTH}. Skipping.")
        else: valid_context_lengths.append(length)
    CONTEXT_LENGTHS_TO_TEST = valid_context_lengths
    if not CONTEXT_LENGTHS_TO_TEST: print("Error: No valid context lengths to test."); exit()
    print(f"Will test context lengths: {CONTEXT_LENGTHS_TO_TEST}")
except Exception as e: print(f"Error loading model '{MODEL_NAME}': {e}"); exit()
# --- (End of Pasted Block) ---


# --- 2. Helper Functions ---
# (Keep calculate_cosine_similarity as before)
# (Keep get_word_embedding_in_context as before - it still needs precise token finding)
# --- (Paste Helper Functions Here) ---
def get_word_embedding_in_context(text, target_word, tokenizer, model, device):
    """Gets the embedding for a target word within a given text context."""
    # Note: Ensure text length respects model's max length if not using a long-context model
    # You might need to truncate or implement a sliding window here depending on strategy.
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=MODEL_MAX_LENGTH)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    target_token_ids = tokenizer.encode(' ' + target_word, add_special_tokens=False)
    token_indices = []
    input_ids_list = input_ids[0].tolist()
    for i in range(len(input_ids_list) - len(target_token_ids) + 1):
        if input_ids_list[i:i+len(target_token_ids)] == target_token_ids:
            token_indices.extend(list(range(i, i+len(target_token_ids))))
            break # Use first occurrence
    if not token_indices:
        print(text)
        print(target_word)
        print(input_ids_list)
        print(target_token_ids)
        return None # Word not found
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[0]
    word_embeddings = last_hidden_state[token_indices]
    average_embedding = torch.mean(word_embeddings, dim=0)
    return average_embedding.cpu().numpy()




def calculate_cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None or vec1.shape != vec2.shape: return None
    vec1 = vec1.astype(np.float64); vec2 = vec2.astype(np.float64)
    norm_vec1 = np.linalg.norm(vec1); norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    try: similarity = 1 - cosine_distance(vec1, vec2); return np.clip(similarity, -1.0, 1.0)
    except ValueError: return 0.0
# --- (End of Pasted Block) ---


# --- <<<< START OF OUTER LOOP >>>> ---
results_by_length = {}
pattern = re.compile(r"\b" + re.escape(TARGET_WORD) + r"\b", flags=re.IGNORECASE)

for CONTEXT_LENGTH_TARGET in CONTEXT_LENGTHS_TO_TEST:

    print("\n" + "="*80)
    print(f"Processing for Context Length: {CONTEXT_LENGTH_TARGET}")
    print("="*80)
    start_time_length = time.time()

    # --- 4. Main Experiment Loop (Using Pre-filtered Data, Simplified Chunking Check) ---
    current_length_results = []
    valid_chunks_for_length = []

    print(f"Creating chunks of length {CONTEXT_LENGTH_TARGET} from pre-filtered data...")
    # Iterate through the pre-filtered examples and chunk them
    for example in filtered_examples:
        text = example.get("text", "")
        if not text:
            continue

        half = CONTEXT_LENGTH_TARGET // 2

        # find each occurrence of "foobar"
        for m in pattern.finditer(text):
            # split into words by whitespace
            before_words = text[: m.start() ].split()
            after_words  = text[m.end() :].split()
            match_word   = m.group()

            # take half from before, half from after
            left  = before_words[-half:]
            right = after_words[:half]

            chunk = " ".join(left + [match_word] + right)
            valid_chunks_for_length.append(chunk)


    # now valid_chunks_for_length holds all slices around each TARGET_WORD
    print(f"Created {len(valid_chunks_for_length)} valid chunks for length {CONTEXT_LENGTH_TARGET}.")

    if len(valid_chunks_for_length) < 2:
        print(f"Not enough valid chunks generated for length {CONTEXT_LENGTH_TARGET}. Skipping length.")
        results_by_length[CONTEXT_LENGTH_TARGET] = {'mean_same_word': None, 'mean_diff_word': None, 'samples': 0}
        continue

    # --- Inner loop for collecting samples (Same as before) ---
    num_collected_samples = 0
    max_attempts = NUM_SAMPLES_PER_LENGTH * 5
    attempts = 0
    run_chunks = list(valid_chunks_for_length) # Use copy

    while num_collected_samples < NUM_SAMPLES_PER_LENGTH and attempts < max_attempts and len(run_chunks) >= 2:
        attempts += 1
        # (Rest of inner loop: sampling chunks, getting other word, calculating embeddings/similarity, storing results - remains the same as previous version)
        # --- (Paste Inner Loop Block Here) ---
        idx1, idx2 = random.sample(range(len(run_chunks)), 2)
        context1_text = run_chunks[idx1]
        context2_text = run_chunks[idx2]
        tokens_context1 = tokenizer.tokenize(context1_text)
        words = re.findall(r"\b[^\W\d_]+\b", context1_text)  
        #    └─ \b          word boundary
        #       [^\W\d_]+  one or more letters (no digits, no underscores, no punctuation)

        # 2) filter out any unwanted items
        potential_other_words = [
            w for w in words
            if w.lower() != TARGET_WORD.lower()   # not the target
            and len(w) > 1                        # skip single‐letter words, if you like
        ]

        # 3) pick one at random
        if not potential_other_words:
            raise ValueError("No candidate words found!")

        # e.g. shuffle and pick one at random
        other_word_in_context1 = random.choice(potential_other_words) if potential_other_words else None        
        embedding_target_c1 = get_word_embedding_in_context(context1_text, TARGET_WORD, tokenizer, model, device)
        embedding_target_c2 = get_word_embedding_in_context(context2_text, TARGET_WORD, tokenizer, model, device)
        embedding_other_c1 = get_word_embedding_in_context(context1_text, other_word_in_context1, tokenizer, model, device)
        sim_same_word_diff_context = calculate_cosine_similarity(embedding_target_c1, embedding_target_c2)
        sim_diff_word_same_context = calculate_cosine_similarity(embedding_target_c1, embedding_other_c1)
        if sim_same_word_diff_context is None or sim_diff_word_same_context is None: run_chunks.pop(max(idx1, idx2)); run_chunks.pop(min(idx1, idx2)); continue
        current_length_results.append({"sim_same_word_diff_context": sim_same_word_diff_context, "sim_diff_word_same_context": sim_diff_word_same_context})
        num_collected_samples += 1
        run_chunks.pop(max(idx1, idx2)); run_chunks.pop(min(idx1, idx2)) # Remove used chunks
        # --- (End of Pasted Block) ---


    # --- 5. Analyze Results for the current context length (Same as before) ---
    print(f"\n--- Results Summary for Context Length: {CONTEXT_LENGTH_TARGET} ---")
    # ... (Keep analysis code) ...
    # --- (Paste Analysis Block Here) ---
    mean_same_word_current = None; mean_diff_word_current = None
    if current_length_results:
        mean_same_word_current = np.mean([r['sim_same_word_diff_context'] for r in current_length_results])
        mean_diff_word_current = np.mean([r['sim_diff_word_same_context'] for r in current_length_results])
        print(f"Collected {num_collected_samples} samples.")
        print(f"Average Similarity (Same Word): {mean_same_word_current:.4f}")
        print(f"Average Similarity (Diff Word): {mean_diff_word_current:.4f}")
    else: 
        print(f"No successful results were collected for context length {CONTEXT_LENGTH_TARGET}.")
    results_by_length[CONTEXT_LENGTH_TARGET] = {'mean_same_word': mean_same_word_current, 'mean_diff_word': mean_diff_word_current, 'samples': num_collected_samples}
    end_time_length = time.time(); print(f"Time taken for length {CONTEXT_LENGTH_TARGET}: {end_time_length - start_time_length:.2f} seconds")
    # --- (End of Pasted Block) ---


# --- <<<< END OF OUTER LOOP >>>> ---


# --- Plotting Section (Sections 7 and 8 - Same as before) ---
print("\n" + "="*80 + "\nGenerating Plots\n" + "="*80)
# ... (Keep plotting code sections 7 and 8 exactly as in the previous response) ...
# --- (Paste Plotting Sections 7 and 8 Here) ---
# --- 7. Generate KDE Plot (Example for the *last* completed context length) ---
figure_dir = "figures"; os.makedirs(figure_dir, exist_ok=True) # Ensure figure dir exists
last_successful_length = None
for length in reversed(CONTEXT_LENGTHS_TO_TEST):
    if results_by_length.get(length) and results_by_length[length]['samples'] > 0: last_successful_length = length; break
if last_successful_length is not None and current_length_results:
    print(f"Generating KDE plot for last successful length: {last_successful_length}...")
    similarities_same_word = [r['sim_same_word_diff_context'] for r in current_length_results]
    similarities_diff_word = [r['sim_diff_word_same_context'] for r in current_length_results]
    mean_same_word = results_by_length[last_successful_length]['mean_same_word']; mean_diff_word = results_by_length[last_successful_length]['mean_diff_word']
    plt.figure(figsize=(2.3 * 2, 1.8 * 2), dpi=200)
    plot_names = {'same_word': f"Same Word, Diff Contexts (Mean: {mean_same_word:.2f})", 'diff_word': f"Diff Word, Same Context (Mean: {mean_diff_word:.2f})"}
    sns.kdeplot(similarities_same_word, label=plot_names['same_word'], color="skyblue", fill=True, alpha=0.5, linewidth=1.5)
    sns.kdeplot(similarities_diff_word, label=plot_names['diff_word'], color="lightcoral", fill=True, alpha=0.5, linewidth=1.5)
    plt.axvline(mean_same_word, color='blue', linestyle='--', linewidth=1); plt.axvline(mean_diff_word, color='red', linestyle='--', linewidth=1)
    plt.legend(fontsize=7, frameon=True); plt.xlabel("Cosine Similarity", fontsize=8); plt.ylabel("Density", fontsize=8)
    plt.title(f"Similarity Distribution for '{TARGET_WORD}'\n({MODEL_NAME}, Ctx Len {last_successful_length})", fontsize=9)
    plt.grid(True, c="0.95")
    figure_filename_kde = f"{figure_dir}/similarity_dist_{TARGET_WORD}_{MODEL_NAME.split('/')[-1]}_ctx{last_successful_length}.pdf"
    try: plt.savefig(figure_filename_kde, format="pdf", bbox_inches="tight"); print(f"KDE Plot saved to {figure_filename_kde}")
    except Exception as e: print(f"Error saving KDE plot: {e}")
    plt.show()
else: print("Skipping KDE plot.")

# --- 8. Generate Line Plot (Mean Similarities vs. Context Length) ---
print("\nGenerating Line plot (Mean Similarity vs Context Length)...")
context_lengths_plotted = []; mean_same_word_values = []; mean_diff_word_values = []
for length in CONTEXT_LENGTHS_TO_TEST:
    if results_by_length.get(length) and results_by_length[length]['samples'] > 0: context_lengths_plotted.append(length); mean_same_word_values.append(results_by_length[length]['mean_same_word']); mean_diff_word_values.append(results_by_length[length]['mean_diff_word'])
if not context_lengths_plotted: print("No data available to generate the line plot.")
else:
    plt.figure(figsize=(5, 3.5), dpi=150)
    plt.plot(context_lengths_plotted, mean_same_word_values, label='Mean Sim (Same Word, Diff Contexts)', marker='o', linestyle='-', color='blue')
    plt.plot(context_lengths_plotted, mean_diff_word_values, label='Mean Sim (Diff Word, Same Context)', marker='s', linestyle='--', color='red')
    plt.legend(fontsize=9, frameon=True); plt.xlabel("Context Length (Tokens)", fontsize=10); plt.ylabel("Mean Cosine Similarity", fontsize=10)
    plt.title(f"Mean Similarity vs. Context Length for '{TARGET_WORD}'\n({MODEL_NAME})", fontsize=11)
    plt.xticks(context_lengths_plotted); plt.grid(True, c="0.95", linestyle=':')
    figure_filename_line = f"{figure_dir}/mean_similarity_vs_length_{TARGET_WORD}_{MODEL_NAME.split('/')[-1]}.pdf"
    try: plt.savefig(figure_filename_line, format="pdf", bbox_inches="tight"); print(f"Line Plot saved to {figure_filename_line}")
    except Exception as e: print(f"Error saving Line plot: {e}")
    plt.show()
# --- (End of Pasted Block) ---

print("\nScript finished.")