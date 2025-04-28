import os
PATH = '/home/chenboc1/localscratch2/chenboc1/Meaning_loss/cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
# --- Imports ---
import torch
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

# --- Configuration ---
# --- Choose your model ---
MODEL_NAME = "bert-base-uncased"
# Examples:
MODEL_NAME = "openai-community/gpt2-large"
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
PREFILTERED_DATA_DIR = "./prefiltered_data" # Directory to save/load filtered data
FORCE_REFILTER = False # Set to True to re-run filtering even if file exists

# --- Create directory for filtered data ---
os.makedirs(PREFILTERED_DATA_DIR, exist_ok=True)
# Define filename based on dataset and target word
filtered_data_file = os.path.join(PREFILTERED_DATA_DIR, f"{DATASET_NAME.replace('/','_')}_{TARGET_WORD}_{NUM_PREFILTERED_EXAMPLES}.jsonl")

# --- <<<< NEW SECTION: Preprocessing - Filter Dataset >>>> ---
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
        filtered_examples = [] # Reset if loading failed

if not filtered_examples:
    print("-" * 50)
    print(f"Preprocessing: Filtering dataset '{DATASET_NAME}' to find {NUM_PREFILTERED_EXAMPLES} examples containing '{TARGET_WORD}'...")
    print("-" * 50)
    start_filter_time = time.time()
    try:
        # Load the streaming dataset
        raw_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True, trust_remote_code=True)
        # Use a temporary tokenizer just for filtering if model isn't loaded yet
        # Note: Ideally use the *same* tokenizer as the main experiment
        try:
             filter_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception as e:
             print(f"Warning: Could not load tokenizer {MODEL_NAME} for filtering. Using basic split(). This might be less accurate.")
             filter_tokenizer = None

        count = 0
        processed_count = 0
        # Iterate through the stream and collect examples
        for example in raw_dataset:
            processed_count += 1
            text = example.get('text', '')
            if text:
                 # Use string search for broad filtering first (faster)
                 if TARGET_WORD in text:
                     # Optional: More precise check using tokenizer if available
                     word_found_robustly = True
                     if filter_tokenizer:
                          try:
                              tokens = filter_tokenizer.encode(TARGET_WORD, add_special_tokens=False)
                              text_tokens = filter_tokenizer.encode(text, add_special_tokens=False, max_length=8192, truncation=True) # Limit length for performance
                              found_tokens = False
                              for i in range(len(text_tokens) - len(tokens) + 1):
                                   if text_tokens[i:i+len(tokens)] == tokens:
                                       found_tokens = True
                                       break
                              if not found_tokens:
                                   word_found_robustly = False
                          except Exception as e:
                               # Handle potential tokenization errors on weird text
                               # print(f"Tokenization error during filtering check: {e}") # Debug
                               pass # Assume string check is sufficient if token check fails


                     if word_found_robustly:
                        filtered_examples.append({'text': text}) # Store only the text
                        count += 1
                        if count % 100 == 0:
                            print(f"  Found {count}/{NUM_PREFILTERED_EXAMPLES} examples... (Processed {processed_count})")
                        if count >= NUM_PREFILTERED_EXAMPLES:
                            print(f"Reached target of {NUM_PREFILTERED_EXAMPLES} filtered examples.")
                            break # Stop after finding enough examples

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
             print("No examples containing the target word were found.")
             exit() # Exit if no relevant data found

    except Exception as e:
        print(f"Error during dataset filtering: {e}")
        exit() # Exit if filtering fails

# --- Load the preprocessed data into a datasets.Dataset object for easier handling ---
if filtered_examples:
     try:
         # Convert list of dicts to Dataset object (requires pandas potentially)
         # Simpler: just use the list directly in the loop below.
         print(f"Using {len(filtered_examples)} pre-filtered examples.")
     except Exception as e:
          print(f"Could not load filtered data into Dataset object: {e}")
          # Fallback: exit or try using the list directly if possible
          exit()
else:
     print("No pre-filtered examples available. Exiting.")
     exit()
# --- <<<< END OF PREPROCESSING SECTION >>>> ---


# --- 1. Load Model and Tokenizer (Same as before, placed after preprocessing) ---
print(f"\nLoading model: {MODEL_NAME}...")
# ... (Keep the model loading code from the previous version here, including login checks, padding token handling, max length checks) ...
# ... (Make sure MODEL_MAX_LENGTH is defined here) ...
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
    if MODEL_MAX_LENGTH is None or MODEL_MAX_LENGTH > 100000: MODEL_MAX_LENGTH = 1024; print(f"Warning: Model max length not found, falling back to {MODEL_MAX_LENGTH}.")
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


# --- 2. Helper Functions (Same as before) ---
# ... (Keep get_word_embedding_in_context and calculate_cosine_similarity) ...
# --- (Paste Helper Functions Here) ---
def get_word_embedding_in_context(text, target_word, tokenizer, model, device):
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding="max_length", max_length=MODEL_MAX_LENGTH, return_attention_mask=True)
        input_ids = encoded_input['input_ids'].to(device); attention_mask = encoded_input['attention_mask'].to(device)
        target_token_ids = tokenizer.encode(target_word, add_special_tokens=False)
        token_indices = []; input_ids_list = input_ids[0].tolist()
        try: actual_length = attention_mask[0].nonzero().max().item() + 1
        except: actual_length = len(input_ids_list)
        for i in range(min(actual_length, len(input_ids_list)) - len(target_token_ids) + 1):
            if input_ids_list[i:i+len(target_token_ids)] == target_token_ids: token_indices.extend(list(range(i, i+len(target_token_ids)))); break
        if not token_indices: return None
        with torch.no_grad(): outputs = model(input_ids=input_ids, attention_mask=attention_mask); last_hidden_state = outputs.last_hidden_state[0]
        word_embeddings = last_hidden_state[token_indices]; average_embedding = torch.mean(word_embeddings, dim=0)
        return average_embedding.cpu().numpy()
    except Exception as e: print(f"Error in get_word_embedding_in_context for word '{target_word}': {e}"); return None

def calculate_cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None or vec1.shape != vec2.shape: return None
    vec1 = vec1.astype(np.float64); vec2 = vec2.astype(np.float64)
    norm_vec1 = np.linalg.norm(vec1); norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    try: similarity = 1 - cosine_distance(vec1, vec2); return np.clip(similarity, -1.0, 1.0)
    except ValueError: return 0.0
# --- (End of Pasted Block) ---


# --- <<<< START OF OUTER LOOP >>>> ---
results_by_length = {} # Store final mean results per length

for CONTEXT_LENGTH_TARGET in CONTEXT_LENGTHS_TO_TEST:

    print("\n" + "="*80)
    print(f"Processing for Context Length: {CONTEXT_LENGTH_TARGET}")
    print("="*80)
    start_time_length = time.time()

    # --- 4. Main Experiment Loop (Using Pre-filtered Data) ---
    current_length_results = [] # Store individual sample results for this length
    valid_chunks_for_length = [] # Store chunks of the *correct length* from the filtered data

    print(f"Creating chunks of length {CONTEXT_LENGTH_TARGET} from pre-filtered data...")
    # Iterate through the pre-filtered examples and chunk them
    for example in filtered_examples:
        full_text = example['text']
        # Chunk the text (this logic is similar to the old helper, but applied here)
        try:
            all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            num_tokens = len(all_tokens)
            num_chunks = num_tokens // CONTEXT_LENGTH_TARGET
            target_tokens = tokenizer.encode(TARGET_WORD, add_special_tokens=False)
            if not target_tokens: continue

            for i in range(num_chunks):
                start_idx = i * CONTEXT_LENGTH_TARGET
                end_idx = start_idx + CONTEXT_LENGTH_TARGET
                chunk_token_ids = all_tokens[start_idx:end_idx]
                # Verify target word tokens are still in this specific chunk
                found_in_chunk_tokens = False
                for k in range(len(chunk_token_ids) - len(target_tokens) + 1):
                     if chunk_token_ids[k:k+len(target_tokens)] == target_tokens:
                         found_in_chunk_tokens = True
                         break
                if found_in_chunk_tokens:
                    chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                    # Final check just in case decoding messes things up
                    if TARGET_WORD in chunk_text:
                         valid_chunks_for_length.append(chunk_text)

        except Exception as e:
            # print(f"Error chunking prefiltered text: {e}") # Debug
            pass # Skip problematic text

    print(f"Created {len(valid_chunks_for_length)} valid chunks for length {CONTEXT_LENGTH_TARGET}.")

    if len(valid_chunks_for_length) < 2:
        print(f"Not enough valid chunks generated for length {CONTEXT_LENGTH_TARGET}. Skipping length.")
        results_by_length[CONTEXT_LENGTH_TARGET] = {'mean_same_word': None, 'mean_diff_word': None, 'samples': 0}
        continue

    # --- Inner loop for collecting samples for the current length ---
    num_collected_samples = 0
    max_attempts = NUM_SAMPLES_PER_LENGTH * 5 # Allow for failures in embedding/similarity steps
    attempts = 0
    # Use a copy to allow removal during sampling without affecting next length
    run_chunks = list(valid_chunks_for_length)

    while num_collected_samples < NUM_SAMPLES_PER_LENGTH and attempts < max_attempts and len(run_chunks) >= 2:
        attempts += 1
        # print(f"Attempt {attempts}...") # Verbose

        idx1, idx2 = random.sample(range(len(run_chunks)), 2)
        context1_text = run_chunks[idx1]
        context2_text = run_chunks[idx2]

        tokens_context1 = tokenizer.tokenize(context1_text)
        potential_other_words = [t for t in tokens_context1 if t not in tokenizer.all_special_tokens and t != TARGET_WORD and len(t)>1 and '##' not in t]
        if not potential_other_words:
            # print("No other word found.") # Verbose
            run_chunks.pop(max(idx1, idx2)); run_chunks.pop(min(idx1, idx2)) # Remove pair
            continue
        other_word_in_context1 = random.choice(potential_other_words)

        embedding_target_c1 = get_word_embedding_in_context(context1_text, TARGET_WORD, tokenizer, model, device)
        embedding_target_c2 = get_word_embedding_in_context(context2_text, TARGET_WORD, tokenizer, model, device)
        embedding_other_c1 = get_word_embedding_in_context(context1_text, other_word_in_context1, tokenizer, model, device)

        sim_same_word_diff_context = calculate_cosine_similarity(embedding_target_c1, embedding_target_c2)
        sim_diff_word_same_context = calculate_cosine_similarity(embedding_target_c1, embedding_other_c1)

        if sim_same_word_diff_context is None or sim_diff_word_same_context is None:
             # print("Failed similarity calculation.") # Verbose
             run_chunks.pop(max(idx1, idx2)); run_chunks.pop(min(idx1, idx2)) # Remove pair
             continue

        current_length_results.append({
            "sim_same_word_diff_context": sim_same_word_diff_context,
            "sim_diff_word_same_context": sim_diff_word_same_context
        })
        num_collected_samples += 1

        # Remove used chunks to prevent reuse in subsequent samples *for this length*
        run_chunks.pop(max(idx1, idx2))
        run_chunks.pop(min(idx1, idx2))

    # --- 5. Analyze Results for the current context length ---
    print(f"\n--- Results Summary for Context Length: {CONTEXT_LENGTH_TARGET} ---")
    # (Same analysis code as before)
    mean_same_word_current = None; mean_diff_word_current = None
    if current_length_results:
        mean_same_word_current = np.mean([r['sim_same_word_diff_context'] for r in current_length_results])
        mean_diff_word_current = np.mean([r['sim_diff_word_same_context'] for r in current_length_results])
        print(f"Collected {num_collected_samples} samples.")
        print(f"Average Similarity (Same Word): {mean_same_word_current:.4f}")
        print(f"Average Similarity (Diff Word): {mean_diff_word_current:.4f}")
    else: print(f"No successful results were collected for context length {CONTEXT_LENGTH_TARGET}.")
    results_by_length[CONTEXT_LENGTH_TARGET] = {'mean_same_word': mean_same_word_current, 'mean_diff_word': mean_diff_word_current, 'samples': num_collected_samples}
    end_time_length = time.time(); print(f"Time taken for length {CONTEXT_LENGTH_TARGET}: {end_time_length - start_time_length:.2f} seconds")
    # --- (End of Pasted Block) ---

# --- <<<< END OF OUTER LOOP >>>> ---


# --- Plotting Section ---
print("\n" + "="*80 + "\nGenerating Plots\n" + "="*80)
# --- (Keep Plotting Sections 7 and 8 exactly as in the previous response) ---
# --- (Paste Plotting Sections 7 and 8 Here) ---
# --- 7. Generate KDE Plot (Example for the *last* completed context length) ---
# (Code from previous response to generate KDE plot for last successful length)
last_successful_length = None
for length in reversed(CONTEXT_LENGTHS_TO_TEST):
    if results_by_length.get(length) and results_by_length[length]['samples'] > 0:
        last_successful_length = length
        break
if last_successful_length is not None and current_length_results: # Check if last run produced results
    print(f"Generating KDE plot for last successful length: {last_successful_length}...")
    # Filter results for the specific length if needed, assuming current_length_results holds the last one
    # If loop structure guarantees current_length_results is from the last successful run, use it directly.
    # Otherwise, you might need to store all samples per length and retrieve them here.
    # Assuming current_length_results holds the necessary data from the last successful iteration:
    similarities_same_word = [r['sim_same_word_diff_context'] for r in current_length_results]
    similarities_diff_word = [r['sim_diff_word_same_context'] for r in current_length_results]
    mean_same_word = results_by_length[last_successful_length]['mean_same_word']
    mean_diff_word = results_by_length[last_successful_length]['mean_diff_word']
    plt.figure(figsize=(2.3 * 2, 1.8 * 2), dpi=200)
    plot_names = {'same_word': f"Same Word, Diff Contexts (Mean: {mean_same_word:.2f})", 'diff_word': f"Diff Word, Same Context (Mean: {mean_diff_word:.2f})"}
    sns.kdeplot(similarities_same_word, label=plot_names['same_word'], color="skyblue", fill=True, alpha=0.5, linewidth=1.5)
    sns.kdeplot(similarities_diff_word, label=plot_names['diff_word'], color="lightcoral", fill=True, alpha=0.5, linewidth=1.5)
    plt.axvline(mean_same_word, color='blue', linestyle='--', linewidth=1); plt.axvline(mean_diff_word, color='red', linestyle='--', linewidth=1)
    plt.legend(fontsize=7, frameon=True); plt.xlabel("Cosine Similarity", fontsize=8); plt.ylabel("Density", fontsize=8)
    plt.title(f"Similarity Distribution for '{TARGET_WORD}'\n({MODEL_NAME}, Ctx Len {last_successful_length})", fontsize=9)
    plt.grid(True, c="0.95")
    # Apply saving logic from your example
    figure_dir = "figures"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)    
    figure_filename_kde = f"{figure_dir}/similarity_dist_{TARGET_WORD}_{MODEL_NAME.split('/')[-1]}_ctx{last_successful_length}.pdf"
    try: plt.savefig(figure_filename_kde, format="pdf", bbox_inches="tight"); print(f"KDE Plot saved to {figure_filename_kde}")
    except Exception as e: print(f"Error saving KDE plot: {e}")
    plt.show()
else: print("Skipping KDE plot.")

# --- 8. Generate Line Plot (Mean Similarities vs. Context Length) ---
# (Code from previous response to generate line plot)
print("\nGenerating Line plot (Mean Similarity vs Context Length)...")
context_lengths_plotted = []; mean_same_word_values = []; mean_diff_word_values = []
for length in CONTEXT_LENGTHS_TO_TEST:
    if results_by_length.get(length) and results_by_length[length]['samples'] > 0:
        context_lengths_plotted.append(length); mean_same_word_values.append(results_by_length[length]['mean_same_word']); mean_diff_word_values.append(results_by_length[length]['mean_diff_word'])
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