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
from datasets import load_dataset
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time # For timing

# --- Configuration ---
# --- Choose your model ---
MODEL_NAME = "bert-base-uncased"
# Examples:
MODEL_NAME = "openai-community/gpt2-large" # Max length ~1024
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B" # Max length ~8192 (Requires login/permission)

# --- Choose your dataset ---
DATASET_NAME = "openwebtext"
DATASET_SPLIT = "train"

# --- Experiment Parameters ---
TARGET_WORD = "sorry"
# --- Define Context Lengths to Test ---
CONTEXT_LENGTHS_TO_TEST = [100, 200, 300, 500, 1000, 2000] # Example list
# Ensure these are <= MODEL_MAX_LENGTH

NUM_SAMPLES_PER_LENGTH = 50      # How many successful comparisons per context length
MIN_CHUNK_FACTOR = 3             # Fetch factor for needed chunks (e.g., 3x samples)
MAX_DOCS_TO_PROCESS = 10000      # Safety limit for pre-fetching documents

# --- 1. Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
# --- Add Hugging Face Hub login check/prompt if using gated models ---
if "llama" in MODEL_NAME.lower():
     try:
         from huggingface_hub import HfFolder
         if HfFolder.get_token() is None:
             print("="*80)
             print(" Llama model detected. Please login to Hugging Face Hub.")
             print(" Run: huggingface-cli login")
             print("="*80)
             # Optionally exit if login is required: exit()
     except ImportError:
         print("Warning: huggingface_hub not installed. Login might be required for Llama models.")
# --- End login check ---

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # --- Handle potential padding token issues ---
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure model config reflects this if needed, although AutoModel usually handles it
    # --- End padding token handling ---

    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir = PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on device: {device}")

    MODEL_MAX_LENGTH = tokenizer.model_max_length
    # Handle models without explicit max length set (rare, fallback)
    if MODEL_MAX_LENGTH is None or MODEL_MAX_LENGTH > 100000:
        MODEL_MAX_LENGTH = 1024 # Default fallback
        print(f"Warning: Model max length not found or very large, falling back to {MODEL_MAX_LENGTH}.")
    print(f"Model max sequence length: {MODEL_MAX_LENGTH}")

    # --- Validate chosen context lengths ---
    valid_context_lengths = []
    for length in CONTEXT_LENGTHS_TO_TEST:
        if length > MODEL_MAX_LENGTH:
            print(f"Warning: CONTEXT_LENGTH_TARGET {length} exceeds model max length {MODEL_MAX_LENGTH}. Skipping this length.")
        else:
            valid_context_lengths.append(length)
    CONTEXT_LENGTHS_TO_TEST = valid_context_lengths
    if not CONTEXT_LENGTHS_TO_TEST:
        print("Error: No valid context lengths to test based on model max length.")
        exit()
    print(f"Will test context lengths: {CONTEXT_LENGTHS_TO_TEST}")

except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    print("Ensure the model name is correct, you have internet access,")
    print("and you are logged in via 'huggingface-cli login' if required.")
    exit()

# --- 2. Helper Functions (Same as before) ---

def get_word_embedding_in_context(text, target_word, tokenizer, model, device):
    """Gets the embedding for a target word within a given text context."""
    try:
        # Use padding and truncation, return attention mask explicitly
        encoded_input = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding="max_length", # Pad to max_length for consistent tensor shapes if needed by model
            max_length=MODEL_MAX_LENGTH,
            return_attention_mask=True # Ensure attention mask is returned
        )
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        target_token_ids = tokenizer.encode(target_word, add_special_tokens=False)
        token_indices = []
        input_ids_list = input_ids[0].tolist()

        # Adjust search range based on actual non-padding tokens if using padding
        try:
             actual_length = attention_mask[0].nonzero().max().item() + 1 # Find last non-padded token index
        except:
             actual_length = len(input_ids_list) # Fallback if no non-zero elements (empty input?)

        for i in range(min(actual_length, len(input_ids_list)) - len(target_token_ids) + 1): # Search only within actual tokens
            if input_ids_list[i:i+len(target_token_ids)] == target_token_ids:
                token_indices.extend(list(range(i, i+len(target_token_ids))))
                break

        if not token_indices:
            return None

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state[0]

        word_embeddings = last_hidden_state[token_indices]
        average_embedding = torch.mean(word_embeddings, dim=0)
        return average_embedding.cpu().numpy()

    except Exception as e:
        print(f"Error in get_word_embedding_in_context for word '{target_word}': {e}")
        return None


def calculate_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two numpy vectors."""
    # (Same as before)
    if vec1 is None or vec2 is None or vec1.shape != vec2.shape:
        return None
    vec1 = vec1.astype(np.float64)
    vec2 = vec2.astype(np.float64)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    try:
        similarity = 1 - cosine_distance(vec1, vec2)
        return np.clip(similarity, -1.0, 1.0)
    except ValueError:
        return 0.0


def chunk_text_and_filter(full_text, target_word, tokenizer, chunk_token_length):
    """
    Splits text into non-overlapping chunks of specified token length
    and returns only chunks containing the target word.
    """
    # (Same as before)
    valid_chunks = []
    if not full_text: return valid_chunks
    try:
        all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        num_tokens = len(all_tokens)
        num_chunks = num_tokens // chunk_token_length
        target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
        if not target_tokens: return valid_chunks # Skip if target word tokenizes to nothing

        for i in range(num_chunks):
            start_idx = i * chunk_token_length
            end_idx = start_idx + chunk_token_length
            chunk_token_ids = all_tokens[start_idx:end_idx]
            found_in_chunk_tokens = False
            for k in range(len(chunk_token_ids) - len(target_tokens) + 1):
                 if chunk_token_ids[k:k+len(target_tokens)] == target_tokens:
                     found_in_chunk_tokens = True
                     break
            if found_in_chunk_tokens:
                chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                if target_word in chunk_text: # Final check on decoded text
                     valid_chunks.append(chunk_text)
    except Exception as e:
        # Reduce verbosity: print(f"Error during chunking/filtering: {e}")
        pass
    return valid_chunks


# --- 3. Load Dataset Stream ---
print(f"\nLoading dataset stream: {DATASET_NAME}...")
try:
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True, trust_remote_code=True, cache_dir = PATH) # Added trust_remote_code
    dataset_iterator = iter(dataset)
    print("Dataset stream ready.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- <<<< START OF OUTER LOOP >>>> ---
# --- Dictionary to store results for each context length ---
results_by_length = {}

for CONTEXT_LENGTH_TARGET in CONTEXT_LENGTHS_TO_TEST:

    print("\n" + "="*80)
    print(f"Processing for Context Length: {CONTEXT_LENGTH_TARGET}")
    print("="*80)

    start_time_length = time.time()

    # --- 4. Main Experiment Loop (Chunk-First Strategy for current length) ---
    current_length_results = [] # Results for this specific length
    available_target_chunks = []
    processed_doc_ids = set() # Reset for each length to allow reusing documents if needed

    min_chunks_for_current_length = NUM_SAMPLES_PER_LENGTH * MIN_CHUNK_FACTOR
    print(f"Pre-fetching at least {min_chunks_for_current_length} chunks containing '{TARGET_WORD}' for length {CONTEXT_LENGTH_TARGET}...")
    docs_processed = 0
    # Re-use the dataset iterator if possible, otherwise re-initialize if needed
    # Note: If the iterator gets exhausted, subsequent lengths might fail.
    # Consider re-initializing the iterator for each length if dataset is small or pre-fetching is exhaustive.
    # try:
    #     dataset_iterator = iter(load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True, trust_remote_code=True))
    # except Exception as e:
    #     print(f"Could not reset dataset iterator: {e}")
    #     break # Stop processing further lengths

    while len(available_target_chunks) < min_chunks_for_current_length and docs_processed < MAX_DOCS_TO_PROCESS:
        try:
            example = next(dataset_iterator)
            docs_processed += 1
            doc_id = example.get('id', hash(example.get('text', '')) ) # Simple ID

            # Check processed_doc_ids if you strictly want chunks from different docs per length run
            # if doc_id not in processed_doc_ids:
            full_text = example.get('text', '')
            if full_text:
                chunks = chunk_text_and_filter(full_text, TARGET_WORD, tokenizer, CONTEXT_LENGTH_TARGET)
                if chunks:
                    available_target_chunks.extend(chunks)
                    # processed_doc_ids.add(doc_id) # Mark doc as processed for this length

        except StopIteration:
            print("Dataset depleted during pre-fetching.")
            break # Stop pre-fetching for this length
        except Exception as e:
             print(f"Error during pre-fetching: {e}")
             # Optionally break here

    print(f"Finished pre-fetching for length {CONTEXT_LENGTH_TARGET}. Found {len(available_target_chunks)} valid chunks.")

    if len(available_target_chunks) < 2:
        print(f"Not enough chunks found for length {CONTEXT_LENGTH_TARGET} to proceed with comparisons.")
        results_by_length[CONTEXT_LENGTH_TARGET] = {'mean_same_word': None, 'mean_diff_word': None, 'samples': 0}
        continue # Move to the next context length

    # --- Inner loop for collecting samples for the current length ---
    num_collected_samples = 0
    max_attempts = NUM_SAMPLES_PER_LENGTH * 5
    attempts = 0

    while num_collected_samples < NUM_SAMPLES_PER_LENGTH and attempts < max_attempts and len(available_target_chunks) >= 2:
        attempts += 1
        # print(f"\n--- Length {CONTEXT_LENGTH_TARGET}, Attempt {attempts} (Collecting Sample {num_collected_samples + 1}/{NUM_SAMPLES_PER_LENGTH}) ---") # Verbose

        idx1, idx2 = random.sample(range(len(available_target_chunks)), 2)
        context1_text = available_target_chunks[idx1]
        context2_text = available_target_chunks[idx2]

        tokens_context1 = tokenizer.tokenize(context1_text)
        potential_other_words = [t for t in tokens_context1 if t not in tokenizer.all_special_tokens and t != TARGET_WORD and len(t)>1 and '##' not in t]
        if not potential_other_words:
            # print("Could not find suitable other word. Skipping attempt.") # Verbose
            available_target_chunks.pop(max(idx1, idx2))
            available_target_chunks.pop(min(idx1, idx2))
            continue
        other_word_in_context1 = random.choice(potential_other_words)

        embedding_target_c1 = get_word_embedding_in_context(context1_text, TARGET_WORD, tokenizer, model, device)
        embedding_target_c2 = get_word_embedding_in_context(context2_text, TARGET_WORD, tokenizer, model, device)
        embedding_other_c1 = get_word_embedding_in_context(context1_text, other_word_in_context1, tokenizer, model, device)

        sim_same_word_diff_context = calculate_cosine_similarity(embedding_target_c1, embedding_target_c2)
        sim_diff_word_same_context = calculate_cosine_similarity(embedding_target_c1, embedding_other_c1)

        if sim_same_word_diff_context is None or sim_diff_word_same_context is None:
             # print("Failed to calculate similarities. Skipping attempt.") # Verbose
             available_target_chunks.pop(max(idx1, idx2))
             available_target_chunks.pop(min(idx1, idx2))
             continue

        # Store results for this specific sample
        current_length_results.append({
            "sim_same_word_diff_context": sim_same_word_diff_context,
            "sim_diff_word_same_context": sim_diff_word_same_context
        })
        num_collected_samples += 1

        # Optional: Remove used chunks
        available_target_chunks.pop(max(idx1, idx2))
        available_target_chunks.pop(min(idx1, idx2))

    # --- 5. Analyze Results for the current context length ---
    print(f"\n--- Results Summary for Context Length: {CONTEXT_LENGTH_TARGET} ---")
    mean_same_word_current = None
    mean_diff_word_current = None
    if current_length_results:
        mean_same_word_current = np.mean([r['sim_same_word_diff_context'] for r in current_length_results])
        mean_diff_word_current = np.mean([r['sim_diff_word_same_context'] for r in current_length_results])
        print(f"Collected {num_collected_samples} samples.")
        print(f"Average Similarity (Same Word, Different Context): {mean_same_word_current:.4f}")
        print(f"Average Similarity (Different Word, Same Context): {mean_diff_word_current:.4f}")
    else:
        print(f"No successful results were collected for context length {CONTEXT_LENGTH_TARGET}.")

    # Store the means for the final line plot
    results_by_length[CONTEXT_LENGTH_TARGET] = {
        'mean_same_word': mean_same_word_current,
        'mean_diff_word': mean_diff_word_current,
        'samples': num_collected_samples
        }

    end_time_length = time.time()
    print(f"Time taken for length {CONTEXT_LENGTH_TARGET}: {end_time_length - start_time_length:.2f} seconds")

# --- <<<< END OF OUTER LOOP >>>> ---


# --- Plotting Section ---
print("\n" + "="*80)
print("Generating Plots")
print("="*80)

figure_dir = "figures"
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# --- 7. Generate KDE Plot (Example for the *last* completed context length) ---
last_successful_length = None
for length in reversed(CONTEXT_LENGTHS_TO_TEST):
    if results_by_length[length]['samples'] > 0:
        last_successful_length = length
        break

if last_successful_length is not None and current_length_results: # Check if last run produced results
    print(f"Generating KDE plot for last successful length: {last_successful_length}...")
    similarities_same_word = [r['sim_same_word_diff_context'] for r in current_length_results] # Use results from last loop
    similarities_diff_word = [r['sim_diff_word_same_context'] for r in current_length_results] # Use results from last loop
    mean_same_word = results_by_length[last_successful_length]['mean_same_word']
    mean_diff_word = results_by_length[last_successful_length]['mean_diff_word']

    plt.figure(figsize=(2.3 * 2, 1.8 * 2), dpi=200)
    plot_names = {
        'same_word': f"Same Word, Diff Contexts (Mean: {mean_same_word:.2f})",
        'diff_word': f"Diff Word, Same Context (Mean: {mean_diff_word:.2f})"
    }
    sns.kdeplot(similarities_same_word, label=plot_names['same_word'], color="skyblue", fill=True, alpha=0.5, linewidth=1.5)
    sns.kdeplot(similarities_diff_word, label=plot_names['diff_word'], color="lightcoral", fill=True, alpha=0.5, linewidth=1.5)
    plt.axvline(mean_same_word, color='blue', linestyle='--', linewidth=1)
    plt.axvline(mean_diff_word, color='red', linestyle='--', linewidth=1)
    plt.legend(fontsize=7, frameon=True)
    plt.xlabel("Cosine Similarity", fontsize=8)
    plt.ylabel("Density", fontsize=8)
    plt.title(f"Similarity Distribution for '{TARGET_WORD}'\n({MODEL_NAME}, Ctx Len {last_successful_length})", fontsize=9)
    plt.grid(True, c="0.95")
    figure_filename_kde = f"{figure_dir}/similarity_dist_{TARGET_WORD}_{MODEL_NAME.split('/')[-1]}_ctx{last_successful_length}.pdf"
    try:
        plt.savefig(figure_filename_kde, format="pdf", bbox_inches="tight")
        print(f"KDE Plot saved to {figure_filename_kde}")
    except Exception as e:
        print(f"Error saving KDE plot: {e}")
    plt.show()
else:
    print("Skipping KDE plot as no results were available for the last context length.")


# --- 8. Generate Line Plot (Mean Similarities vs. Context Length) ---
print("\nGenerating Line plot (Mean Similarity vs Context Length)...")

context_lengths_plotted = []
mean_same_word_values = []
mean_diff_word_values = []

for length in CONTEXT_LENGTHS_TO_TEST:
    if results_by_length[length]['samples'] > 0: # Only plot lengths with successful samples
        context_lengths_plotted.append(length)
        mean_same_word_values.append(results_by_length[length]['mean_same_word'])
        mean_diff_word_values.append(results_by_length[length]['mean_diff_word'])

if not context_lengths_plotted:
    print("No data available to generate the line plot.")
else:
    plt.figure(figsize=(5, 3.5), dpi=150) # Adjust size for line plot

    # Plot Mean Same Word vs Context Length
    plt.plot(context_lengths_plotted, mean_same_word_values,
             label='Mean Sim (Same Word, Diff Contexts)',
             marker='o', # Add markers
             linestyle='-', # Solid line
             color='blue')

    # Plot Mean Diff Word vs Context Length
    plt.plot(context_lengths_plotted, mean_diff_word_values,
             label='Mean Sim (Diff Word, Same Context)',
             marker='s', # Add different markers (square)
             linestyle='--', # Dashed line
             color='red')

    plt.legend(fontsize=9, frameon=True)
    plt.xlabel("Context Length (Tokens)", fontsize=10)
    plt.ylabel("Mean Cosine Similarity", fontsize=10)
    plt.title(f"Mean Similarity vs. Context Length for '{TARGET_WORD}'\n({MODEL_NAME})", fontsize=11)
    plt.xticks(context_lengths_plotted) # Ensure ticks appear at tested lengths
    plt.grid(True, c="0.95", linestyle=':') # Lighter grid

    figure_filename_line = f"{figure_dir}/mean_similarity_vs_length_{TARGET_WORD}_{MODEL_NAME.split('/')[-1]}.pdf"
    try:
        plt.savefig(figure_filename_line, format="pdf", bbox_inches="tight")
        print(f"Line Plot saved to {figure_filename_line}")
    except Exception as e:
        print(f"Error saving Line plot: {e}")

    plt.show()


print("\nScript finished.")