from transformers import GPT2Tokenizer

# 1) load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 2) example text, ending in two newlines + "sure"
text = (
    "Dead, the group's central hero had his spirit crushed during Sunday's season "
    "seven premiere by the zombie drama's most famous villain Negan. As if losing "
    "two huge members of the group — Steven Yeun's Glenn Rhee and Michael Cudlitz's "
    "Sgt. Abraham Ford — wasn't enough, Negan (Jeffrey Dean Morgan) made sure to "
    "emotionally torment Rick (Andrew Lincoln) into obedience, with the group's strong "
    "leader falling in line after nearly having to cut off his son Carl's (Chandler Riggs) "
    "arm. Below, Lincoln talks with THR about filming the jaw-dropping episode, losing "
    "Yeun and Cudlitz from the core cast and what's next in\n\n"
    "sure"
)

print("FULL TEXT:", repr(text))
print()

# tokenize the full text
input_ids = tokenizer.encode(text)
print("INPUT_IDS ({} tokens):\n{}".format(len(input_ids), input_ids))
print()

# prepare two possible target encodings:
targets = {
    "'sure' (no leading space)": tokenizer.encode("sure"),
    "' sure' (with leading space)": tokenizer.encode(" sure"),
    "'sure ' (with trail space)": tokenizer.encode("sure "),
}

# try to find each in input_ids
for desc, target_ids in targets.items():
    print(f"Trying to find {desc} -> {target_ids}")
    found = False
    for i in range(len(input_ids) - len(target_ids) + 1):
        if input_ids[i : i + len(target_ids)] == target_ids:
            print(f"  → FOUND at position {i}")
            found = True
            break
    if not found:
        print("  → NOT found")
    print()

# as a sanity check, show the tokenizer’s text for each target_ids
for desc, target_ids in targets.items():
    print(f"{desc}  decodes to ->", tokenizer.decode(target_ids))