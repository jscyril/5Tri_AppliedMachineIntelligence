# Lab 04 — Character-Level Language Model with RNN and LSTM

A detailed walkthrough of `lab04.ipynb`: what every section does, why it is done that way, and how to read the results.

---

## Contents

1. [The assignment in one paragraph](#1-the-assignment-in-one-paragraph)
2. [Files and how to run](#2-files-and-how-to-run)
3. [Background: what is a character-level language model?](#3-background-what-is-a-character-level-language-model)
4. [Section-by-section walkthrough](#4-section-by-section-walkthrough)
   - [§0 Imports and reproducibility](#0-imports-and-reproducibility)
   - [§1 Load text and build mappings (Task 1)](#1-load-text-and-build-mappings-task-1)
   - [§2 Vectorize and make 40→41 sequences (Task 2)](#2-vectorize-and-make-4041-sequences-task-2)
   - [§3 Training sequences / train-validation split (Task 3)](#3-training-sequences--train-validation-split-task-3)
   - [§4 Build the RNN model (Task 4)](#4-build-the-rnn-model-task-4)
   - [§5 Train the model (Task 5)](#5-train-the-model-task-5)
   - [§6 Text generation function (Task 6)](#6-text-generation-function-task-6)
   - [§7 Generate and validate (Task 7)](#7-generate-and-validate-task-7)
   - [§8 Extra: n-gram baselines](#8-extra-n-gram-baselines)
   - [§9 Extra: LSTM model and comparison](#9-extra-lstm-model-and-comparison)
   - [§10 Extra: temperature sweep](#10-extra-temperature-sweep)
   - [§11 Extra: top-5 next-character probabilities](#11-extra-top-5-next-character-probabilities)
   - [§12 Extra: save and reload](#12-extra-save-and-reload)
5. [Results summary](#5-results-summary)
6. [Limitations and how to improve](#6-limitations-and-how-to-improve)
7. [Glossary](#7-glossary)
8. [Likely viva questions](#8-likely-viva-questions)

---

## 1. The assignment in one paragraph

Train a model that reads **40 characters** of text and predicts the **41st**. Once it can do that, you can make it *write*: give it a starting phrase, let it predict one character, append that character, slide the window forward, and repeat. If it learned well, the output should look like the training text in **style**: same vocabulary, punctuation, line lengths and "feel". The required model is two stacked `SimpleRNN(256)` layers. As extras, the notebook adds an **LSTM**, baselines, temperature experiments, a look at the model's predictions, and model saving.

| Task from the sheet | Where in notebook |
|---|---|
| 1. Load text → char-to-index and index-to-char | §1 |
| 2. 40-char sequences → predict 41st; vocabulary, mapping, vectorize | §2 |
| 3. Create training sequences | §3 |
| 4. Build RNN: `SimpleRNN(256, return_sequences=True)` → `SimpleRNN(256)` | §4 |
| 5. Train the model | §5 |
| 6. Write text generation function | §6 |
| 7. Generate new text and validate | §7 |

---

## 2. Files and how to run

```
lab04/
├── lab04.ipynb              # the notebook (already executed, outputs saved)
├── lab04_explanation.md     # this file
├── data/
│   └── sonnets.txt          # training text, 47.8 KB
└── models/
    └── lstm_char_model.keras  # created by §12 when you run the notebook
```

**Dataset.** The first 80 of *Shakespeare's Sonnets*, taken from Project Gutenberg (public domain). Header, licence text and Roman-numeral sonnet titles were removed. A blank line separates sonnets. Curly quotes were changed to plain ASCII quotes. The file is 48,900 characters (47.8 KB), under the 50 KB limit, and the notebook asserts that.

**Requirements.** Python 3.10–3.12 (TensorFlow does not support newer Python yet), `tensorflow` (or `tensorflow-cpu`), `numpy`, `matplotlib`, `jupyter`.

```bash
# example with uv (any venv works)
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python tensorflow-cpu matplotlib jupyter
.venv/bin/jupyter notebook lab04.ipynb
```

Run it from inside `lab04/` so `data/sonnets.txt` resolves. On a 12-core CPU the whole notebook takes about 8 minutes: SimpleRNN about 6 s per epoch, LSTM about 25 s per epoch. No GPU is needed.

---

## 3. Background: what is a character-level language model?

A **language model** assigns probabilities to text. Here it learns

```
P(next character | previous 40 characters)
```

**Character-level** means the units are single characters (`a`, `b`, space, newline, `,` …) and not words. Pros and cons:

| | Character-level | Word-level |
|---|---|---|
| Vocabulary | tiny (36 here) | thousands |
| Unknown words | impossible: any word can be spelled | out-of-vocabulary problem |
| What must be learned | spelling, *then* words, *then* grammar | grammar directly |
| Sequence length for same meaning | long | short |

It suits a small dataset well: 49 K characters gives about 16 K training examples, but only about 9 K *words*, which is far too few for a word-level model.

### Why a Recurrent Neural Network?

A normal dense network takes a fixed-size input with no idea of order. An **RNN** reads a sequence one step at a time and keeps a **hidden state** `h`, a vector that summarises everything read so far:

```
h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

The *same* weights `W_x, W_h, b` are used at every time step (weight sharing). After 40 steps, `h_40` is a 256-number summary of the whole window, and a softmax layer turns it into a probability for each of the 36 characters.

```
 x1     x2     x3          x40
  │      │      │            │
  ▼      ▼      ▼            ▼
[RNN]─►[RNN]─►[RNN]─► … ─►[RNN]──► h40 ──► Dense+softmax ──► P(41st char)
 h1     h2     h3
```

---

## 4. Section-by-section walkthrough

### §0 Imports and reproducibility

```python
keras.utils.set_random_seed(42)
```
This seeds Python, NumPy and TensorFlow together. Weight initialisation, shuffling, dropout masks and text sampling are all random, so without a seed every run would print different numbers. (Some TF CPU kernels are still not perfectly deterministic, so numbers can differ slightly between machines.)

### §1 Load text and build mappings (Task 1)

```python
assert size_kb <= 50
text = f.read().lower()
vocab = sorted(set(text))
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)
```

* **Size check:** enforces the "≤ 50 KB" input rule from the sheet.
* **`.lower()`:** cuts the vocabulary from 58 to **36** characters. With so little data, the model would otherwise have to learn `T` and `t` separately even though they behave almost the same. The trade-off is that generated text has no capitals.
* **`sorted(set(text))`:** the unique characters in a fixed, repeatable order, so each character always gets the same ID.
* **`char2idx`** (dict) maps character → integer. **`idx2char`** (NumPy array) maps integer → character. Because it is an array, it can decode a whole list at once: `idx2char[[15, 27, 24, 22]]` → `['f','r','o','m']`.

Result: vocabulary is `"\n !',-.:;?abcdefghijklmnopqrstuvwxyz"`. The **newline** is a character too, which is how the model learns where lines end.

**Self-check:** encodes 100 characters and decodes them back. The `assert` proves the two mappings are exact inverses.

### §2 Vectorize and make 40→41 sequences (Task 2)

**Vectorize** = turn the whole text into one integer array `encoded_text` (length 48,900).

**Sliding window:**

```python
for i in range(0, len(encoded_text) - SEQ_LEN, STEP):
    X_all.append(encoded_text[i : i + 40])   # 40 input chars
    y_all.append(encoded_text[i + 40])       # the 41st char = label
```

Example from the output:

| input (40 chars) | target |
|---|---|
| `'from fairest creatures we desire increas'` | `'e'` |
| `'m fairest creatures we desire increase,\n'` | `'t'` |
| `'airest creatures we desire increase,\ntha'` | `'t'` |

* **`STEP = 3`:** each window starts 3 characters after the previous one, giving **16,287** examples. `STEP = 1` would triple the data but the extra windows are almost duplicates, so each epoch would take 3× longer for little gain.
* **Shapes:** `X_all` is `(16287, 40)` and `y_all` is `(16287,)`. Labels are plain integers, not one-hot (see §4, loss function).
* **Self-check asserts** confirm the second window really starts at offset 3 and its label is the character right after it.

**One-hot vs Embedding (the small demo cell).** The classic textbook way feeds each character as a one-hot vector of length 36 (`'f'` → 1 at position 15, 0 elsewhere). This notebook uses an **Embedding layer** instead: a learned table of 36 rows × 64 columns, where row *i* is the vector for character *i*. Multiplying a one-hot vector by a weight matrix just *selects one row*, so an embedding is the same maths without building the big sparse one-hot tensor. It is also *learned*, so similar characters (for example vowels) can end up with similar vectors.

### §3 Training sequences / train-validation split (Task 3)

```python
split = int(len(X_all) * 0.9)
X_train, X_val = X_all[:split], X_all[split:]
```

**14,658** training and **1,629** validation sequences. The validation set is the **last 10 % of the text**, not a random 10 %.

**Why not a random split?** Window *k* and window *k+1* share 37 of their 40 characters. With a random split, nearly every validation window would have an almost-identical twin in the training set, so validation loss would mostly measure memorisation. This is called **data leakage**. A contiguous split means validation lines (here, roughly the last 8 sonnets) were never seen in any form.

The Keras `fit` call still **shuffles the training batches** every epoch (its default), which is fine. Order inside the 40-char window is preserved; only the order of windows changes.

### §4 Build the RNN model (Task 4)

```python
def build_model(rnn_layer, name):
    model = keras.Sequential([
        keras.Input(shape=(SEQ_LEN,)),
        keras.layers.Embedding(VOCAB_SIZE, 64),
        rnn_layer(256, return_sequences=True),
        rnn_layer(256),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(VOCAB_SIZE, activation='softmax'),
    ])
```

The function takes the recurrent layer type as an argument, so **the same code builds the SimpleRNN model (Task 4) and the LSTM model (§9)**. That keeps the comparison fair.

| Layer | Output | Params | Why |
|---|---|---|---|
| Embedding(36, 64) | (40, 64) | 2,304 = 36×64 | char ID → dense vector |
| SimpleRNN(256, return_sequences=True) | (40, 256) | 82,176 = 64·256 + 256·256 + 256 | outputs `h_t` for **every** step |
| SimpleRNN(256) | (256,) | 131,328 = 256·256 + 256·256 + 256 | outputs only the **final** `h_40` |
| Dropout(0.2) | (256,) | 0 | during training, randomly zeroes 20 % of values |
| Dense(36, softmax) | (36,) | 9,252 = 256·36 + 36 | probability for each character |
| **Total** | | **225,060** | |

**`return_sequences=True` explained.** By default an RNN layer returns only its last hidden state. To **stack** RNNs, the second layer needs a whole *sequence* as input, so the first layer must return all 40 hidden states. The second, final RNN returns just the last one because we predict a single next character.

**Stacking** lets layer 1 learn low-level patterns (letter combinations, syllables) and layer 2 build higher-level ones (words, phrase rhythm) on top.

**Compile settings:**

* **Loss `sparse_categorical_crossentropy`:** the same cross-entropy as `categorical_crossentropy`, but it accepts integer labels (`y = 14`) instead of one-hot labels. This saves memory.
  Cross-entropy for one example = `−log P(correct char)`. If the model gives the right character probability 1.0, the loss is 0. If it gives 0.01, the loss is 4.6.
* **Adam, lr = 0.002:** an adaptive optimizer. 0.002 is a bit above the 0.001 default so that early learning goes faster. The LR scheduler lowers it later.
* **`clipnorm=1.0`, gradient clipping:** in backpropagation through time, the gradient is multiplied by `W_h` once per time step. If that matrix's values are a bit above 1, gradients can **explode** (grow exponentially) and one bad batch can wreck the weights. Clipping rescales any gradient whose norm is above 1.0. This is standard practice for RNNs.

### §5 Train the model (Task 5)

```python
EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-4)
```

* **EarlyStopping:** stops when validation loss has not improved for 4 epochs, then **rolls back** to the weights from the best epoch. `EPOCHS = 40` is only an upper limit.
* **ReduceLROnPlateau:** when validation loss stalls for 2 epochs, it halves the learning rate (0.002 → 0.001 → 0.0005 …). Smaller steps help the model settle into a better minimum.
* **Batch size 128:** 115 gradient updates per epoch.

**What happened in the run (SimpleRNN):**

| Epoch | train loss | val loss | val acc | note |
|---|---|---|---|---|
| 1 | 3.27 | 3.05 | 0.16 | barely better than guessing "space" every time |
| 4 | 2.73 | 2.43 | 0.29 | breaks out of the plateau |
| 10 | 2.03 | 2.01 | 0.41 | |
| 18 | 1.71 | 1.95 | 0.43 | LR halves after 19–20 |
| **22** | 1.50 | **1.935** | **0.448** | **best epoch, weights restored** |
| 26 | 1.32 | 1.98 | 0.44 | stopped: train loss falls, val rises = overfitting |

**Reading the curves:** the train and validation lines move together at first. After about epoch 20 they **diverge**: training loss keeps dropping while validation loss starts rising. That is the model starting to memorise the 44 K training characters instead of learning general patterns. EarlyStopping catches this.

**The first-epoch plateau** (loss stuck near 3.0–3.2) is typical for SimpleRNN. The fastest way to lower loss at first is to predict the most common characters (space, `e`), so the model needs a few epochs before it starts using the context.

### §6 Text generation function (Task 6)

```python
def generate_text(model, seed, length=400, temperature=0.7):
    seed = ''.join(c for c in seed.lower() if c in char2idx)
    ids = [char2idx[c] for c in seed]
    for _ in range(length):
        window = ids[-SEQ_LEN:]
        window = [char2idx[' ']] * (SEQ_LEN - len(window)) + window
        probs = model(np.array([window]), training=False).numpy()[0]
        ids.append(sample_index(probs, temperature))
    return ''.join(idx2char[ids])
```

Step by step:

1. **Clean the seed:** lowercase it and drop characters not in the vocabulary (for example digits). Otherwise the dict lookup would crash.
2. **Take the last 40 IDs.** If the seed is shorter than 40, **left-pad with spaces** so the input shape is always `(1, 40)`.
3. **Forward pass** gives a probability vector of length 36. `training=False` turns Dropout off. Calling `model(...)` directly is much faster than `model.predict(...)` for one sample in a loop, because `predict` has per-call setup overhead.
4. **Sample** the next character and append it. The window slides forward by one, so the model now conditions on its *own* output. This is called *autoregressive* generation.

**Sampling with temperature:**

```python
logits = np.log(probs + 1e-10) / temperature
probs  = exp(logits - max) / sum(...)
```

* Dividing log-probabilities by `T` then re-normalising is the same as `p_i^(1/T)`.
* `T → 0`: the largest probability takes over, giving **greedy** decoding (always the most likely character). This tends to loop: "the world the world the world…".
* `T = 1`: sample exactly from the model's distribution.
* `T > 1`: the distribution gets flatter, so rare characters get picked more and spelling falls apart.
* `+1e-10` avoids `log(0)`. Subtracting `max` before `exp` prevents overflow. This is the standard *numerically stable softmax*.

**Why sample instead of always taking the argmax?** Language is not deterministic. After "my love is as a " many words are reasonable. Greedy decoding throws that variety away and gets stuck in loops.

### §7 Generate and validate (Task 7)

**7a. Sample** (SimpleRNN, T = 0.6, seed "shall i compare thee to a summer's day?"):

```
hoth farth when i former the bearts por ard in their stren
i dome as wom the beauty she the sours colder of i eeturs, thou art the ginker on my deartes dive thou is the stare,
thou art the herst thou hading seef thou mend,
...
and seath the beauty, and carse of the world till as me the spare ...
```

What it learned: the sonnet vocabulary (*thou, thy, beauty, world, love*), commas and line breaks, the address "thou art the …", and mostly correct short words. What it did not learn: long-range sense, consistent spelling of longer words ("deartes", "eeturs"), and the 14-line form.

**7b. Validation metrics.** "Similar in style" is judged in two ways.

*Prediction quality on unseen text:*

| Metric | Meaning | SimpleRNN |
|---|---|---|
| val loss | average `−log P(correct char)` on the held-out 10 % | 1.935 |
| val accuracy | how often the top guess is right | 44.8 % |
| **perplexity** = e^loss | "as uncertain as choosing uniformly among *N* characters" | **6.92** (uniform guess = 36) |

So instead of a 1-in-36 guess, the model is effectively choosing among about 7 plausible characters.

*Style of the generated text:*

| Metric | How it's computed | SimpleRNN | Ideal |
|---|---|---|---|
| real-word rate | share of generated words that appear somewhere in training text | 66 % | high |
| char-distribution distance | total variation distance = ½ Σ \|p_gen(c) − p_train(c)\| | 0.147 | near 0 |
| avg line length | mean characters per non-empty line | 61.6 | training ≈ 42.6 |
| copied 20-grams | share of generated 20-char chunks found verbatim in training | 0 % | low (not memorising) |

How to read this: about two-thirds of the "words" are real, the letter frequencies are close to Shakespeare's, and nothing is copied. **The text is new, not memorised.** Lines run long (61 vs 43 characters), which shows the model learned *that* lines end with punctuation and a newline but not exactly *when*.

**Self-check assert:** running `style_metrics` on the first real sonnet must give 100 % real words and 100 % copied 20-grams. This proves the metric code is correct before it is used to judge the model. (It caught a bug while this notebook was being written: cutting the text at a fixed 2,000 characters split the last word in half.)

### §8 Extra: n-gram baselines

A perplexity of 6.92 only means something next to a comparison. Two non-neural models are scored on the *same* validation targets:

* **Unigram:** P(c) = how often `c` appears in training text. It ignores context completely.
* **Bigram:** P(c | previous char), counted from training text. **Add-one (Laplace) smoothing** (start every count at 1) ensures no pair has probability zero; otherwise one unseen pair would make the loss infinite.

`np.add.at(bi, (prev, next), 1)` fills the 36×36 count table in one vectorised call. (Plain `bi[prev, next] += 1` would count repeated pairs only once.)

| Model | Context used | Val perplexity |
|---|---|---|
| Uniform guess | none | 36.00 |
| Unigram | none (frequency only) | 19.83 |
| Bigram | 1 previous char | 10.58 |
| **SimpleRNN** | 40 previous chars | **6.92** |
| **LSTM** | 40 previous chars | **6.58** |

Each step down shows that more context helps. The RNNs beat the bigram by a wide margin, which proves they use information from beyond the last character.

### §9 Extra: LSTM model and comparison

**The problem with SimpleRNN: vanishing gradients.** To learn "the `t` 30 characters ago affects this prediction", the gradient must flow backwards through 30 copies of `tanh` and `W_h`. Each step usually multiplies it by a number below 1, so after 30 steps it is close to zero. In practice a SimpleRNN mostly uses the last 10–20 characters.

**LSTM (Long Short-Term Memory)** adds a separate **cell state** `c_t`, a memory lane controlled by three sigmoid **gates** whose values are between 0 and 1:

```
f_t = σ(W_f·[h_{t-1}, x_t] + b_f)      forget gate: how much old memory to keep
i_t = σ(W_i·[h_{t-1}, x_t] + b_i)      input gate: how much new info to write
g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)   candidate new info
o_t = σ(W_o·[h_{t-1}, x_t] + b_o)      output gate: how much memory to reveal

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        ← additive update: the key idea
h_t = o_t ⊙ tanh(c_t)
```

Because `c_t` is updated by **addition**, and the forget gate can stay close to 1, the gradient can pass back through many steps without shrinking. The network *learns* what to remember.

**Cost:** 4 weight sets (f, i, g, o) instead of 1, so about **4× the parameters** (865,572 vs 225,060) and about **4× slower per epoch** (about 25 s vs 6 s).

**Head-to-head (same data, same hyper-parameters, same callbacks):**

| Metric | SimpleRNN | LSTM |
|---|---|---|
| parameters | 225,060 | 865,572 |
| epochs until early stop | 26 (best: 22) | 11 (best: 7) |
| val loss | 1.935 | **1.884** |
| val accuracy | **44.8 %** | 44.0 % |
| perplexity | 6.92 | **6.58** |
| real-word rate | **66 %** | 64 % |
| char-distribution distance | 0.147 | **0.131** |
| copied 20-grams | 0 % | 0 % |

**Interpretation:**

* The LSTM reaches a **lower validation loss in a third of the epochs** (best at epoch 7 vs 22). Its gates make learning much faster, with no slow start like the SimpleRNN's.
* It also **overfits faster**: by epoch 11 its training accuracy is 61 % while validation stays at 46 %. With 865 K parameters and only 44 K training characters, the LSTM has far more capacity than the data needs. Early stopping is essential here.
* The gap is **small on this task** because the window is only 40 characters, where the SimpleRNN's short memory is almost enough. LSTMs pull ahead with longer windows and larger datasets.
* The style metrics are close and somewhat noisy, since they come from one 500-character random sample. Validation loss and perplexity are the more reliable comparison.

The comparison plot shows validation loss per epoch for both models, plus a bar chart of all perplexities from uniform guess down to LSTM.

### §10 Extra: temperature sweep

The better model (LSTM, chosen automatically by validation loss) generates 250 characters at four temperatures:

| T | Real words | Character |
|---|---|---|
| 0.2 | 93 % | "the world the with the world the dead" — real words, endless repetition |
| 0.5 | 72 % | "and then thine own more the mord still from whought more," — best balance |
| 1.0 | 37 % | "be the indward stweeth bered." — creative, many invented words |
| 1.5 | 24 % | "ovad uperutt! lotkienyr dbow" — mostly noise |

**Takeaway:** temperature is a *generation-time* setting, not a training one. The same trained weights can produce safe, boring text or wild text. For this model, **0.5–0.7** gives the most Shakespeare-like output.

### §11 Extra: top-5 next-character probabilities

For three prompts, a bar chart shows the five most likely next characters:

What the saved LSTM actually predicted in this run:

* `"...more lovely and more temperat"` → `e` 47 %, `i` 18 % (*temperate* / *temperature*-like). The model has learned the common `-ate` ending. This tests **spelling** knowledge.
* `"my love is as a fev"` → `a` 29 %, `o` 24 %, `e` 15 %. The correct `e` (*fever*) is only third. *fever* is rare in the training text, so the model falls back on "a vowel usually follows a consonant". This shows the **limit of a small dataset**.
* `"...summer's lease hath all too short a "` → `m` 11 %, `s` 10 %, `l` 9 %, `f` 8 %, `t` 8 %. After a space, many words can start, so probability spreads thinly over common first letters. This shows **uncertainty** where the language really is open.

This turns the model from a black box into something you can inspect: a confident spike means the model "knows", and a flat spread means many options are plausible.

### §12 Extra: save and reload

```python
best_model.save('models/lstm_char_model.keras')
reloaded = keras.models.load_model(path)
assert np.allclose(best_model(probe), reloaded(probe), atol=1e-6)
```

The `.keras` format stores architecture, weights and optimizer state in one zip file (10.4 MB, mostly because the optimizer keeps 2 extra copies of every weight). The assert proves the reloaded model gives identical predictions. A final sample is generated from the reloaded model to show it works on its own.

> Note: `models/` is an output file. You probably do not want to commit a 10 MB binary to git.

---

## 5. Results summary

* **Task met:** the model generates text that clearly resembles the sonnets in vocabulary (*thou, thee, thy, beauty, love, world*), punctuation (commas and colons at line ends), structure (short lines broken by newlines) and tone. It writes new text without copying any 20-character chunk.
* **Quantitatively:** the SimpleRNN reaches perplexity **6.92** and the LSTM **6.58** on unseen text, against 10.6 for a bigram model and 36 for random guessing.
* **LSTM vs SimpleRNN:** the LSTM learns about 3× faster in epochs and ends with slightly lower loss, but uses 4× the parameters and overfits sooner on this small dataset.
* **Temperature** around 0.5 gives the best balance between correctness and variety.

---

## 6. Limitations and how to improve

| Limitation | Why | Fix |
|---|---|---|
| Many misspelled / invented words | only 49 K characters of training data | more text (all 154 sonnets, the plays) |
| Overfits after ~7–22 epochs | model capacity ≫ data size | more dropout, `recurrent_dropout`, smaller layers, more data |
| No long-range meaning | 40-char window ≈ 7 words | longer `SEQ_LEN` (100+), stateful RNN, or a Transformer |
| Lines too long (61 vs 43 chars) | line breaks depend on syllable count, which is hard to learn | longer context, more data |
| Lowercase only | we lowercased to shrink the vocabulary | keep case if there is enough data |
| Style metrics are noisy | based on one random sample | average over several samples / seeds |

Other ideas: **GRU** (a lighter gated RNN with 3 gates instead of 4), **bidirectional** layers (only for analysis, not generation, since they need the future), **beam search** decoding, and **top-k / nucleus (top-p) sampling** as alternatives to temperature.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **Vocabulary** | set of distinct tokens; here, 36 characters |
| **Vectorize** | convert text to numbers (integer IDs) |
| **One-hot** | a vector of zeros with a single 1 marking the token |
| **Embedding** | learned dense vector for each token; a lookup table |
| **Hidden state `h_t`** | the RNN's running summary of the sequence up to step *t* |
| **`return_sequences`** | return all `h_1…h_T` (True) or only `h_T` (False) |
| **BPTT** | backpropagation through time: unrolling the RNN over 40 steps to compute gradients |
| **Vanishing / exploding gradient** | gradients shrinking to 0 or blowing up after many multiplications by the same matrix |
| **Gradient clipping** | rescaling gradients whose norm exceeds a limit |
| **Cell state `c_t`** | the LSTM's additive long-term memory |
| **Gate** | sigmoid-valued (0–1) multiplier that decides how much information passes |
| **Cross-entropy** | `−log P(correct answer)`, averaged |
| **Perplexity** | `exp(cross-entropy)`, the effective number of choices |
| **Temperature** | divides log-probabilities before sampling; controls randomness |
| **Autoregressive** | each output is fed back in as the next input |
| **Early stopping** | stop training when validation stops improving |
| **Data leakage** | validation data that overlaps training data, giving over-optimistic scores |
| **Laplace smoothing** | add 1 to every count so nothing has zero probability |

---

## 8. Likely viva questions

**Q: Why 40 characters?** It is long enough to cover about 7 words (a phrase or half a line of verse) and short enough to train quickly. It is a trade-off between context and speed or memory.

**Q: Why does the first SimpleRNN have `return_sequences=True` and the second not?** The second RNN needs a sequence as input. The final one only needs to produce one vector for one prediction.

**Q: Why `sparse_` categorical cross-entropy?** The labels are integer IDs, not one-hot vectors. The maths is identical and it uses less memory.

**Q: Why not split train/validation randomly?** Overlapping windows would leak almost-identical sequences into validation.

**Q: How do you know the model isn't just copying?** 0 % of generated 20-character chunks appear in the training text.

**Q: Why does accuracy look low (≈45 %)?** Next-character prediction is uncertain by nature. After a space, dozens of letters are valid. Perplexity 6.6–6.9 vs 36 for random guessing is the better measure.

**Q: Why is the LSTM better, and why only slightly?** Gates and the additive cell state avoid vanishing gradients, so it uses more of the context and learns faster. With a short 40-character window and little data, the SimpleRNN's shorter memory is almost enough, and the LSTM's extra capacity mostly goes to overfitting.

**Q: What does temperature do?** It sharpens (T < 1) or flattens (T > 1) the predicted distribution before sampling, trading correctness for variety.

**Q: What is an Embedding layer equivalent to?** A one-hot vector multiplied by a weight matrix (a dense layer without bias), done as a fast row lookup.
