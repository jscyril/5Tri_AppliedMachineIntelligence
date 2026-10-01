from pathlib import Path
import re
import html
import numpy as np
import matplotlib.pyplot as plt

HTML = Path('text_latent_representation_report.html')
PLOT = Path('pca_embeddings.png')

sentences = [
    'The product quality is excellent',
    'I really love this product',
    'The quality of this item is amazing',
    'The product performance is poor',
    'I dislike this product completely',
]
tokens = [[w.lower() for w in re.findall(r"[A-Za-z]+", s)] for s in sentences]
vocab = {}
for row in tokens:
    for word in row:
        if word not in vocab:
            vocab[word] = len(vocab) + 1
ids = [[vocab[w] for w in row] for row in tokens]
max_len = max(map(len, ids))
padded = [row + [0] * (max_len - len(row)) for row in ids]

rng = np.random.default_rng(7)
dim = 4
embedding = {0: np.zeros(dim)}
for word, idx in vocab.items():
    embedding[idx] = rng.normal(0, 1, dim).round(3)
matrices = [np.array([embedding[i] for i in row]) for row in padded]
flattened = [m.flatten() for m in matrices]

word_matrix = np.array([embedding[i] for i in range(1, len(vocab) + 1)])
centered = word_matrix - word_matrix.mean(axis=0)
_, _, vh = np.linalg.svd(centered, full_matrices=False)
coords = centered @ vh[:2].T

plt.figure(figsize=(8.2, 5.3), dpi=180)
plt.scatter(coords[:, 0], coords[:, 1], s=45, color='#1f4e79')
for word, idx in vocab.items():
    x, y = coords[idx - 1]
    plt.annotate(word, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=8)
plt.title('PCA Projection of Word Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(PLOT, bbox_inches='tight')
plt.close()

def esc(x):
    return html.escape(str(x))

def vector(v):
    return '[' + ', '.join(f'{x:.3f}' for x in v) + ']'

rows_vocab = []
items = list(vocab.items())
for i in range(0, len(items), 2):
    a = items[i]
    b = items[i + 1] if i + 1 < len(items) else ('', '')
    rows_vocab.append(f'<tr><td>{esc(a[0])}</td><td class="center">{a[1]}</td><td>{esc(b[0])}</td><td class="center">{b[1] if b[0] else ""}</td></tr>')

rows_seq = []
for i, (seq, pad) in enumerate(zip(ids, padded), 1):
    rows_seq.append(f'<tr><td class="center">{i}</td><td>{esc(seq)}</td><td>{esc(pad)}</td></tr>')

matrices_html = []
for i, (s, row) in enumerate(zip(sentences, matrices), 1):
    matrices_html.append(f'<p class="label">Sample {i}: {esc(s)}</p><pre>{esc(chr(10).join(vector(v) for v in row))}</pre>')

flat_html = ''.join(f'<pre>Sample {i}: {esc(vector(v))}</pre>' for i, v in enumerate(flattened, 1))

document = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Text Latent Representation</title>
<style>
@page {{ size: Letter; margin: 0.65in; }}
body {{ font-family: Arial, sans-serif; color:#222; font-size:10.5pt; line-height:1.35; }}
h1 {{ font-size:16pt; color:#000; margin-top:20px; margin-bottom:7px; }}
h2 {{ font-size:12.5pt; color:#000; margin-top:14px; margin-bottom:5px; }}
.title {{ font-size:24pt; font-weight:700; line-height:1.1; margin-bottom:5px; }}
.subtitle {{ font-size:12pt; color:#1f4e79; margin-bottom:18px; }}
p {{ margin: 0 0 7px 0; }}
ul {{ margin-top:3px; margin-bottom:8px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 14px 0; }}
th, td {{ border:1px solid #d9d9d9; padding:5px 7px; vertical-align:middle; }}
th {{ background:#1f4e79; color:white; text-align:center; }}
tr:nth-child(even) td {{ background:#f3f6f9; }}
.center {{ text-align:center; }}
pre {{ font-family:"Courier New", monospace; font-size:8pt; background:#f7f8fa; border-left:3px solid #1f4e79; padding:6px 10px; margin:3px 0 7px 12px; white-space:pre-wrap; }}
.label {{ font-weight:bold; font-size:9.5pt; margin-top:7px; margin-bottom:1px; }}
.figure {{ text-align:center; margin-top:10px; }}
.figure img {{ width:6.4in; max-width:100%; }}
.caption {{ text-align:center; font-style:italic; font-size:9pt; }}
</style></head><body>
<div class="title">Text Latent Representation Using Tokenization and Word Embeddings</div>
<div class="subtitle">Customer review dataset laboratory report</div>
<p>This report transforms five customer reviews into numerical representations. It shows the complete path from cleaned text to token IDs, padded sequences, four-dimensional word embeddings, flattened sentence vectors, and a two-dimensional PCA visualization.</p>

<h1>1 Dataset and preprocessing</h1>
<p>The text is converted to lowercase and punctuation is removed. The resulting word tokens are kept in their original sentence order. Padding uses token ID 0, which is reserved for the padding symbol.</p>
<ul>{''.join(f'<li>{esc(s)}</li>' for s in sentences)}</ul>

<h1>2 Vocabulary and token IDs</h1>
<p>The vocabulary is created by assigning IDs in first-appearance order. Token ID 0 is reserved for padding.</p>
<table><tr><th>Word</th><th>Token ID</th><th>Word</th><th>Token ID</th></tr>{''.join(rows_vocab)}</table>

<h1>3 Token sequences and padding</h1>
<p>All sequences are padded to the maximum sentence length of {max_len} tokens.</p>
<table><tr><th>Sample</th><th>Token sequence</th><th>Padded sequence</th></tr>{''.join(rows_seq)}</table>

<h1>4 Word embedding representation</h1>
<p>For demonstration, each non-padding token is mapped to a reproducible four-dimensional dense vector generated with a fixed random seed. These vectors illustrate the representation format; they are not pretrained semantic embeddings. Padding is represented by [0.000, 0.000, 0.000, 0.000].</p>
<h2>Embedding matrix for each sentence</h2>
{''.join(matrices_html)}

<h1>5 Flattened vector representations</h1>
<p>Each {max_len} × {dim} embedding matrix is flattened row by row into a single vector of length {max_len * dim}.</p>
{flat_html}

<h1>6 PCA visualization</h1>
<p>PCA reduces the four-dimensional word vectors to two principal components. Words that appear near one another in the plot have similar coordinates under this illustrative embedding assignment. With a real pretrained embedding model, relative positions would be learned from language data and would carry stronger semantic meaning.</p>
<div class="figure"><img src="{PLOT.name}"><div class="caption">Figure 1. Two-dimensional PCA projection of the word embeddings.</div></div>

<h1>7 Analysis and conclusion</h1>
<p>Tokenization converts the reviews into discrete IDs, while padding makes every sample compatible with a single rectangular tensor. The embedding layer changes each ID into a dense vector, preserving a fixed four-value representation per token. Flattening then produces a fixed-length sentence vector with 28 values. The PCA plot provides a compact visual summary of the embedding space, but the positions should be interpreted as illustrative because the vectors were generated randomly rather than trained on a large corpus.</p>
</body></html>'''

HTML.write_text(document, encoding='utf-8')
print(HTML)

