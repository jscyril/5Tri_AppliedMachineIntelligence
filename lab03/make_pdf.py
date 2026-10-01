from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path('text_latent_representation_report.pdf')
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

def page_title(fig, title, subtitle=None):
    fig.text(0.08, 0.94, title, fontsize=19, weight='bold', color='#000000')
    if subtitle:
        fig.text(0.08, 0.905, subtitle, fontsize=10.5, color='#1f4e79')

def add_table(ax, data, headers, font=9, scale=(1, 1.25)):
    ax.axis('off')
    t = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='left')
    t.auto_set_font_size(False)
    t.set_fontsize(font)
    t.scale(*scale)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#D9D9D9')
        if r == 0:
            cell.set_facecolor('#1F4E79')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
            cell.get_text().set_ha('center')
        elif r % 2 == 0:
            cell.set_facecolor('#F3F6F9')
    return t

with PdfPages(OUT) as pdf:
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig, 'Text Latent Representation Using Tokenization and Word Embeddings', 'Customer review dataset laboratory report')
    fig.text(0.08, 0.83, 'Objective', fontsize=13, weight='bold')
    fig.text(0.08, 0.795, 'Transform raw customer reviews into numerical representations using preprocessing, tokenization, padding, dense word embeddings, flattening, and PCA.', fontsize=11, wrap=True)
    fig.text(0.08, 0.70, 'Dataset', fontsize=13, weight='bold')
    y = 0.665
    for s in sentences:
        fig.text(0.11, y, '• ' + s, fontsize=11)
        y -= 0.045
    fig.text(0.08, 0.40, 'Method summary', fontsize=13, weight='bold')
    fig.text(0.08, 0.365, 'Lowercase and clean text → assign token IDs → pad to a common length → map IDs to four-dimensional vectors → flatten each matrix → project word vectors with PCA.', fontsize=11, wrap=True)
    fig.text(0.08, 0.20, 'Note', fontsize=13, weight='bold')
    fig.text(0.08, 0.165, 'The embeddings in this report are reproducible illustrative vectors generated with a fixed random seed. They are not pretrained semantic embeddings.', fontsize=10.5, wrap=True)
    pdf.savefig(fig); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(8.5, 11))
    page_title(fig, 'Vocabulary and Token Sequences')
    items = list(vocab.items())
    vocab_rows = []
    for i in range(0, len(items), 2):
        a = items[i]; b = items[i + 1] if i + 1 < len(items) else ('', '')
        vocab_rows.append([a[0], a[1], b[0], b[1] if b[0] else ''])
    add_table(axs[0], vocab_rows, ['Word', 'Token ID', 'Word', 'Token ID'], font=9, scale=(1, 1.35))
    axs[0].set_title('Vocabulary with corresponding token IDs', loc='left', fontsize=12, pad=10)
    seq_rows = [[i, str(seq), str(pad)] for i, (seq, pad) in enumerate(zip(ids, padded), 1)]
    add_table(axs[1], seq_rows, ['Sample', 'Token sequence', 'Padded sequence'], font=8.5, scale=(1, 1.45))
    axs[1].set_title(f'Numerical representation and padding to length {max_len}', loc='left', fontsize=12, pad=10)
    pdf.savefig(fig); plt.close(fig)

    for i, (s, row) in enumerate(zip(sentences, matrices), 1):
        fig = plt.figure(figsize=(8.5, 11))
        page_title(fig, f'Word Embedding Representation Sample {i}', s)
        fig.text(0.08, 0.84, 'Each token is represented by a dense vector with four dimensions. The final row is padding when the sentence is shorter than the maximum length.', fontsize=10.5, wrap=True)
        ax = fig.add_axes([0.10, 0.18, 0.80, 0.57])
        ax.axis('off')
        cell = [[f'Position {j}', *[f'{x:.3f}' for x in v]] for j, v in enumerate(row, 1)]
        t = ax.table(cellText=cell, colLabels=['Position', 'd1', 'd2', 'd3', 'd4'], loc='center', cellLoc='center')
        t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.2, 1.7)
        for (r, c), cell_obj in t.get_celld().items():
            cell_obj.set_edgecolor('#D9D9D9')
            if r == 0 or c == 0:
                cell_obj.set_facecolor('#1F4E79'); cell_obj.get_text().set_color('white'); cell_obj.get_text().set_weight('bold')
            elif r % 2 == 0:
                cell_obj.set_facecolor('#F3F6F9')
        pdf.savefig(fig); plt.close(fig)

    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig, 'Flattened Vector Representations')
    fig.text(0.08, 0.885, f'Each {max_len} × {dim} embedding matrix is flattened row by row into a single vector of length {max_len * dim}.', fontsize=10.5)
    y = 0.82
    for i, v in enumerate(flattened, 1):
        fig.text(0.08, y, f'Sample {i}', fontsize=10.5, weight='bold')
        fig.text(0.08, y - 0.032, '[' + ', '.join(f'{x:.3f}' for x in v) + ']', fontsize=7.4, family='monospace', wrap=True)
        y -= 0.145
    pdf.savefig(fig); plt.close(fig)

    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig, 'PCA Visualization and Analysis')
    ax = fig.add_axes([0.11, 0.47, 0.78, 0.38])
    ax.scatter(coords[:, 0], coords[:, 1], s=48, color='#1f4e79')
    for word, idx in vocab.items():
        x, y = coords[idx - 1]
        ax.annotate(word, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Principal Component 1'); ax.set_ylabel('Principal Component 2')
    ax.grid(alpha=0.25)
    fig.text(0.08, 0.415, 'Figure 1. Two-dimensional PCA projection of the word embeddings.', fontsize=9, style='italic')
    fig.text(0.08, 0.34, 'Analysis', fontsize=13, weight='bold')
    fig.text(0.08, 0.255, 'Tokenization converts the reviews into discrete IDs, while padding makes every sample compatible with a single rectangular tensor. The embedding layer changes each ID into a dense vector, preserving a fixed four-value representation per token. Flattening then produces a fixed-length sentence vector with 28 values.', fontsize=10.5, wrap=True)
    fig.text(0.08, 0.16, 'Interpretation', fontsize=13, weight='bold')
    fig.text(0.08, 0.105, 'PCA provides a compact visual summary of the embedding space. Because these vectors are illustrative and randomly generated, the plotted distances should not be interpreted as learned language semantics.', fontsize=10.5, wrap=True)
    pdf.savefig(fig); plt.close(fig)

print(OUT)
