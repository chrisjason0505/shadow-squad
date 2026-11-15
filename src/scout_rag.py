"""'Why this player?': the method notes and the built-in scout's report for the Streamlit app.

The method notes in app/knowledge/*.md are split by section and indexed with TF-IDF (scikit-learn) so the app can show
the notes behind a pick. `offline_answer` writes the scout's report from the candidate's dossier. No language-model API is called."""
import glob, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _chunks():
    out = []
    for path in sorted(glob.glob(f'{ROOT}/app/knowledge/*.md')):
        for part in re.split(r'\n(?=## )', open(path, encoding='utf-8').read()):
            lines = part.strip().splitlines()
            if not lines or lines[0].startswith('# '): lines = lines[1:]
            if lines: out.append({'title': lines[0].lstrip('# ').strip(), 'text': '\n'.join(lines[1:]).strip(), 'source': os.path.basename(path)})
    return [c for c in out if c['text']]


class Retriever:
    def __init__(self):
        self.chunks = _chunks()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        self.M = self.vec.fit_transform([f"{c['title']} {c['title']} {c['text']}".replace('_', ' ') for c in self.chunks])

    def search(self, query, k=4, exclude=()):
        """Top-k sections by TF-IDF cosine; `exclude` drops sections by title (the other mode's engine description)."""
        q = self.vec.transform([query.replace('_', ' ')]); s = (self.M @ q.T).toarray().ravel()
        idx = [i for i in np.argsort(-s) if s[i] > 0 and self.chunks[i]['title'] not in exclude][:k]
        return [{**self.chunks[i], 'score': float(s[i])} for i in idx]


def retrieval_query(question, drivers):
    """The question plus the names of the features that moved this candidate most."""
    return question + ' ' + ' '.join(drivers)


def build_prompt(question, dossier, passages):
    notes = '\n\n'.join(f"[{i + 1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages))
    return f"<method_notes>\n{notes}\n</method_notes>\n\n<dossier>\n{dossier}\n</dossier>\n\nQuestion: {question}"


def offline_answer(d):
    """Template scout report from the football reading, used when no API key is configured.
    d keys: name, rank, lo, hi, starter, club, role_s, role_c, pros, cons, keeps, down, adds, price (cand, starter), age (cand, starter), vs1, style, dims [(dim, starter tier, cand tier)], hs, top3, question."""
    cap = lambda x: x[0].upper() + x[1:] if x else x
    head = (f"**{d['name']} is the best replacement on the list for {d['starter']}.**" if d['rank'] == 1 else f"**{d['name']} is No. {d['rank']} on the list to replace {d['starter']}.**")
    txt = f"{head} {d['starter']} is {d['role_s']}; {d['name']} is {d['role_c']}. "
    if d['pros']: txt += 'The case for him: ' + '; '.join(d['pros'][:3]) + '. '
    if d['vs1'] and d['vs1'][1]: txt += f"{d['vs1'][0]} is ahead of him because " + ' and '.join(d['vs1'][1]) + '. '
    txt += f"He costs {d['price'][0]} against {d['starter']}'s {d['price'][1]}, at {d['age'][0]:.0f} against {d['age'][1]:.0f}."
    if d['lo'] is not None and d['hi'] - d['lo'] > 4: txt += f" The models disagree on him (anywhere from No. {d['lo']} to No. {d['hi']}), so treat the spot as a band."
    nerd = ['**For the football nerds**']
    if d['dims']: nerd.append('- Part by part (starter vs him): ' + '; '.join(f"{k.lower()} {a} vs {b}" for k, a, b in d['dims']) + '.')
    if d['keeps']: nerd.append('- Keeps what the club loses: ' + '; '.join(d['keeps'][:3]) + '.')
    if d['down']: nerd.append('- Step down: ' + '; '.join(d['down'][:3]) + '.')
    if d['adds']: nerd.append('- Adds: ' + '; '.join(d['adds'][:3]) + '.')
    if d['style']: nerd.append(f"- Fit: {d['club']} were {d['style']}; his numbers here are already adjusted to that style.")
    for x in d.get('reads') or []: nerd.append('- Style read: ' + x)
    if d['cons']: nerd.append('- Question marks: ' + '; '.join(d['cons']) + '.')
    show3 = d['top3'] and any(w in d['question'].lower() for w in ('top 3', 'top three', 'now', 'hindsight', 'where', 'value'))
    if d['hs'] and not (show3 and d['rank'] <= 3): nerd.append('- Hindsight: ' + d['hs'])
    if show3:
        nerd.append('- The top 3 today:\n' + '\n'.join('  - ' + x for x in d['top3']))
    return txt + '\n\n' + '\n'.join(nerd)
