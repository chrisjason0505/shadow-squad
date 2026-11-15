"""'Why this player?': a small retrieval-augmented explainer for the Streamlit app.

Retrieval: the method notes in app/knowledge/*.md are split by section and indexed with TF-IDF (scikit-learn, no
embedding service). A question is matched against them together with the names of the features that moved this
candidate most, so the passages that come back explain the numbers in the dossier.
Generation: the dossier (every number the engine has for this candidate and this query) and the retrieved passages go
to Claude, which answers in plain words first and then with the statistics. Without an API key the app falls back to
`offline_answer`, a template built from the same dossier, so the feature never breaks on hosting."""
import glob, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = 'claude-opus-5-5'
SYSTEM = """You are the chief scout behind Shadow Squad, a tool that ranks replacements for a player leaving a European club.
You talk to football people: fans, analysts and coaches who know the game better than they know statistics.

How to answer:
- Speak football. Talk about roles, profiles, what a player does on the pitch (progressive carries, passes into the box, winning the ball back, threat in the box, aerial duels), the club's style, price bracket, age and career stage, and what the team would keep, lose or gain.
- Translate the numbers. Say "top 10% of wingers for progressive carries", not "z = +1.4"; say "the model rates his price bracket", not "value_gap SHAP". Only use SHAP, cosine similarity or z-scores if the user asks about the model or the statistics.
- Make the case. Say why he is (or is not) the best replacement on the list, and if he is not No. 1, what No. 1 has over him.
- Use hindsight when the dossier has it: where the candidate and the top 3 are now, and how their Transfermarkt value moved, with the as-of date. Say whether the list looks good in hindsight, honestly.
- Use only the dossier and the notes. Never invent stats, fees, injuries, wages, quotes, news or current clubs. If something is not in the data, say so. Values are Transfermarkt estimates, not fees.
- Be honest about doubt: a wide rank range, a step down in an area that mattered, or that the engine finds the real signing in its top 10 only about a third of the time.

Format, unless the user asks for something else: a short verdict paragraph (two to four sentences), then a section headed "For the football nerds" with four to six bullet points on the like-for-like profile, the part of the game where he is a step up or down, the fit with the club's style, price and age, and hindsight if available. Under 300 words."""


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


def api_key(secrets=None):
    """The key from Streamlit secrets first, then the environment."""
    try:
        if secrets is not None and 'ANTHROPIC_API_KEY' in secrets: return secrets['ANTHROPIC_API_KEY']
    except Exception: pass   # st.secrets raises when no secrets file exists
    return os.environ.get('ANTHROPIC_API_KEY')


def stream_answer(question, dossier, passages, key, history=(), model=None):
    """Yields the answer text as it streams. history: earlier (question, answer) pairs about the same candidate.
    Server-side fallbacks are on, so a request a safety classifier declines is retried on a fallback model."""
    import anthropic
    client = anthropic.Anthropic(api_key=key, max_retries=2, timeout=90.0); msgs = []
    for q, a in history: msgs += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
    msgs.append({'role': 'user', 'content': build_prompt(question, dossier, passages)})
    with client.beta.messages.stream(model=model or DEFAULT_MODEL, max_tokens=4000, system=SYSTEM, messages=msgs, output_config={'effort': 'low'},
                                     betas=['server-side-fallback-2026-07-01'], fallbacks='default') as stream:
        for text in stream.text_stream: yield text
        if stream.get_final_message().stop_reason == 'refusal': yield '\n\n_The model declined to answer this question. Try rephrasing it._'


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
    if d['cons']: nerd.append('- Question marks: ' + '; '.join(d['cons']) + '.')
    show3 = d['top3'] and any(w in d['question'].lower() for w in ('top 3', 'top three', 'now', 'hindsight', 'where', 'value'))
    if d['hs'] and not (show3 and d['rank'] <= 3): nerd.append('- Hindsight: ' + d['hs'])
    if show3:
        nerd.append('- The top 3 today:\n' + '\n'.join('  - ' + x for x in d['top3']))
    return txt + '\n\n' + '\n'.join(nerd)
