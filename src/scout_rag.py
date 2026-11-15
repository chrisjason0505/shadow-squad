"""'Why this player?': a small retrieval-augmented explainer for the Streamlit app.

Retrieval: the method notes in app/knowledge/*.md are split by section and indexed with TF-IDF (scikit-learn, no
embedding service). A question is matched against them together with the names of the features that moved this
candidate most, so the passages that come back explain the numbers in the dossier.
Generation: the dossier (every number the engine has for this candidate and this query) and the retrieved passages go
to Gemini (a Google AI Studio API key, the same key Google Antigravity uses), which answers in plain words first and then with the statistics. Without an API key the app falls back to
`offline_answer`, a template built from the same dossier, so the feature never breaks on hosting."""
import glob, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = 'gemini-3.8-flash'   # override with GEMINI_MODEL in the app secrets
SYSTEM = """You are the chief scout behind Shadow Squad, a tool that ranks replacements for a player leaving a European club.
You talk to football people: fans, analysts and coaches who know the game better than they know statistics.

How to answer:
- Speak football. Talk about roles, profiles, what a player does on the pitch (progressive carries, passes into the box, winning the ball back, threat in the box, aerial duels), the club's style, price bracket, age and career stage, and what the team would keep, lose or gain.
- Translate the numbers. Say "top 10% of wingers for progressive carries", not "z = +1.4"; say "the model rates his price bracket", not "value_gap SHAP". Only use SHAP, cosine similarity or z-scores if the user asks about the model or the statistics.
- Make the case. Say why he is (or is not) the best replacement on the list, and if he is not No. 1, what No. 1 has over him.
- Use hindsight when the dossier has it: where the candidate and the top 3 are now, and how their Transfermarkt value moved, with the as-of date. Say whether the list looks good in hindsight, honestly.
- Use only the dossier and the notes. Never invent stats, fees, injuries, wages, quotes, news or current clubs. If something is not in the data, say so. Values are Transfermarkt estimates, not fees.
- Read numbers in their team context. The dossier says how each team played (possession, pressing, xG against) and gives style reads: a centre-back with few tackles in a dominant side with a tight defence is positionally sound, not passive; a creator who keeps taking risks in a deep side carries that team; a forward who presses hard in a high-intensity side and still beats men is rare. Prefer the style-adjusted rating to the raw one and say why they differ. Say "far more than his team's style predicts", not standard deviations.
- Be honest about doubt: a wide rank range, a step down in an area that mattered, or that the engine finds the real signing in its top 10 only about a third of the time.

Format, unless the user asks for something else: a short verdict paragraph (two to four sentences), then a section headed "For the football nerds" with four to six bullet points on the like-for-like profile, the style read, the part of the game where he is a step up or down, the fit with the club's style, price and age, and hindsight if available. Under 300 words."""


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
    """The Gemini key (Google AI Studio, the same key Antigravity uses) from Streamlit secrets first, then the environment."""
    for k in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        try:
            if secrets is not None and k in secrets: return secrets[k]
        except Exception: pass   # st.secrets raises when no secrets file exists
        if os.environ.get(k): return os.environ[k]
    return None


def _pick_model(client):
    """When the configured model is retired, fall back to the newest Flash model the key can use."""
    names = [m.name.split('/')[-1] for m in client.models.list() if 'generateContent' in (getattr(m, 'supported_actions', None) or [])]
    flash = sorted((n for n in names if 'flash' in n and not any(x in n for x in ('image', 'tts', 'live', 'lite', 'audio'))), reverse=True)
    return (flash or names or [DEFAULT_MODEL])[0]


def stream_answer(question, dossier, passages, key, history=(), model=None):
    """Yields the answer text as it streams from Gemini. history: earlier (question, answer) pairs about the same candidate.
    A model name the key cannot use (404) falls back to the newest Flash model on the account."""
    import itertools
    from google import genai
    from google.genai import types, errors
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=90_000))
    contents = []
    for q, a in history: contents += [types.Content(role='user', parts=[types.Part(text=q)]), types.Content(role='model', parts=[types.Part(text=a)])]
    contents.append(types.Content(role='user', parts=[types.Part(text=build_prompt(question, dossier, passages))]))
    cfg = types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=4000, temperature=0.4)
    def start(name):
        it = iter(client.models.generate_content_stream(model=name, contents=contents, config=cfg)); return itertools.chain([next(it)], it)
    try: chunks = start(model or DEFAULT_MODEL)
    except errors.ClientError as e:
        if getattr(e, 'code', None) != 404: raise
        chunks = start(_pick_model(client))
    blocked = False
    for chunk in chunks:
        if chunk.text: yield chunk.text
        fr = chunk.candidates[0].finish_reason if chunk.candidates else None
        if fr is not None and str(fr).split('.')[-1] in ('SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'RECITATION'): blocked = True
    if blocked: yield '\n\n_The model stopped this answer early. Try rephrasing the question._'


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
