# candidate_extractor.py
import re
from typing import List, Callable

def default_prompt() -> str:
    return (
        "Look at this clinical form page. "
        "List all visible headings, labels, and short text snippets that look like variable names "
        "or medication mentions (brands/abbreviations). "
        "Return a comma-separated list (no explanations)."
    )

def normalize_tokens(raw: str) -> List[str]:
    parts = re.split(r"[,;\n]+", raw or "")
    toks = []
    for p in parts:
        p = p.strip().strip("-•*").lower()
        if not p:
            continue
        p = re.sub(r"\s+", " ", p)
        toks.append(p)
    # dedupe preserving order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

class CandidateExtractor:
    """
    model_runner: Callable(image_path:str, prompt:str) -> str
    If None, uses a stub so you can smoke-test the notebook without a model.
    """
    def __init__(self, model_runner: Callable[[str, str], str] | None = None):
        if model_runner is None:
            def _stub(image_path: str, prompt: str) -> str:
                return "Symptoms, Duration, Tacroz 0.1% oint bd, Xyzal tab"
            model_runner = _stub
        self.model_runner = model_runner

    def extract_candidates(self, image_path: str) -> List[str]:
        raw = self.model_runner(image_path, default_prompt())
        return normalize_tokens(raw)

# Optional: helper to wire Gemma HF pipeline as a runner
def make_gemma_runner(pipe):
    """
    pipe: a Hugging Face 'image-text-to-text' pipeline for google/gemma-3-27b-it
    Returns a function(image_path:str, prompt:str)->str suitable for CandidateExtractor.
    """
    def _runner(image_path: str, prompt: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }]
        out = pipe(text=messages, max_new_tokens=256)
        # HF returns list; take last turn's content text
        return out[0]["generated_text"][-1]["content"]
    return _runner

