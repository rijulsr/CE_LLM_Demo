
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterable

@dataclass
class RAGPaths:
    base_dir: Path
    field_cards_jsonl: Path
    policy_dir: Path
    abbr_dir: Path
    range_dir: Path
    lexicon_dir: Path

    @classmethod
    def from_base(cls, base: str | Path) -> "RAGPaths":
        base = Path(base)
        return cls(
            base_dir=base,
            field_cards_jsonl=base / "field_cards.jsonl",
            policy_dir=base / "policy",
            abbr_dir=base / "abbr",
            range_dir=base / "range",
            lexicon_dir=base / "lexicon",
        )

@dataclass
class RAGStore:
    paths: RAGPaths
    fields_by_name: Dict[str, dict] = field(default_factory=dict)
    fields_by_synonym: Dict[str, List[str]] = field(default_factory=dict)
    policy: Dict[str, dict] = field(default_factory=dict)
    abbr: Dict[str, dict] = field(default_factory=dict)
    ranges: Dict[str, dict] = field(default_factory=dict)
    lexicons: Dict[str, dict] = field(default_factory=dict)

    def load(self) -> "RAGStore":
        self._load_field_cards(self.paths.field_cards_jsonl)
        self._load_dir(self.paths.policy_dir, into=self.policy)
        self._load_dir(self.paths.abbr_dir, into=self.abbr)
        self._load_dir(self.paths.range_dir, into=self.ranges)
        self._load_dir(self.paths.lexicon_dir, into=self.lexicons)
        return self

    def _load_field_cards(self, jsonl_path: Path) -> None:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Field cards JSONL not found: {jsonl_path}")
        self.fields_by_name.clear()
        self.fields_by_synonym.clear()
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                except Exception:
                    continue
                name = card.get("canonical_name")
                if not name:
                    continue
                self.fields_by_name[name] = card
                for syn in card.get("synonyms", []):
                    s = str(syn).strip().lower()
                    if not s:
                        continue
                    self.fields_by_synonym.setdefault(s, []).append(name)

    def _load_dir(self, d: Path, into: Dict[str, dict]) -> None:
        if not d.exists():
            return
        for p in d.glob("*.json"):
            try:
                obj = json.loads(p.read_text())
                key = obj.get("card_id", p.stem)
                into[key] = obj
            except Exception:
                pass

    def get_field_cards(self, names: Iterable[str]) -> List[dict]:
        return [self.fields_by_name[n] for n in names if n in self.fields_by_name]

    def search_fields_by_synonyms(self, cues: Iterable[str]) -> List[str]:
        hits = set()
        for c in cues:
            key = str(c).lower().strip()
            if key in self.fields_by_synonym:
                hits.update(self.fields_by_synonym[key])
        return list(hits)

    def slice_lexicon(self, lexicon_key: str, include_tokens: Optional[Iterable[str]] = None, top_k: int = 10) -> dict:
        card = self.lexicons.get(lexicon_key)
        if not card:
            return {}
        if not include_tokens:
            entries = card.get("entries", [])[:top_k]
        else:
            toks = {t.lower() for t in include_tokens}
            entries = []
            for e in card.get("entries", []):
                vs = [v.lower() for v in e.get("variants", [])]
                if any(v in toks for v in vs):
                    entries.append(e)
            if not entries:
                entries = card.get("entries", [])[:top_k]
        return {"card_id": card.get("card_id"), "entries": entries}

@dataclass
class ContextAssembler:
    store: RAGStore

    def build_context(
        self,
        target_fields: List[str],
        page_tokens: Optional[List[str]] = None,
        include_abbr: bool = True,
        include_policies: bool = True,
        include_ranges: bool = True,
        meds_lexicon_key: str = "lexicon/meds_observed:v1",
        meds_top_k: int = 12
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"field_cards": [], "policies": {}, "abbr": {}, "ranges": {}, "lexicon": {}}
        payload["field_cards"] = self.store.get_field_cards(target_fields)
        if include_policies:
            for k, v in self.store.policy.items():
                payload["policies"][k] = v
        if include_abbr:
            for k, v in self.store.abbr.items():
                payload["abbr"][k] = v
        if include_ranges:
            for k, v in self.store.ranges.items():
                payload["ranges"][k] = v
        if any(f in {"treatment_history", "followup_visits", "treatment_followup_notes"} for f in target_fields):
            payload["lexicon"][meds_lexicon_key] = self.store.slice_lexicon(
                meds_lexicon_key, include_tokens=page_tokens, top_k=meds_top_k
            )
        return payload

    def to_prompt_chunks(self, context_payload: Dict[str, Any]) -> List[str]:
        chunks: List[str] = []
        if context_payload.get("field_cards"):
            fc = {"field_cards": [
                {
                    "canonical_name": c.get("canonical_name"),
                    "type": c.get("type"),
                    "synonyms": c.get("synonyms", [])[:8],
                    "cuewords": c.get("cuewords", [])[:12],
                    "patterns": c.get("patterns", [])[:2],
                    "normalization": c.get("normalization", {}),
                    "ranges": c.get("ranges", {})
                } for c in context_payload["field_cards"]
            ]}
            chunks.append(json.dumps(fc, ensure_ascii=False))
        pol = context_payload.get("policies", {})
        small_pol = {}
        for key in pol:
            if "notation" in key or "units" in key or "date" in key:
                small_pol[key] = pol[key]
        if small_pol:
            chunks.append(json.dumps({"policies": small_pol}, ensure_ascii=False))
        abbr = context_payload.get("abbr", {})
        if abbr:
            for k, v in abbr.items():
                if "dermatology" in k:
                    chunks.append(json.dumps({"abbr": v}, ensure_ascii=False))
        rng = context_payload.get("ranges", {})
        if rng:
            keep = {}
            for k, v in rng.items():
                if any(tag in k for tag in ["labs", "scorad", "anthro"]):
                    keep[k] = v
            if keep:
                chunks.append(json.dumps({"ranges": keep}, ensure_ascii=False))
        lex = context_payload.get("lexicon", {})
        if lex:
            for k, v in lex.items():
                short_entries = [{"canonical": e.get("canonical"), "variants": e.get("variants", [])} for e in v.get("entries", [])]
                chunks.append(json.dumps({"lexicon_key": k, "entries": short_entries}, ensure_ascii=False))
        return chunks

SECTION_MAP = {
    "history": ["duration","site_of_onset","mode_of_spread","symptoms","treatment_history",
                "personal_history","birth_term","birth_weight","socio_economic_status",
                "household_members","family_history","past_history","milestones","vaccination_status"],
    "scorad": ["extent_bsa","intensity_erythema","intensity_edema_papulation","intensity_excoriations",
               "intensity_oozing_crusting","intensity_dryness","intensity_lichenification",
               "intensity_total_b","itchiness_vas","sleeplessness_vas","scorad_final"],
    "investigations": ["hemoglobin","tlc","dlc","esr","platelets","sodium","potassium","urea","creatinine",
                       "ast","alt","serum_bilirubin","serum_proteins","serum_ige","fbs",
                       "ana","urine_re_me","chest_xray","ecg","echo","biopsy_histopathology","immunofluorescence","ultrasound","mri","other_investigations"],
    "followups": ["followup_visits","treatment_followup_notes"]
}

def fields_for_section(section: str) -> List[str]:
    return SECTION_MAP.get(section, [])
