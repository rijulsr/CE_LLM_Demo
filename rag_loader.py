
from pathlib import Path
from rag_store import RAGPaths, RAGStore, ContextAssembler, fields_for_section

CARDS_DIR = Path("/home/rijul/Gitlaboratory/Context_Engineering_LLM/cards")
paths = RAGPaths.from_base(CARDS_DIR)
store = RAGStore(RAGPaths.from_base(CARDS_DIR)).load()
print("Fields loaded:", len(store.fields_by_name))
print("Policies:", list(store.policy.keys()))
print("Abbr:", list(store.abbr.keys()))
print("Ranges:", list(store.ranges.keys()))
print("Lexicons:", list(store.lexicons.keys()))


target_fields = fields_for_section("history")
page_tokens = ["symptoms", "duration", "tacroz", "xyzal"]

ctx = ContextAssembler(store).build_context(target_fields, page_tokens=page_tokens)
chunks = ContextAssembler(store).to_prompt_chunks(ctx)

print("=== Prompt chunks (send before the image) ===")
for i, ch in enumerate(chunks, 1):
    print(f"[Chunk {i}] {ch[:240]}...")
