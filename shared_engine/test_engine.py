
from pathlib import Path
from model import ClassBasedLanguageModel

folder = Path(__file__).parent
corpus = (folder / "corpus.txt").read_text(encoding="utf-8")

model = ClassBasedLanguageModel()
model.train(corpus)

print("\n--- NEXT WORD PREDICTIONS ---")
print(model.predict_next("the student", top_k=3))

print("\n--- GENERATED SENTENCE ---")
print(model.generate_sentence(seed=42))

print("\n--- LEARNED WORD CLASSES ---")
for class_name, info in model.explore_classes().items():
    print(class_name, info)

print("\n--- WORD CLASS ANALYSIS ---")
print(model.predict_class("the player scores a goal"))
