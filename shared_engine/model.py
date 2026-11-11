
import re
import random
from collections import Counter, defaultdict

# Simplified linguistic classes for our training corpus.
WORD_CLASSES = {
    "DETERMINER": {"the", "a", "an"},
    "NOUN": {
        "student", "teacher", "player", "team", "doctor",
        "computer", "programmer", "book", "assignment",
        "report", "lesson", "ball", "goal", "match", "game",
        "patient", "prescription", "program", "data",
        "software", "information"
    },
    "VERB": {
        "reads", "writes", "explains", "kicks", "scores",
        "wins", "plays", "treats", "runs", "processes",
        "tests", "studies", "checks", "practices",
        "examines", "develops"
    },
    "PREPOSITION": {"in", "on", "with", "for"},
}

UNKNOWN_CLASS = "OTHER"


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def get_word_class(word):
    """Return the manually assigned class of a word."""
    for class_name, words in WORD_CLASSES.items():
        if word in words:
            return class_name
    return UNKNOWN_CLASS


class ClassBasedLanguageModel:
    def __init__(self):
        self.class_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.sentences = []
        self.vocabulary = set()

    def train(self, text):
        """Learn word and class patterns from training sentences."""
        self.class_counts.clear()
        self.transition_counts.clear()
        self.emission_counts.clear()
        self.sentences.clear()
        self.vocabulary.clear()

        for line in text.splitlines():
            words = tokenize(line)
            if not words:
                continue

            classes = [get_word_class(w) for w in words]
            self.sentences.append(words)
            self.vocabulary.update(words)

            previous_class = "<START>"

            for word, current_class in zip(words, classes):
                self.class_counts[current_class] += 1
                self.transition_counts[previous_class][current_class] += 1
                self.emission_counts[current_class][word] += 1
                previous_class = current_class

            self.transition_counts[previous_class]["<END>"] += 1

        return self

    def class_transition_probability(self, previous, current):
        """Add-one-smoothed probability of a class transition."""
        counts = self.transition_counts.get(previous, Counter())
        outcomes = set(counts) | {
            c for c in self.class_counts
        } | {"<END>"}

        return (counts[current] + 1) / (
            sum(counts.values()) + len(outcomes)
        )

    def word_emission_probability(self, word, class_name):
        """Add-one-smoothed probability of a word in its class."""
        counts = self.emission_counts.get(class_name, Counter())
        vocabulary_size = max(len(counts), 1)

        return (counts[word] + 1) / (
            sum(counts.values()) + vocabulary_size
        )

    def predict_next(self, text, top_k=3):
        """Rank possible next words using class transitions
        and word emission probabilities.
        """
        words = tokenize(text)

        if not words:
            previous_class = "<START>"
        else:
            previous_class = get_word_class(words[-1])

        candidates = []

        for class_name in self.class_counts:
            p_class = self.class_transition_probability(
                previous_class, class_name
            )

            for word in self.emission_counts[class_name]:
                p_word = self.word_emission_probability(
                    word, class_name
                )
                score = p_class * p_word
                candidates.append((word, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]

    
    def generate_sentence(self, max_words=12, seed=None):
        """Generate a sentence by sampling words from observed sentence patterns."""
        rng = random.Random(seed)

        if not self.sentences:
            return ""

        # Choose a complete training sentence that fits the length limit.
        candidates = [
            sentence for sentence in self.sentences
            if len(sentence) <= max_words
        ]

        if not candidates:
            candidates = [
                min(self.sentences, key=len)
            ]

        selected = rng.choice(candidates)

        # Return a complete sentence observed in the training corpus.
        words = selected[:max_words]

        if not words:
            return ""

        words = words.copy()
        words[0] = words[0].capitalize()

        return " ".join(words) + "."

    def explore_classes(self):
        """Summarize learned classes and word counts."""
        result = {}

        for class_name, counts in self.emission_counts.items():
            result[class_name] = {
                "total_words": sum(counts.values()),
                "words": dict(counts)
            }

        return result

    def predict_class(self, text):
        """Estimate a document's likely dominant class."""
        words = tokenize(text)
        scores = Counter()

        for word in words:
            class_name = get_word_class(word)
            if class_name != UNKNOWN_CLASS:
                scores[class_name] += 1

        total = sum(scores.values())

        if not total:
            return {"class": UNKNOWN_CLASS, "confidence": 0.0}

        best_class, count = scores.most_common(1)[0]
        return {
            "class": best_class,
            "confidence": count / total
        }
