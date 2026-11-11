
import sys
from pathlib import Path

import streamlit as st
import pandas as pd


# --------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ClassWrite | NLP Sentence Generator",
    page_icon="✍️",
    layout="wide"
)


# --------------------------------------------------
# 2. PROJECT PATH CONFIGURATION
# --------------------------------------------------

# app.py is located directly inside ClassWrite_Deploy.
PROJECT_ROOT = Path(__file__).resolve().parent

# Make the main project folder accessible to Python.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Path to the NLP corpus.
CORPUS_PATH = PROJECT_ROOT / "shared_engine" / "corpus.txt"


# --------------------------------------------------
# 3. IMPORT THE SHARED NLP ENGINE
# --------------------------------------------------

from shared_engine.model import (
    ClassBasedLanguageModel,
    tokenize,
    get_word_class
)


# --------------------------------------------------
# 4. LOAD AND TRAIN THE NLP MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    # Check whether the corpus file exists.
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(
            f"Corpus file not found at: {CORPUS_PATH}"
        )

    # Read the corpus.
    corpus = CORPUS_PATH.read_text(encoding="utf-8")

    # Initialize and train the model.
    model = ClassBasedLanguageModel()
    model.train(corpus)

    return model


# Load the trained model.
model = load_model()


# --------------------------------------------------
# 5. WEBSITE TITLE AND INTRODUCTION
# --------------------------------------------------

st.title("✍️ ClassWrite")

st.header("Class-Based Sentence Generator")

st.write(
    "Generate sample sentences using learned word classes, "
    "class-transition patterns, and word-emission frequencies."
)


# --------------------------------------------------
# 6. SIDEBAR - PROJECT INFORMATION
# --------------------------------------------------

with st.sidebar:

    st.header("About this project")

    st.write("**Topic:** Class-Based Language Models")

    st.write("**Course:** Natural Language Processing")

    st.write("**Model:** Class transitions + word emissions")

    st.info(
        "This is a small educational model trained on a sample corpus. "
        "Generated sentences may not always be grammatically correct."
    )


# --------------------------------------------------
# 7. SENTENCE GENERATION SETTINGS
# --------------------------------------------------

st.subheader("Generate a sentence")

col1, col2 = st.columns(2)


with col1:

    max_words = st.slider(
        "Maximum number of words",
        min_value=4,
        max_value=15,
        value=8
    )


with col2:

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=10000,
        value=42,
        step=1,
        help="Change the seed to explore different generated sentences."
    )


# --------------------------------------------------
# 8. GENERATE SENTENCE
# --------------------------------------------------

if st.button("✨ Generate sentence", type="primary"):

    sentence = model.generate_sentence(
        max_words=max_words,
        seed=int(seed)
    )

    st.session_state["generated_sentence"] = sentence


# --------------------------------------------------
# 9. DISPLAY GENERATED OUTPUT
# --------------------------------------------------

if "generated_sentence" in st.session_state:

    sentence = st.session_state["generated_sentence"]

    st.subheader("Generated output")

    st.success(sentence)


    # --------------------------------------------------
    # 10. WORD-CLASS ANALYSIS
    # --------------------------------------------------

    words = tokenize(sentence)

    class_rows = [
        {
            "Word": word,
            "Word class": get_word_class(word)
        }
        for word in words
    ]

    st.subheader("Word-class analysis")


    if class_rows:

        df = pd.DataFrame(class_rows)

        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )


        # --------------------------------------------------
        # 11. CLASS DISTRIBUTION CHART
        # --------------------------------------------------

        st.subheader("Class distribution")

        class_counts = df["Word class"].value_counts()

        st.bar_chart(class_counts)


        # --------------------------------------------------
        # 12. SENTENCE STATISTICS
        # --------------------------------------------------

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                "Words generated",
                len(words)
            )

        with metric_col2:

            st.metric(
                "Distinct word classes",
                df["Word class"].nunique()
            )


    # Educational note.
    st.caption(
        "The generated output is sampled from a simplified class-based "
        "language model and is intended for educational demonstration."
    )


# --------------------------------------------------
# 13. EXPLANATION OF THE NLP MODEL
# --------------------------------------------------

st.divider()

st.subheader("How does it work?")

st.markdown(
    """
    1. **Training:** The model reads the sample text corpus.

    2. **Word classification:** Each word is assigned a simplified class,
       such as NOUN, VERB, or DETERMINER.

    3. **Class transitions:** The model learns which classes tend to follow
       other classes.

    4. **Word emissions:** A word is sampled from the selected class.

    5. **Sentence output:** The selected words are combined into a sentence.
    """
)


# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------

st.caption(
    "ClassWrite | Educational NLP project | Class-Based Language Models"
)
