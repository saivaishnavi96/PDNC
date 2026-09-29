import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    page_icon="🎯",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "./trained_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


# ============================================================
# LABELS
# ============================================================

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ============================================================
# SENTIMENT WORDS
# ============================================================

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome",
    "wonderful", "fantastic", "best", "love", "loved",
    "like", "liked", "nice", "perfect", "happy",
    "helpful", "friendly", "comfortable", "fast", "quick",
    "beautiful", "clean", "smooth", "easy", "enjoyable",
    "satisfied", "brilliant", "superb"
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "horrible",
    "worst", "hate", "hated", "slow", "dirty", "broken",
    "difficult", "frustrating", "frustrated", "disappointing",
    "disappointed", "expensive", "late", "uncomfortable",
    "rude", "useless", "weak", "problem", "problems",
    "issue", "issues", "dislike", "disliked", "failed",
    "failure"
}


# ============================================================
# ASPECT RELEVANCE CHECK
# ============================================================

def normalize_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())


def aspect_is_relevant(review, aspect):
    """
    Checks whether the selected aspect is actually discussed
    in the review.

    Returns:
        True  -> aspect appears / is clearly referenced
        False -> aspect is not present
    """

    review_clean = normalize_text(review)
    aspect_clean = normalize_text(aspect)

    if not aspect_clean.strip():
        return False

    # Direct aspect match
    if aspect_clean in review_clean:
        return True

    # Check individual aspect words
    aspect_words = aspect_clean.split()

    if not aspect_words:
        return False

    # For multi-word aspects, require at least one meaningful
    # aspect word to occur.
    meaningful_words = [
        word for word in aspect_words
        if len(word) > 2
    ]

    for word in meaningful_words:
        if re.search(r"\b" + re.escape(word) + r"\b", review_clean):
            return True

    return False


# ============================================================
# GET ASPECT-FOCUSED CLAUSE
# ============================================================

def get_aspect_clause(review, aspect):

    sentences = re.split(r"[.!?]+", review)

    for sentence in sentences:

        if aspect.lower() in sentence.lower():

            clauses = re.split(
                r"\b(?:but|however|although|though|while|whereas|yet|and)\b",
                sentence,
                flags=re.IGNORECASE
            )

            for clause in clauses:

                if aspect.lower() in clause.lower():
                    return clause.strip()

            return sentence.strip()

    return review


# ============================================================
# RULE-BASED SENTIMENT
# ============================================================

def rule_sentiment(text):

    words = normalize_text(text).split()

    positive_score = 0
    negative_score = 0

    negations = {
        "not",
        "never",
        "no",
        "isn't",
        "wasn't",
        "doesn't",
        "don't",
        "didn't"
    }

    for i, word in enumerate(words):

        previous_words = words[max(0, i - 3):i]

        negated = any(
            negation in previous_words
            for negation in negations
        )

        if word in POSITIVE_WORDS:

            if negated:
                negative_score += 1
            else:
                positive_score += 1

        elif word in NEGATIVE_WORDS:

            if negated:
                positive_score += 1
            else:
                negative_score += 1

    if positive_score > negative_score:
        return "Positive"

    if negative_score > positive_score:
        return "Negative"

    return None


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_predict(aspect, context):

    encoded = tokenizer(
        aspect,
        context,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_class = torch.argmax(probabilities).item()

    confidence = probabilities[predicted_class].item()

    return (
        LABELS[predicted_class],
        confidence,
        probabilities
    )


# ============================================================
# MAIN SENTIMENT FUNCTION
# ============================================================

def analyze_sentiment(review, aspect):

    # --------------------------------------------------------
    # STEP 1: RELEVANCE CHECK
    # --------------------------------------------------------

    if not aspect_is_relevant(review, aspect):

        return {
            "sentiment": "Not Enough Information",
            "confidence": None,
            "method": "Aspect Relevance Check",
            "context": review
        }


    # --------------------------------------------------------
    # STEP 2: GET ASPECT-SPECIFIC CLAUSE
    # --------------------------------------------------------

    aspect_clause = get_aspect_clause(
        review,
        aspect
    )


    # --------------------------------------------------------
    # STEP 3: RULE-BASED CHECK
    # --------------------------------------------------------

    rule_result = rule_sentiment(
        aspect_clause
    )


    if rule_result is not None:

        return {
            "sentiment": rule_result,
            "confidence": 0.90,
            "method": "Aspect-Focused Rule + Sentiment",
            "context": aspect_clause
        }


    # --------------------------------------------------------
    # STEP 4: DISTILBERT
    # --------------------------------------------------------

    sentiment, confidence, probabilities = model_predict(
        aspect,
        aspect_clause
    )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "method": "DistilBERT",
        "context": aspect_clause,
        "probabilities": probabilities
    }


# ============================================================
# UI
# ============================================================

st.title("🎯 Aspect-Based Sentiment Analysis")

st.write(
    "Enter a review and select the aspect you want to analyze."
)

st.info(
    "The system first checks whether the review contains "
    "information about the selected aspect. If not, it returns "
    "'Not Enough Information'."
)


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

review = st.text_area(
    "📝 Review",
    height=150,
    placeholder=(
        "Example: The food was excellent but the service was slow."
    )
)

aspect = st.text_input(
    "🎯 Aspect",
    placeholder="Example: service"
)


# ------------------------------------------------------------
# ANALYZE BUTTON
# ------------------------------------------------------------

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not review.strip():

        st.warning("Please enter a review.")

    elif not aspect.strip():

        st.warning("Please enter an aspect.")

    else:

        result = analyze_sentiment(
            review,
            aspect
        )

        sentiment = result["sentiment"]


        # ====================================================
        # NOT ENOUGH INFORMATION
        # ====================================================

        if sentiment == "Not Enough Information":

            st.warning(
                "ℹ️ Not Enough Information"
            )

            st.write(
                f"The review does not contain enough information "
                f"about the aspect **'{aspect}'**."
            )


        # ====================================================
        # POSITIVE
        # ====================================================

        elif sentiment == "Positive":

            st.success(
                "😊 Positive"
            )

            if result["confidence"] is not None:

                st.write(
                    f"Confidence: "
                    f"{result['confidence'] * 100:.2f}%"
                )


        # ====================================================
        # NEGATIVE
        # ====================================================

        elif sentiment == "Negative":

            st.error(
                "😞 Negative"
            )

            if result["confidence"] is not None:

                st.write(
                    f"Confidence: "
                    f"{result['confidence'] * 100:.2f}%"
                )


        # ====================================================
        # NEUTRAL
        # ====================================================

        elif sentiment == "Neutral":

            st.info(
                "😐 Neutral"
            )

            if result["confidence"] is not None:

                st.write(
                    f"Confidence: "
                    f"{result['confidence'] * 100:.2f}%"
                )


        # ====================================================
        # ASPECT CONTEXT
        # ====================================================

        st.subheader("Aspect-Focused Context")

        st.write(
            result["context"]
        )

        st.caption(
            f"Method: {result['method']}"
        )


# ============================================================
# EXAMPLES
# ============================================================

st.divider()

st.subheader("💡 Try These Examples")

st.write(
    "**Positive:** The food is excellent and fresh."
)

st.write(
    "**Negative:** The service was extremely slow."
)

st.write(
    "**Conflicting:** The food is good but the service is slow."
)

st.write(
    "**Not Enough Information:** "
    "The hotel has a swimming pool and a large parking area."
)

st.caption(
    "For the last example, selecting 'breakfast' as the aspect "
    "should return Not Enough Information."
)