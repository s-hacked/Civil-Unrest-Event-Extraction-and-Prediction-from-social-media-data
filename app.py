import os
import re
import json
import warnings

import numpy as np
import pandas as pd
import torch
import gradio as gr
import joblib
import tensorflow as tf

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
)

from dateparser.search import search_dates

warnings.filterwarnings("ignore")


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

BILINGUAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "xlm_roberta_bilingual_event_detector"
)

NER_MODEL_NAME = "Davlan/xlm-roberta-base-ner-hrl"


# ============================================================
# LOAD EVENT DETECTION MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING XLM-R EVENT DETECTION MODEL")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

event_tokenizer = AutoTokenizer.from_pretrained(
    BILINGUAL_MODEL_PATH
)

event_model = AutoModelForSequenceClassification.from_pretrained(
    BILINGUAL_MODEL_PATH
)

event_model.to(device)
event_model.eval()

print("Event detector loaded successfully.")


# ============================================================
# LOAD MULTILINGUAL NER MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING MULTILINGUAL NER MODEL")
print("=" * 60)

ner_pipeline = pipeline(
    "ner",
    model=NER_MODEL_NAME,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

print("Multilingual NER loaded successfully.")


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

RF_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "random_forest_temporal_model.joblib"
)

RF_FEATURES_PATH = os.path.join(
    MODELS_DIR,
    "rf_features.json"
)

rf_model = None
FEATURES = []


if os.path.exists(RF_MODEL_PATH):

    try:
        rf_model = joblib.load(RF_MODEL_PATH)

        if os.path.exists(RF_FEATURES_PATH):

            with open(RF_FEATURES_PATH, "r", encoding="utf-8") as f:
                FEATURES = json.load(f)

        print("\nRandom Forest temporal model loaded.")

    except Exception as e:

        print("\nWarning: Could not load Random Forest:")
        print(e)

else:

    print("\nWarning: Random Forest model not found.")


# ============================================================
# LOAD GDELT DATA
# ============================================================

GDELT_PATH = os.path.join(
    DATA_DIR,
    "gdelt_daily_protests.csv"
)

gdelt_df = pd.DataFrame()


def load_gdelt_data():

    global gdelt_df

    if not os.path.exists(GDELT_PATH):

        print("\nWarning: GDELT CSV not found:")
        print(GDELT_PATH)

        return

    try:

        df = pd.read_csv(GDELT_PATH)

        print("\n" + "=" * 60)
        print("GDELT DATA")
        print("=" * 60)

        print("Columns found:")
        print(list(df.columns))

        # ----------------------------------------------------
        # FIND DATE COLUMN
        # ----------------------------------------------------

        date_candidates = [
            "date",
            "Date",
            "event_date",
            "EventDate",
            "SQLDATE",
            "sqldate",
            "DATE"
        ]

        date_column = None

        for col in date_candidates:

            if col in df.columns:
                date_column = col
                break

        if date_column is None:

            # Try to identify a date-like column automatically
            for col in df.columns:

                try:

                    converted = pd.to_datetime(
                        df[col],
                        errors="coerce"
                    )

                    if converted.notna().sum() > 0:

                        date_column = col
                        break

                except Exception:
                    continue

        if date_column is None:

            print("ERROR: No date column found in GDELT CSV.")
            return

        # ----------------------------------------------------
        # FIND PROTEST COUNT COLUMN
        # ----------------------------------------------------

        count_candidates = [
            "protest_count",
            "Protest_Count",
            "protest_events",
            "event_count",
            "count",
            "Count",
            "events",
            "Events"
        ]

        count_column = None

        for col in count_candidates:

            if col in df.columns:

                count_column = col
                break

        if count_column is None:

            # Automatically find numeric column
            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_columns:

                # Prefer a column that looks like a count
                for col in numeric_columns:

                    col_lower = str(col).lower()

                    if (
                        "count" in col_lower
                        or "event" in col_lower
                        or "protest" in col_lower
                    ):

                        count_column = col
                        break

                if count_column is None:
                    count_column = numeric_columns[-1]

        if count_column is None:

            print("ERROR: No protest-count column found.")
            return

        print("Detected date column:", date_column)
        print("Detected count column:", count_column)

        # ----------------------------------------------------
        # STANDARDIZE
        # ----------------------------------------------------

        df = df.rename(
            columns={
                date_column: "date",
                count_column: "protest_count"
            }
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df["protest_count"] = pd.to_numeric(
            df["protest_count"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["date", "protest_count"]
        )

        df = df.sort_values("date")

        df = df[
            [
                "date",
                "protest_count"
            ]
        ]

        df = df.reset_index(drop=True)

        gdelt_df = df

        print("\nGDELT loaded successfully.")

        print("Shape:", gdelt_df.shape)

        if len(gdelt_df) > 0:

            print(
                "Date range:",
                gdelt_df["date"].min().date(),
                "to",
                gdelt_df["date"].max().date()
            )

            print(
                "Total protest events:",
                int(gdelt_df["protest_count"].sum())
            )

    except Exception as e:

        print("\nERROR loading GDELT:")
        print(e)


load_gdelt_data()


# ============================================================
# BILINGUAL EVENT TRIGGERS
# ============================================================

EVENT_TRIGGERS = {

    "protest": [
        "protest",
        "protested",
        "protesting",
        "demonstration",
        "demonstrate",
        "demonstrated",
        "march",
        "marching",
        "rally",
        "rallied",
        "sit-in",
        "sit in",
        "विरोध",
        "विरोध प्रदर्शन",
        "प्रदर्शन",
        "प्रदर्शन किया",
        "प्रदर्शन करेंगे",
        "धरना",
        "रैली",
        "मार्च"
    ],

    "strike": [
        "strike",
        "strikes",
        "striking",
        "struck",
        "हड़ताल",
        "हड़ताल कर",
        "हड़ताल किया",
        "हड़ताल करेंगे"
    ],

    "clash": [
        "clash",
        "clashes",
        "clashed",
        "fighting",
        "fight",
        "झड़प",
        "संघर्ष",
        "मारपीट"
    ],

    "riot": [
        "riot",
        "riots",
        "rioting",
        "दंगा",
        "दंगे",
        "दंगों"
    ],

    "unrest": [
        "unrest",
        "civil unrest",
        "अशांति",
        "उपद्रव",
        "अराजकता"
    ],

    "attack": [
        "attack",
        "attacked",
        "attacking",
        "हमला",
        "हमला किया",
        "हमले"
    ],

    "arrest": [
        "arrest",
        "arrested",
        "detained",
        "गिरफ्तार",
        "गिरफ्तारी",
        "हिरासत"
    ],

    "abduction": [
        "abduct",
        "abducted",
        "kidnap",
        "kidnapped",
        "अपहरण"
    ],

    "suppression": [
        "suppression",
        "suppressed",
        "crackdown",
        "दमन",
        "कार्रवाई",
        "छापेमारी"
    ]
}


# ============================================================
# EVENT TYPE + TRIGGER
# ============================================================

def detect_event_type_trigger(text):

    text_lower = text.lower()

    candidates = []

    for event_type, triggers in EVENT_TRIGGERS.items():

        for trigger in triggers:

            if trigger.lower() in text_lower:

                candidates.append(
                    (
                        len(trigger),
                        event_type,
                        trigger
                    )
                )

    if not candidates:

        return {
            "event_type": None,
            "trigger": None
        }

    candidates.sort(
        reverse=True
    )

    _, event_type, trigger = candidates[0]

    return {
        "event_type": event_type,
        "trigger": trigger
    }


# ============================================================
# NER
# ============================================================

def run_ner(text):

    try:

        entities = ner_pipeline(text)

        return entities

    except Exception:

        return []


# ============================================================
# ACTOR EXTRACTION
# ============================================================

def extract_actor(text):

    entities = run_ner(text)

    actor_candidates = []

    for entity in entities:

        label = entity.get(
            "entity_group",
            ""
        ).upper()

        word = entity.get(
            "word",
            ""
        ).strip()

        if label in [
            "PER",
            "PERSON",
            "ORG",
            "ORGANIZATION"
        ]:

            if word:

                actor_candidates.append(word)

    if actor_candidates:

        return actor_candidates[0]

    # English fallback
    english_patterns = [
        r"\b(thousands of students)\b",
        r"\b(students)\b",
        r"\b(workers)\b",
        r"\b(farmers)\b",
        r"\b(residents)\b",
        r"\b(activists)\b",
        r"\b(protesters)\b",
        r"\b(teachers)\b",
        r"\b(unions?)\b"
    ]

    # Hindi fallback
    hindi_patterns = [
        r"(हजारों छात्र)",
        r"(छात्र)",
        r"(मजदूर)",
        r"(किसान)",
        r"(निवासी)",
        r"(कार्यकर्ता)",
        r"(प्रदर्शनकारी)",
        r"(शिक्षक)"
    ]

    for pattern in english_patterns + hindi_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            return match.group(1)

    return None


# ============================================================
# LOCATION EXTRACTION
# ============================================================

def extract_location(text):

    entities = run_ner(text)

    locations = []

    for entity in entities:

        label = entity.get(
            "entity_group",
            ""
        ).upper()

        word = entity.get(
            "word",
            ""
        ).strip()

        if label in [
            "LOC",
            "LOCATION",
            "GPE",
            "FAC"
        ]:

            if word:

                locations.append(word)

    if locations:

        return locations[0]

    # Common location fallback
    locations = [
        "Delhi",
        "Mumbai",
        "Lucknow",
        "Bengaluru",
        "Bangalore",
        "Kolkata",
        "Chennai",
        "Hyderabad",
        "Pune",
        "Ahmedabad",
        "Noida",
        "Gurgaon",
        "New Delhi",

        "दिल्ली",
        "मुंबई",
        "लखनऊ",
        "बेंगलुरु",
        "कोलकाता",
        "चेन्नई",
        "हैदराबाद",
        "पुणे",
        "अहमदाबाद"
    ]

    for location in locations:

        if location.lower() in text.lower():

            return location

    return None


# ============================================================
# TIME EXTRACTION
# ============================================================

HINDI_TIME_PATTERNS = [

    r"आज",
    r"कल",
    r"कल रात",
    r"कल सुबह",
    r"कल शाम",
    r"कल दोपहर",

    r"आज रात",
    r"आज सुबह",
    r"आज शाम",
    r"आज दोपहर",

    r"पिछले दिन",
    r"पिछले सप्ताह",
    r"पिछले हफ्ते",
    r"पिछले महीने",

    r"अगले दिन",
    r"अगले सप्ताह",
    r"अगले हफ्ते",
    r"अगले महीने",

    r"कुछ दिन पहले",
    r"कुछ सप्ताह पहले",
    r"कुछ महीने पहले",

    r"पिछले साल",
    r"अगले साल",

    r"सोमवार",
    r"मंगलवार",
    r"बुधवार",
    r"गुरुवार",
    r"शुक्रवार",
    r"शनिवार",
    r"रविवार",

    r"\d+\s*दिन पहले",
    r"\d+\s*दिन बाद",

    r"\d+\s*सप्ताह पहले",
    r"\d+\s*सप्ताह बाद",

    r"\d+\s*महीने पहले",
    r"\d+\s*महीने बाद"
]


ENGLISH_TIME_PATTERNS = [

    r"\btoday\b",
    r"\byesterday\b",
    r"\btomorrow\b",
    r"\blast night\b",

    r"\bthis morning\b",
    r"\bthis evening\b",

    r"\blast week\b",
    r"\bnext week\b",

    r"\blast month\b",
    r"\bnext month\b",

    r"\blast year\b",
    r"\bnext year\b",

    r"\b\d+\s+days?\s+ago\b",
    r"\bin\s+\d+\s+days?\b",

    r"\b\d+\s+weeks?\s+ago\b",
    r"\bin\s+\d+\s+weeks?\b"
]


def extract_time(text):

    patterns = (
        HINDI_TIME_PATTERNS
        + ENGLISH_TIME_PATTERNS
    )

    matches = []

    for pattern in patterns:

        found = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for item in found:

            matches.append(item)

    if matches:

        return max(
            matches,
            key=len
        )

    try:

        parsed = search_dates(
            text,
            languages=["en", "hi"],
            settings={
                "PREFER_DATES_FROM": "past",
                "RETURN_AS_TIMEZONE_AWARE": False
            }
        )

        if parsed:

            return parsed[0][0]

    except Exception:

        pass

    return None


# ============================================================
# TARGET EXTRACTION
# ============================================================

def extract_target(text):

    english_patterns = [

        r"\bagainst\s+(.+?)(?:\.|$)",
        r"\ndemanding\s+(.+?)(?:\.|$)",
        r"\nover\s+(.+?)(?:\.|$)",
        r"\nfor\s+(.+?)(?:\.|$)"
    ]

    hindi_patterns = [

        r"के खिलाफ\s+(.+?)(?:।|$)",
        r"के विरोध में\s+(.+?)(?:।|$)",
        r"की मांग को लेकर\s+(.+?)(?:।|$)",
        r"के लिए\s+(.+?)(?:।|$)"
    ]

    for pattern in (
        english_patterns
        + hindi_patterns
    ):

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            if value:

                return value

    return None


# ============================================================
# CONTEXT
# ============================================================

def detect_event_context(text):

    text_lower = text.lower()

    # Cancelled
    cancelled_patterns = [

        "cancelled",
        "canceled",
        "called off",
        "रद्द",
        "स्थगित"
    ]

    for pattern in cancelled_patterns:

        if pattern in text_lower:

            return "CANCELLED"

    # Ongoing
    ongoing_patterns = [

        "right now",
        "currently",
        "at the moment",
        "ongoing",
        "are protesting",
        "are striking",
        "are demonstrating",

        "अभी",
        "इस समय",
        "जारी",
        "कर रहे हैं",
        "कर रही हैं",
        "हो रहा है"
    ]

    for pattern in ongoing_patterns:

        if pattern in text_lower:

            return "ONGOING"

    # Planned
    planned_patterns = [

        "will protest",
        "will demonstrate",
        "will strike",
        "will march",
        "will rally",

        "tomorrow",
        "next week",
        "next month",

        "करेंगे",
        "करेगी",
        "करने वाले हैं",
        "कल प्रदर्शन",
        "अगले सप्ताह"
    ]

    for pattern in planned_patterns:

        if pattern in text_lower:

            return "PLANNED"

    # Past
    past_patterns = [

        "yesterday",
        "last week",
        "last month",
        "last year",
        "ago",

        "protested",
        "demonstrated",
        "struck",
        "clashed",
        "attacked",

        "कल",
        "पिछले",
        "पहले",
        "किया",
        "किए",
        "हुआ",
        "हुई"
    ]

    for pattern in past_patterns:

        if pattern in text_lower:

            return "PAST"

    return "UNKNOWN"


# ============================================================
# COMPLETE EVENT EXTRACTION
# ============================================================

def extract_complete_event(text):

    event_info = detect_event_type_trigger(text)

    return {

        "event_type": event_info["event_type"],

        "trigger": event_info["trigger"],

        "actor": extract_actor(text),

        "location": extract_location(text),

        "time": extract_time(text),

        "target": extract_target(text),

        "context": detect_event_context(text)
    }


# ============================================================
# EVENT DETECTION
# ============================================================

def detect_event(text):

    inputs = event_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = event_model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    event_probability = (
        probabilities[0][1].item()
    )

    non_event_probability = (
        probabilities[0][0].item()
    )

    prediction = int(
        event_probability >= 0.5
    )

    return (
        prediction,
        event_probability,
        non_event_probability
    )


# ============================================================
# ESCALATION LEVEL
# ============================================================

def get_escalation_level(event_probability):

    percentage = event_probability * 100

    if percentage >= 75:

        return (
            "HIGH ESCALATION",
            "The model assigns a high probability that this text describes a civil unrest event."
        )

    elif percentage >= 50:

        return (
            "MODERATE ESCALATION",
            "The model assigns a moderate probability that this text describes a civil unrest event."
        )

    else:

        return (
            "LOW ESCALATION",
            "The model assigns a low probability that this text describes a civil unrest event."
        )


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def calculate_temporal_features(selected_date):

    if gdelt_df.empty:

        return None

    date = pd.Timestamp(
        selected_date
    )

    history = gdelt_df[
        gdelt_df["date"] <= date
    ].copy()

    if history.empty:

        return None

    history = history.set_index(
        "date"
    )

    daily = history[
        "protest_count"
    ].resample("D").sum()

    daily = daily.asfreq(
        "D",
        fill_value=0
    )

    if len(daily) < 15:

        return None

    lag_1 = daily.iloc[-1]

    lag_3 = (
        daily.iloc[-3]
        if len(daily) >= 3
        else 0
    )

    lag_7 = (
        daily.iloc[-7]
        if len(daily) >= 7
        else 0
    )

    lag_14 = (
        daily.iloc[-14]
        if len(daily) >= 14
        else 0
    )

    rolling_mean_3 = (
        daily.tail(3).mean()
    )

    rolling_mean_7 = (
        daily.tail(7).mean()
    )

    rolling_mean_14 = (
        daily.tail(14).mean()
    )

    rolling_std_7 = (
        daily.tail(7).std()
    )

    trend_7 = (
        rolling_mean_7
        - rolling_mean_14
    )

    features = {

        "lag_1": lag_1,

        "lag_3": lag_3,

        "lag_7": lag_7,

        "lag_14": lag_14,

        "rolling_mean_3":
            rolling_mean_3,

        "rolling_mean_7":
            rolling_mean_7,

        "rolling_mean_14":
            rolling_mean_14,

        "rolling_std_7":
            rolling_std_7,

        "trend_7":
            trend_7
    }

    return features


# ============================================================
# TEMPORAL PREDICTION
# ============================================================

def temporal_prediction(selected_date):

    if rf_model is None:

        return (
            "Temporal model unavailable.",
            ""
        )

    features = calculate_temporal_features(
        selected_date
    )

    if features is None:

        return (
            "Not enough GDELT history for this date.",
            ""
        )

    try:

        if FEATURES:

            feature_order = FEATURES

        else:

            feature_order = [
                "lag_1",
                "lag_3",
                "lag_7",
                "lag_14",
                "rolling_mean_3",
                "rolling_mean_7",
                "rolling_mean_14",
                "rolling_std_7",
                "trend_7"
            ]

        feature_row = pd.DataFrame(
            [[
                features[f]
                for f in feature_order
            ]],
            columns=feature_order
        )

        probability = rf_model.predict_proba(
            feature_row
        )[0][1]

        prediction = int(
            probability >= 0.5
        )

        if probability >= 0.75:

            risk = "HIGH"

        elif probability >= 0.50:

            risk = "MEDIUM"

        else:

            risk = "LOW"

        status = (
            "INCREASE EXPECTED"
            if prediction == 1
            else "NO INCREASE EXPECTED"
        )

        html = f"""
        <div class="temporal-card">

            <div class="temporal-status">
                {status}
            </div>

            <div class="temporal-probability">
                {probability * 100:.2f}%
            </div>

            <div class="temporal-label">
                Probability of Increased Protest Activity
            </div>

            <div class="risk-badge">
                Risk Level: {risk}
            </div>

            <hr>

            <div class="metric-grid">

                <div>
                    <span>Previous Day</span>
                    <strong>{features["lag_1"]:.0f}</strong>
                </div>

                <div>
                    <span>7-Day Average</span>
                    <strong>{features["rolling_mean_7"]:.1f}</strong>
                </div>

                <div>
                    <span>14-Day Average</span>
                    <strong>{features["rolling_mean_14"]:.1f}</strong>
                </div>

                <div>
                    <span>7-Day Trend</span>
                    <strong>{features["trend_7"]:.1f}</strong>
                </div>

            </div>

        </div>
        """

        return (
            html,
            ""
        )

    except Exception as e:

        return (
            f"Temporal prediction error: {str(e)}",
            ""
        )


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_event(text):

    if text is None:

        text = ""

    text = text.strip()

    if not text:

        return (
            """
            <div class="empty-card">
                Please enter a social media statement.
            </div>
            """,
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        )

    # -----------------------------------------------
    # EVENT DETECTION
    # -----------------------------------------------

    prediction, event_probability, _ = detect_event(
        text
    )

    probability_percentage = (
        event_probability * 100
    )

    # -----------------------------------------------
    # NO EVENT
    # -----------------------------------------------

    if prediction == 0:

        status_html = f"""

        <div class="status-card no-event">

            <div class="status-title">
                No Civil Unrest Event Detected
            </div>

            <div class="status-description">
                The model does not identify this text
                as a civil unrest-related event.
            </div>

            <div class="probability">
                {probability_percentage:.2f}%
            </div>

            <div class="probability-label">
                Event Probability
            </div>

            <div class="escalation low">
                LOW ESCALATION
            </div>

        </div>

        """

        return (
            status_html,
            "",
            "Not detected",
            "Not detected",
            "Not detected",
            "Not detected",
            "Not detected",
            "Not detected"
        )

    # -----------------------------------------------
    # STRUCTURED EXTRACTION
    # -----------------------------------------------

    result = extract_complete_event(
        text
    )

    event_type = (
        result["event_type"]
        or "Not detected"
    )

    trigger = (
        result["trigger"]
        or "Not detected"
    )

    actor = (
        result["actor"]
        or "Not detected"
    )

    location = (
        result["location"]
        or "Not detected"
    )

    time = (
        result["time"]
        or "Not detected"
    )

    target = (
        result["target"]
        or "Not detected"
    )

    context = (
        result["context"]
        or "UNKNOWN"
    )

    # -----------------------------------------------
    # ESCALATION
    # -----------------------------------------------

    escalation_level, escalation_description = (
        get_escalation_level(
            event_probability
        )
    )

    if "HIGH" in escalation_level:

        escalation_class = "high"

    elif "MODERATE" in escalation_level:

        escalation_class = "moderate"

    else:

        escalation_class = "low"

    # -----------------------------------------------
    # STATUS CARD
    # -----------------------------------------------

    status_html = f"""

    <div class="status-card">

        <div class="status-title">
            Civil Unrest Event Detected
        </div>

        <div class="status-description">
            The model identifies this text as an
            unrest-related event.
        </div>

        <div class="probability">
            {probability_percentage:.2f}%
        </div>

        <div class="probability-label">
            Event Probability
        </div>

        <div class="escalation {escalation_class}">
            {escalation_level}
        </div>

        <div class="escalation-description">
            {escalation_description}
        </div>

    </div>

    """

    # -----------------------------------------------
    # STRUCTURED INFORMATION
    # -----------------------------------------------

    structured_html = f"""

    <div class="structured-card">

        <div class="section-label">
            STRUCTURED EXTRACTION
        </div>

        <div class="section-title">
            Event Information
        </div>

        <div class="model-label">
            XLM-R + Multilingual NER
        </div>

        <div class="info-grid">

            <div class="info-item">
                <span>EVENT TYPE</span>
                <strong>{event_type}</strong>
            </div>

            <div class="info-item">
                <span>TRIGGER</span>
                <strong>{trigger}</strong>
            </div>

            <div class="info-item">
                <span>ACTOR</span>
                <strong>{actor}</strong>
            </div>

            <div class="info-item">
                <span>LOCATION</span>
                <strong>{location}</strong>
            </div>

            <div class="info-item">
                <span>TIME</span>
                <strong>{time}</strong>
            </div>

            <div class="info-item">
                <span>TARGET</span>
                <strong>{target}</strong>
            </div>

            <div class="info-item">
                <span>EVENT CONTEXT</span>
                <strong>{context}</strong>
            </div>

        </div>

    </div>

    """

    return (
        status_html,
        structured_html,
        event_type,
        trigger,
        actor,
        location,
        time,
        target
    )


# ============================================================
# CSS
# ============================================================

CSS = """

body {
    background: #f5f7fa !important;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

/* Header */

.hero {
    text-align: center;
    padding: 38px 20px 28px 20px;
}

.hero-small {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #6b7280;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 10px;
}

.hero-description {
    font-size: 15px;
    color: #6b7280;
    max-width: 720px;
    margin: auto;
    line-height: 1.6;
}

/* Input */

.input-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.04);
}

/* Buttons */

button.primary {
    border-radius: 12px !important;
    font-weight: 700 !important;
}

/* Status */

.status-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 28px;
    margin-top: 18px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.04);
}

.status-card.no-event {
    border-color: #e5e7eb;
}

.status-title {
    font-size: 24px;
    font-weight: 800;
    color: #111827;
}

.status-description {
    color: #6b7280;
    margin-top: 8px;
}

.probability {
    font-size: 46px;
    font-weight: 900;
    color: #111827;
    margin-top: 20px;
}

.probability-label {
    font-size: 12px;
    font-weight: 700;
    color: #9ca3af;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Escalation */

.escalation {
    display: inline-block;
    margin-top: 20px;
    padding: 10px 22px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 1px;
}

.escalation.high {
    background: #fee2e2;
    color: #991b1b;
}

.escalation.moderate {
    background: #fef3c7;
    color: #92400e;
}

.escalation.low {
    background: #dcfce7;
    color: #166534;
}

.escalation-description {
    margin-top: 12px;
    color: #6b7280;
    font-size: 13px;
}

/* Structured */

.structured-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 28px;
    margin-top: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.04);
}

.section-label {
    font-size: 11px;
    font-weight: 800;
    color: #6b7280;
    letter-spacing: 2px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #111827;
    margin-top: 5px;
}

.model-label {
    font-size: 12px;
    color: #9ca3af;
    margin-top: 5px;
}

.info-grid {
    display: grid;
    grid-template-columns:
        repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 24px;
}

.info-item {
    background: #f8fafc;
    border: 1px solid #eef2f7;
    border-radius: 12px;
    padding: 15px;
}

.info-item span {
    display: block;
    font-size: 10px;
    font-weight: 800;
    color: #9ca3af;
    letter-spacing: 1px;
    margin-bottom: 7px;
}

.info-item strong {
    display: block;
    color: #111827;
    font-size: 15px;
}

/* Temporal */

.temporal-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 28px;
    margin-top: 18px;
    text-align: center;
}

.temporal-status {
    font-size: 22px;
    font-weight: 800;
    color: #111827;
}

.temporal-probability {
    font-size: 44px;
    font-weight: 900;
    margin-top: 15px;
}

.temporal-label {
    font-size: 12px;
    color: #9ca3af;
    font-weight: 700;
    letter-spacing: 1px;
}

.risk-badge {
    display: inline-block;
    margin-top: 15px;
    padding: 9px 18px;
    border-radius: 999px;
    background: #f3f4f6;
    color: #374151;
    font-size: 12px;
    font-weight: 800;
}

.metric-grid {
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-top: 20px;
}

.metric-grid div {
    background: #f8fafc;
    border-radius: 12px;
    padding: 14px;
}

.metric-grid span {
    display: block;
    font-size: 10px;
    color: #9ca3af;
    font-weight: 700;
}

.metric-grid strong {
    display: block;
    font-size: 18px;
    margin-top: 5px;
    color: #111827;
}

/* Footer */

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 11px;
    margin-top: 30px;
    padding-bottom: 25px;
}

/* Mobile */

@media (max-width: 700px) {

    .hero-title {
        font-size: 28px;
    }

    .info-grid {
        grid-template-columns: 1fr;
    }

    .metric-grid {
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }
}

"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    css=CSS,
    title="Civil Unrest Analyzer"
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-small">
                AI-POWERED EARLY WARNING SYSTEM
            </div>

            <div class="hero-title">
                Civil Unrest Analyzer
            </div>

            <div class="hero-description">
                Detect, extract and analyze civil unrest
                events from multilingual social media text,
                with AI-based event probability and
                escalation assessment.
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    with gr.Column(
        elem_classes="input-card"
    ):

        gr.Markdown(
            "### Multilingual Event Extraction"
        )

        gr.Markdown(
            "**English + Hindi**"
        )

        text_input = gr.Textbox(
            label="Social Media Statement",
            placeholder=(
                "Enter an English or Hindi social media statement...\n\n"
                "Example:\n"
                "Thousands of students protested in Delhi yesterday "
                "against the new education policy.\n\n"
                "Example Hindi:\n"
                "दिल्ली में हजारों छात्रों ने नई शिक्षा नीति के "
                "खिलाफ विरोध प्रदर्शन किया।"
            ),
            lines=5
        )

        analyze_button = gr.Button(
            "Analyze Event",
            variant="primary",
            size="lg"
        )

    # --------------------------------------------------------
    # EVENT STATUS
    # --------------------------------------------------------

    event_status_output = gr.HTML()

    # --------------------------------------------------------
    # STRUCTURED EXTRACTION
    # --------------------------------------------------------

    structured_output = gr.HTML()

    # --------------------------------------------------------
    # HIDDEN / CLEAN OUTPUTS
    # --------------------------------------------------------

    with gr.Row(visible=False):

        event_type_output = gr.Textbox()

        trigger_output = gr.Textbox()

        actor_output = gr.Textbox()

        location_output = gr.Textbox()

        time_output = gr.Textbox()

        target_output = gr.Textbox()

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="footer">

            CIVIL UNREST EVENT EXTRACTION & PREDICTION<br>

            XLM-RoBERTa · Multilingual NER · Random Forest · GDELT

        </div>
        """
    )

    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    analyze_button.click(
        fn=analyze_event,
        inputs=text_input,
        outputs=[
            event_status_output,
            structured_output,
            event_type_output,
            trigger_output,
            actor_output,
            location_output,
            time_output,
            target_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("CIVIL UNREST ANALYZER")
    print("=" * 60)

    print(
        "\nOpen this in your browser:"
    )

    print(
        "http://127.0.0.1:7860"
    )

    print("=" * 60 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )