import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import joblib

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =====================================================
# SETUP
# =====================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide"
)


# =====================================================
# TRAINING DATA
# =====================================================

messages = [
    "Congratulations you won 50000 rupees click to claim your prize",
    "Urgent your bank account will be blocked verify immediately",
    "You won a lottery send your bank details to receive money",
    "Click this link to receive your free gift",
    "Send your OTP to receive cashback",
    "You are the lucky winner claim your reward now",
    "Your credit card has been suspended click to activate",
    "Claim your cash reward immediately",
    "Your account will be blocked click to verify now",
    "You have won a free prize claim it today",
    "Urgent send your password to receive your reward",
    "Verify your bank details immediately to receive money",

    "Hi how are you",
    "Please send me the assignment",
    "Your class starts at 9 AM tomorrow",
    "Can you call me when you are free",
    "Your order has been delivered",
    "The meeting is scheduled for tomorrow",
    "Please submit the project before Friday",
    "Happy birthday have a great day",
    "Your college class has been postponed",
    "Please bring your project tomorrow",
    "The team meeting starts at 10 AM",
    "Your food delivery has arrived"
]

labels = [
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,

    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
]

df = pd.DataFrame({
    "message": messages,
    "label": labels
})


# =====================================================
# MACHINE LEARNING MODEL
# =====================================================

MODEL_FILE = "scamshield_model.joblib"

if os.path.exists(MODEL_FILE):

    vectorizer, model = joblib.load(
        MODEL_FILE
    )

else:

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(
        df["message"]
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X,
        df["label"]
    )

    joblib.dump(
        (vectorizer, model),
        MODEL_FILE
    )


# =====================================================
# WARNING CATEGORIES
# =====================================================

warning_categories = {

    "Urgency": [
        "urgent",
        "immediately",
        "now",
        "today",
        "act fast",
        "within 30 minutes"
    ],

    "Money / Reward": [
        "money",
        "cash",
        "prize",
        "reward",
        "winner",
        "lottery",
        "cashback",
        "free",
        "gift"
    ],

    "Sensitive Information": [
        "otp",
        "password",
        "pin",
        "bank details",
        "account details",
        "bank account",
        "verify"
    ],

    "Suspicious Action": [
        "click",
        "claim",
        "activate",
        "blocked",
        "suspended",
        "verify"
    ]
}


# =====================================================
# MESSAGE ANALYSIS
# =====================================================

def analyze_message(text):

    text = text.lower()

    found = {}
    score = 0

    for category, words in warning_categories.items():

        matches = [
            word
            for word in words
            if word in text
        ]

        if matches:

            found[category] = matches
            score += len(matches) * 8

    return min(score, 100), found


# =====================================================
# URL ANALYSIS
# =====================================================

def analyze_url(url):

    score = 0
    warnings = []

    url_lower = url.lower()

    if not url_lower.startswith("https://"):

        score += 20

        warnings.append(
            "URL does not use HTTPS."
        )

    ip_pattern = r"https?://\d{1,3}(\.\d{1,3}){3}"

    if re.match(ip_pattern, url):

        score += 25

        warnings.append(
            "URL uses an IP address instead of a domain."
        )

    suspicious_words = [
        "login",
        "verify",
        "claim",
        "winner",
        "prize",
        "free",
        "reward",
        "bank",
        "password",
        "account",
        "urgent"
    ]

    found = [
        word
        for word in suspicious_words
        if word in url_lower
    ]

    if found:

        score += len(found) * 10

        warnings.append(
            "Suspicious words found: "
            + ", ".join(found)
        )

    if len(url) > 100:

        score += 15

        warnings.append(
            "URL is unusually long."
        )

    if "@" in url:

        score += 15

        warnings.append(
            "URL contains an @ symbol."
        )

    return min(score, 100), warnings


# =====================================================
# FIND URLS
# =====================================================

def find_urls(text):

    return re.findall(
        r"https?://[^\s]+",
        text
    )


# =====================================================
# RISK LEVEL
# =====================================================

def get_risk_level(score):

    if score >= 70:

        return "🔴 High Risk"

    elif score >= 40:

        return "🟠 Medium Risk"

    return "🟢 Low Risk"


# =====================================================
# GEMINI AI EXPLANATION
# =====================================================

def get_gemini_explanation(
    message,
    risk_score,
    warnings
):

    if not GEMINI_API_KEY:

        return (
            "Gemini API key not found. "
            "Please check your .env file."
        )

    warning_text = ""

    for category, words in warnings.items():

        warning_text += (
            f"{category}: "
            f"{', '.join(words)}\n"
        )

    prompt = f"""
You are a cybersecurity awareness assistant.

Analyze this message:

{message}

Machine learning risk score:
{risk_score:.1f}%

Detected warning signs:
{warning_text}

Explain in simple language:

1. Why this message may be suspicious
2. Important warning signs
3. What the user should do

Do not claim with certainty that it is a scam.
Use "potentially suspicious" when appropriate.

Keep the answer short and easy to understand.
"""

    try:

        # =================================================
        # GET AVAILABLE GEMINI MODELS
        # =================================================

        models_url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models"
            f"?key={GEMINI_API_KEY}"
        )

        models_response = requests.get(
            models_url,
            timeout=20
        )

        if models_response.status_code != 200:

            return (
                "Gemini API could not list available models. "
                f"Status: {models_response.status_code}"
            )

        models_data = models_response.json()

        available_models = []

        for model_info in models_data.get(
            "models",
            []
        ):

            methods = model_info.get(
                "supportedGenerationMethods",
                []
            )

            if "generateContent" in methods:

                model_name = model_info.get(
                    "name",
                    ""
                )

                if model_name:

                    available_models.append(
                        model_name
                    )

        if not available_models:

            return (
                "No Gemini model supporting text "
                "generation is available for this API key."
            )

        # =================================================
        # PREFERRED MODELS
        # =================================================

        preferred_models = [
            "models/gemini-3.8-flash",
            "models/gemini-3-flash",
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash"
        ]

        selected_model = None

        for preferred in preferred_models:

            if preferred in available_models:

                selected_model = preferred
                break

        # Use first available model if preferred
        # models are not available

        if selected_model is None:

            selected_model = available_models[0]

        # =================================================
        # GENERATE GEMINI RESPONSE
        # =================================================

        generate_url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/{selected_model}:generateContent"
            f"?key={GEMINI_API_KEY}"
        )

        payload = {

            "contents": [

                {

                    "parts": [

                        {
                            "text": prompt
                        }

                    ]

                }

            ]

        }

        response = requests.post(
            generate_url,
            json=payload,
            timeout=30
        )

        # Successful response
        if response.status_code == 200:

            data = response.json()

            return (
                data["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

        # =================================================
        # TRY ANOTHER MODEL IF 503
        # =================================================

        if response.status_code == 503:

            for backup_model in available_models:

                if backup_model == selected_model:
                    continue

                backup_url = (
                    "https://generativelanguage.googleapis.com/"
                    f"v1beta/{backup_model}:generateContent"
                    f"?key={GEMINI_API_KEY}"
                )

                backup_response = requests.post(
                    backup_url,
                    json=payload,
                    timeout=30
                )

                if backup_response.status_code == 200:

                    backup_data = (
                        backup_response.json()
                    )

                    return (
                        backup_data["candidates"][0]
                        ["content"]["parts"][0]["text"]
                    )

            return (
                "Gemini models are temporarily unavailable. "
                "Please try again later."
            )

        # Request limit
        if response.status_code == 429:

            return (
                "Gemini API request limit reached. "
                "Please try again later."
            )

        # API key problem
        if response.status_code in [
            400,
            401,
            403
        ]:

            return (
                "Gemini API could not process the request. "
                "Please check your API key and API access."
            )

        return (
            "Gemini could not analyze the message. "
            f"Status: {response.status_code}"
        )

    except requests.Timeout:

        return (
            "Gemini took too long to respond. "
            "Please try again."
        )

    except requests.RequestException:

        return (
            "Unable to connect to Gemini API. "
            "Please check your internet connection."
        )

    except (KeyError, IndexError):

        return (
            "Gemini returned an unexpected response."
        )


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "🛡️ ScamShield AI"
)

page = st.sidebar.radio(
    "Choose",
    [
        "📩 Message Analyzer",
        "🔗 URL Analyzer",
        "📊 Model Info",
        "ℹ️ About"
    ]
)


# =====================================================
# MESSAGE ANALYZER
# =====================================================

if page == "📩 Message Analyzer":

    st.header(
        "📩 Message Analyzer"
    )

    message = st.text_area(
        "Paste a suspicious message",
        height=180,
        placeholder=(
            "Example: Congratulations! "
            "You won ₹50,000. "
            "Click here to claim your prize."
        )
    )

    if st.button(
        "🔍 Analyze Message",
        use_container_width=True
    ):

        if not message.strip():

            st.warning(
                "Please enter a message."
            )

        else:

            # ML prediction
            vector = vectorizer.transform(
                [message]
            )

            probability = model.predict_proba(
                vector
            )[0]

            ml_score = probability[1] * 100

            # Rule analysis
            rule_score, warnings = (
                analyze_message(message)
            )

            # Final risk score
            final_score = np.average(
                [ml_score, rule_score],
                weights=[0.7, 0.3]
            )

            final_score = min(
                final_score,
                100
            )

            st.divider()

            st.subheader(
                "Analysis Result"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Risk Score",
                    f"{final_score:.1f}%"
                )

            with col2:

                st.metric(
                    "ML Score",
                    f"{ml_score:.1f}%"
                )

            with col3:

                st.metric(
                    "Warning Categories",
                    len(warnings)
                )

            st.progress(
                int(final_score)
            )

            st.write(
                f"### {get_risk_level(final_score)}"
            )

            # Warning signs
            st.subheader(
                "🚨 Warning Signs"
            )

            if warnings:

                for category, words in warnings.items():

                    st.write(
                        f"**{category}:** "
                        + ", ".join(words)
                    )

            else:

                st.success(
                    "No common warning signs detected."
                )

            # URLs
            urls = find_urls(message)

            if urls:

                st.subheader(
                    "🔗 Links Found"
                )

                for url in urls:

                    st.code(url)

            # Gemini explanation
            st.subheader(
                "🤖 Gemini AI Explanation"
            )

            with st.spinner(
                "Gemini is analyzing the message..."
            ):

                explanation = (
                    get_gemini_explanation(
                        message,
                        final_score,
                        warnings
                    )
                )

            st.info(
                explanation
            )

            # Safety advice
            st.subheader(
                "🛡️ Safety Advice"
            )

            if final_score >= 70:

                st.error(
                    "Do not click suspicious links or "
                    "share OTPs, passwords or banking "
                    "information."
                )

            elif final_score >= 40:

                st.warning(
                    "Verify the sender and information "
                    "before taking any action."
                )

            else:

                st.success(
                    "No major warning signs were detected. "
                    "Still verify unexpected requests."
                )


# =====================================================
# URL ANALYZER
# =====================================================

elif page == "🔗 URL Analyzer":

    st.header(
        "🔗 URL Analyzer"
    )

    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com"
    )

    if st.button(
        "🔍 Check URL",
        use_container_width=True
    ):

        if not url.strip():

            st.warning(
                "Please enter a URL."
            )

        else:

            score, warnings = analyze_url(
                url
            )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Risk Score",
                    f"{score}%"
                )

            with col2:

                st.metric(
                    "Risk Level",
                    get_risk_level(score)
                )

            st.progress(
                int(score)
            )

            st.subheader(
                "🔎 URL Warning Signs"
            )

            if warnings:

                for warning in warnings:

                    st.write(
                        f"⚠️ {warning}"
                    )

            else:

                st.success(
                    "No obvious suspicious characteristics "
                    "detected."
                )


# =====================================================
# MODEL INFORMATION
# =====================================================

elif page == "📊 Model Info":

    st.header(
        "📊 Model Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Algorithm",
            "Logistic Regression"
        )

    with col2:

        st.metric(
            "Feature Extraction",
            "TF-IDF"
        )

    st.write(
        """
        TF-IDF converts messages into numerical features.

        Logistic Regression uses these features to
        estimate whether a message may be suspicious.
        """
    )

    st.subheader(
        "Training Dataset"
    )

    display_df = df.copy()

    display_df["label"] = (
        display_df["label"].map({
            1: "Scam",
            0: "Safe"
        })
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )


# =====================================================
# ABOUT
# =====================================================

elif page == "ℹ️ About":

    st.header(
        "ℹ️ About ScamShield AI"
    )

    st.write(
        """
        ScamShield AI analyzes suspicious messages and URLs.

        It combines machine learning, rule-based analysis
        and Gemini AI to help users understand possible
        scam indicators.
        """
    )

    st.subheader(
        "🎯 Project Goal"
    )

    st.write(
        """
        The goal is not only to detect possible risk,
        but also to explain why a message may be suspicious
        and provide simple safety advice.
        """
    )

    st.subheader(
        "🧠 Technologies Used"
    )

    technologies = pd.DataFrame({

        "Technology": [
            "Python",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Joblib",
            "Requests",
            "Gemini API"
        ],

        "Purpose": [
            "Application logic",
            "Web interface",
            "Data handling",
            "Score calculation",
            "Machine learning",
            "Model saving",
            "API communication",
            "AI explanation"
        ]
    })

    st.table(
        technologies
    )

    st.warning(
        "ScamShield AI is an awareness tool and cannot "
        "guarantee that a message or URL is completely "
        "safe or fraudulent."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "🛡️ ScamShield AI | "
    "Don't just detect the risk — understand the risk."
)