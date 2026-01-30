"""
MISInsight-Pro — Streamlit app (Local + Cloud Friendly)

Features:
- Upload CSV MIS reports
- Preview + summary
- Three analysis modes:
    1) Rule-based (offline)
    2) Ollama (local LLM)
    3) Groq (cloud Llama)  ✅ works in Streamlit Cloud
- Generates at least 5 actionable strategies
- Strict scope: only MIS report questions
- Download output (TXT / JSON)

Requirements:
pip install streamlit pandas requests groq
"""

import streamlit as st
import pandas as pd
import os
import json
import requests
import traceback
from typing import List, Dict
from groq import Groq

# --------------------------- Configuration ---------------------------
FALLBACK_MESSAGE = "I can only help with the MIS report you provided."
MIN_STRATEGIES = 5

# Ollama defaults
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

# Groq defaults
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# --------------------------- Prompting ---------------------------
SYSTEM_PROMPT = (
    "You are an expert MIS analyst called MISInsight-Pro. "
    "You will ONLY answer questions or provide strategies that are directly supported by the provided MIS report. "
    "If the user asks anything outside the report, respond exactly with: "
    "\"I can only help with the MIS report you provided.\""
)

LLM_INSTRUCTION_TEMPLATE = (
    "You are given an MIS report summary below. Produce exactly a JSON object with two keys: "
    "'insights' (a short list of 3-8 bullet insights), "
    "and 'strategies' (a list of at least 5 concrete, actionable strategies tied to the report). "
    "Each strategy should be a short sentence and must reference fields/columns from the summary if possible. "
    "Do NOT invent data not in the summary. If something cannot be inferred, be conservative. "
    "Do NOT include any headings, explanations, markdown formatting, or extra text. "
    "Return ONLY valid JSON with 2 keys: 'insights' and 'strategies'.\n\n"
    "The output MUST be strictly parseable JSON.\n\n"
    "Format:\n"
    "{{\n"
    "  \"insights\": [\"...\", \"...\", \"...\"],\n"
    "  \"strategies\": [\"...\", \"...\", \"...\", \"...\", \"...\"]\n"
    "}}\n\n"
    "MIS SUMMARY:\n{summary}\n\n"
    "End of summary."
)

# --------------------------- Utilities ---------------------------

def read_csv(file) -> pd.DataFrame:
    try:
        try:
            df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1", on_bad_lines="skip")

        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")
        df = df.fillna("")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.error(traceback.format_exc())
        return pd.DataFrame()


def summarize_df(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "No data available in the dataframe."

    try:
        buf = []
        buf.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        buf.append("Columns and dtypes:")
        for c, t in zip(df.columns, df.dtypes):
            buf.append(f"- {c}: {t}")

        buf.append("\nSample rows:")
        sample = df.head(max_rows).fillna("")
        buf.append(sample.to_csv(index=False))

        num = df.select_dtypes(include=["number"])
        if not num.empty:
            buf.append("\nNumeric summary (mean, min, max):")
            stats = num.agg(["mean", "min", "max"]).round(3).to_dict()
            for col, vals in stats.items():
                buf.append(f"- {col}: mean={vals['mean']}, min={vals['min']}, max={vals['max']}")

        return "\n".join(buf)

    except Exception as e:
        return f"Error creating summary: {str(e)}"


def ensure_min_strategies(strategies: List[str], default_fillers: List[str]) -> List[str]:
    s = strategies.copy() if strategies else []
    idx = 0
    while len(s) < MIN_STRATEGIES and idx < len(default_fillers):
        s.append(default_fillers[idx])
        idx += 1
    return s


def safe_json_parse(llm_output: str) -> Dict[str, List[str]]:
    """
    Tries strict JSON parse first.
    If fails, attempts to extract JSON block.
    """
    try:
        return json.loads(llm_output)
    except Exception:
        # Try to extract JSON block inside braces
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        if start != -1 and end != -1 and end > start:
            block = llm_output[start:end+1]
            return json.loads(block)
        raise


# --------------------------- LLM Backends ---------------------------

def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": 900,
        "temperature": 0.2,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if resp.status_code != 200:
            return f"ERROR: Ollama request failed: {resp.status_code} {resp.text}"

        try:
            result = resp.json()
            return result.get("response", "").strip()
        except json.JSONDecodeError:
            return resp.text.strip()

    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure Ollama is running at " + OLLAMA_URL
    except requests.exceptions.Timeout:
        return "ERROR: Ollama request timed out."
    except Exception as e:
        return f"ERROR: Ollama call exception: {e}"


def call_groq(user_prompt: str) -> str:
    """
    Uses Groq chat completion API.
    Reads key from env or Streamlit secrets.
    """
    try:
        api_key = GROQ_API_KEY
        if not api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            return "ERROR: GROQ_API_KEY not set. Add it to Streamlit Secrets or env variables."

        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=900,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"ERROR: Groq call exception: {e}"


# --------------------------- Rule-based Analyzer ---------------------------

def rule_based_analysis(df: pd.DataFrame) -> Dict[str, List[str]]:
    insights = []
    strategies = []

    try:
        insights.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        insights.append(f"Column names: {', '.join(df.columns.tolist())}")

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            insights.append(f"Numeric columns found: {', '.join(numeric_cols)}")

        sales_cols = [c for c in df.columns if any(k in c.lower() for k in ["sales", "revenue", "income"])]
        expense_cols = [c for c in df.columns if any(k in c.lower() for k in ["expense", "cost", "spend"])]
        region_cols = [c for c in df.columns if any(k in c.lower() for k in ["region", "location", "area", "territory"])]

        # Sales patterns
        for col in sales_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                total = float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
                avg = float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())
                insights.append(f"Total {col}={total:.2f}, Average {col}={avg:.2f}")
                if avg < 1000:
                    strategies.append(f"Increase promotions or outreach to improve {col}.")
                else:
                    strategies.append(f"Identify underperforming categories and optimize {col} distribution.")

        # Expense patterns
        for col in expense_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                avg_exp = float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())
                insights.append(f"Average {col}={avg_exp:.2f}")
                if avg_exp > 5000:
                    strategies.append(f"Audit top drivers in {col} and implement cost reduction controls.")

        # Regional analysis
        if region_cols and sales_cols:
            try:
                region_col = region_cols[0]
                sales_col = sales_cols[0]
                if pd.api.types.is_numeric_dtype(df[sales_col]):
                    grouped = df.groupby(region_col)[sales_col].sum().sort_values()
                    if len(grouped) > 0:
                        insights.append(f"Performance by {region_col}: lowest={grouped.index[0]}, highest={grouped.index[-1]}")
                        strategies.append(
                            f"Focus improvement in {grouped.index[0]} and replicate practices from {grouped.index[-1]}."
                        )
            except Exception:
                pass

        default_fillers = [
            "Implement weekly dashboard monitoring for KPIs in the MIS report.",
            "Conduct monthly reviews of the top transactions for anomalies and patterns.",
            "Create automated alerts for performance threshold breaches.",
            "Establish benchmark comparisons with historical performance.",
            "Assign ownership for each strategy and track progress weekly."
        ]

        strategies = ensure_min_strategies(strategies, default_fillers)
        return {"insights": insights, "strategies": strategies}

    except Exception as e:
        return {
            "insights": [
                f"Dataset contains {len(df)} rows and {len(df.columns)} columns",
                f"Analysis error occurred: {str(e)}",
                "Manual review recommended"
            ],
            "strategies": [
                "Review data quality and completeness",
                "Validate key MIS metrics manually",
                "Establish baseline KPI tracking",
                "Create regular monitoring processes",
                "Implement data validation checks"
            ]
        }


# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="MISInsight-Pro", layout="wide")
st.title("MISInsight-Pro — MIS Insight Chatbot")
st.write("Upload your MIS report (CSV exported from Excel). The app will generate insights & strategies strictly from the uploaded report.")

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload MIS report (CSV)", type=["csv"])
    example = st.checkbox("Use example demo dataset (if you don't have a file)")

with col2:
    backend = st.selectbox(
        "Analysis Mode / Backend",
        ["Rule-based (offline)", "Ollama (local LLM)", "Groq (Cloud Llama)"]
    )

    st.markdown("**Strict scope behavior:** If the user asks anything outside the uploaded MIS report, the assistant replies with fallback message.")

    if backend == "Ollama (local LLM)":
        st.write(f"Ollama endpoint: {OLLAMA_URL}")
        st.write(f"Model: {OLLAMA_MODEL}")
    if backend == "Groq (Cloud Llama)":
        st.write(f"Groq model: {GROQ_MODEL}")
        st.info("Groq works on deployed Streamlit apps (no localhost dependency).")


# Load dataset
df = pd.DataFrame()

if example and uploaded_file is None:
    df = pd.DataFrame({
        "Region": ["North", "South", "East", "West", "North"],
        "Sales": [1200, 800, 400, 1500, 1100],
        "Expenses": [4000, 5200, 3000, 4500, 4800],
        "Product": ["A", "B", "C", "A", "D"]
    })
    st.info("Using demo dataset.")
elif uploaded_file:
    df = read_csv(uploaded_file)
    if not df.empty:
        st.success(f"File loaded successfully! {len(df)} rows, {len(df.columns)} columns")

if df.empty:
    st.info("No report loaded yet. Upload a CSV or use the demo dataset.")
    st.stop()

# Preview
st.subheader("Report Preview")
st.dataframe(df, use_container_width=True)

with st.expander("Show Dataset Summary"):
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.write("**Data Types:**")
    st.write(df.dtypes)
    if not df.select_dtypes(include=["number"]).empty:
        st.write("**Numeric Summary:**")
        st.write(df.describe())

# Debug
with st.expander("Debug Information"):
    st.write("DataFrame empty:", df.empty)
    st.write("DataFrame shape:", df.shape)
    st.write("Backend selected:", backend)

# Analyze Button
st.subheader("Analysis & Strategies")
run_button = st.button("🔍 Analyze Report", type="primary")

if run_button:
    with st.spinner("Generating insights and strategies..."):
        try:
            summary = summarize_df(df, max_rows=8)

            # Choose backend
            if backend == "Rule-based (offline)":
                result = rule_based_analysis(df)
                insights = result["insights"]
                strategies = result["strategies"]

            else:
                user_prompt = LLM_INSTRUCTION_TEMPLATE.format(summary=summary)

                if backend == "Ollama (local LLM)":
                    llm_output = call_ollama(SYSTEM_PROMPT + "\n\n" + user_prompt)

                elif backend == "Groq (Cloud Llama)":
                    llm_output = call_groq(user_prompt)

                else:
                    llm_output = "ERROR: Invalid backend."

                if llm_output.startswith("ERROR:"):
                    st.error(llm_output)
                    st.info("Falling back to rule-based analysis...")
                    result = rule_based_analysis(df)
                    insights = result["insights"]
                    strategies = result["strategies"]
                else:
                    # Parse JSON
                    try:
                        parsed = safe_json_parse(llm_output)
                        insights = parsed.get("insights", [])
                        strategies = parsed.get("strategies", [])
                    except Exception:
                        st.warning("LLM output could not be parsed strictly as JSON. Using text fallback parsing.")
                        lines = [l.strip("-• \t") for l in llm_output.splitlines() if l.strip()]
                        insights = lines[:5]
                        strategies = lines[5:10] if len(lines) > 10 else lines[:5]

            # Ensure minimum strategies
            default_fillers = [
                "Implement regular KPI monitoring based on the uploaded MIS data.",
                "Set up automated reporting for key columns and trends.",
                "Add alerts when metrics cross critical thresholds.",
                "Perform weekly reviews of anomalies or outlier values in numeric columns.",
                "Track each strategy with assigned responsibility and timelines."
            ]
            strategies = ensure_min_strategies(strategies, default_fillers)

            st.session_state.analysis_results = {"insights": insights, "strategies": strategies}

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.error(traceback.format_exc())

# Display results
if st.session_state.analysis_results:
    insights = st.session_state.analysis_results["insights"]
    strategies = st.session_state.analysis_results["strategies"]

    st.subheader("📊 Key Insights")
    if insights:
        for i, ins in enumerate(insights, 1):
            st.write(f"**{i}.** {ins}")
    else:
        st.write("No specific insights were generated from the data.")

    st.subheader("🎯 Actionable Strategies")
    for i, s in enumerate(strategies, 1):
        st.write(f"**{i}.** {s}")

    # Downloads
    st.subheader("📥 Download Results")
    result_blob = {
        "insights": insights,
        "strategies": strategies,
        "analysis_timestamp": str(pd.Timestamp.now()),
        "data_shape": list(df.shape),
        "columns": list(df.columns),
        "backend": backend
    }

    download_content = "MIS ANALYSIS RESULTS\n" + "=" * 50 + "\n\n"
    download_content += "INSIGHTS:\n"
    for i, insight in enumerate(insights, 1):
        download_content += f"{i}. {insight}\n"
    download_content += "\nSTRATEGIES:\n"
    for i, strategy in enumerate(strategies, 1):
        download_content += f"{i}. {strategy}\n"

    st.download_button(
        "Download as Text File",
        download_content,
        file_name=f"mis_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )

    st.download_button(
        "Download as JSON",
        json.dumps(result_blob, indent=2),
        file_name=f"mis_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

# Scope-limited Q/A
st.subheader("💬 Ask About Your Report")
user_q = st.text_input("Ask a question about the uploaded MIS report only")

if user_q:
    lower = user_q.lower()
    unrelated_keywords = [
        "weather", "movie", "football", "politics", "love",
        "recipe", "how to make", "what is python", "help me code"
    ]

    if any(k in lower for k in unrelated_keywords):
        st.error(FALLBACK_MESSAGE)
    else:
        prompt_context = summarize_df(df, max_rows=6)

        if backend == "Rule-based (offline)":
            answers = []
            for col in df.columns:
                if col.lower() in lower:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 10:
                        answers.append(f"**{col}** unique values: {list(unique_vals)}")
                    else:
                        answers.append(f"**{col}** has {len(unique_vals)} unique values. Sample: {list(unique_vals[:5])}")

            if answers:
                for answer in answers:
                    st.write(answer)
            else:
                st.write("I can only provide information directly available in your uploaded MIS report data.")

        else:
            qa_prompt = (
                "Report summary:\n" + prompt_context +
                "\n\nQuestion: " + user_q +
                "\n\nProvide a concise answer based only on the report data."
            )

            with st.spinner("Generating answer..."):
                if backend == "Ollama (local LLM)":
                    ans = call_ollama(SYSTEM_PROMPT + "\n\n" + qa_prompt)
                elif backend == "Groq (Cloud Llama)":
                    ans = call_groq(qa_prompt)
                else:
                    ans = "ERROR: Invalid backend."

            if ans.startswith("ERROR:"):
                st.error(ans)
            else:
                st.write(ans)

# Sidebar
st.sidebar.header("About")
st.sidebar.write("**MISInsight-Pro** — MIS Insight Chatbot")
st.sidebar.write("Created by Shaun Mathew.")
st.sidebar.write("Strictly scoped to uploaded MIS report data.")

st.sidebar.header("Configuration")
st.sidebar.write(f"**Ollama URL:** {OLLAMA_URL}")
st.sidebar.write(f"**Ollama Model:** {OLLAMA_MODEL}")
st.sidebar.write(f"**Groq Model:** {GROQ_MODEL}")

st.sidebar.header("Groq Setup (Deployment)")
st.sidebar.write("Set Streamlit secret: GROQ_API_KEY")
