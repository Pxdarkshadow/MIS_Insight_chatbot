"""
MISInsight-Pro — Local Streamlit app
Single-file local implementation that mimics the PartyRock MISInsight-Pro behavior.

Features:
- Upload CSV (exported from Excel) MIS reports
- Quick table preview + automated summary (descriptive stats)
- Three analysis modes: Rule-based, OpenAI (API key), Ollama (local LLM)
- Always returns at least 5 actionable strategies
- Strict scope: only answers questions related to the uploaded MIS report. If the user asks anything outside the report, the app replies with the configured strict fallback.
- Option to download strategies as a text file

Usage:
1. Install dependencies: pip install -r requirements.txt
   (requirements.txt: streamlit,pandas,openai,requests)
2. Run: streamlit run MISInsight-Pro_local_app.py
3. Upload CSV (export from Excel) and choose LLM/backend.

Note: If you want purely offline use, install Ollama (or another local LLM) and point the app to the local Ollama server. If you prefer OpenAI, set OPENAI_API_KEY as environment variable.

"""

import streamlit as st
import pandas as pd
import os
import json
import tempfile
import requests
from typing import List, Dict, Any
import traceback

# --------------------------- Configuration ---------------------------
FALLBACK_MESSAGE = "I can only help with the MIS report you provided."
MIN_STRATEGIES = 5

# Ollama defaults (adjust model name to your local model)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")  # change as needed

# --------------------------- Utilities ---------------------------

def read_csv(file) -> pd.DataFrame:
    try:
        # Try different approaches for reading CSV
        try:
            # First try with encoding utf-8 with the condition of skipping bad lines
            df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            # If that fails, try with encoding latin1 with the condition of skipping bad lines
            file.seek(0)  # Reset file pointer
            df = pd.read_csv(file, encoding="latin1", on_bad_lines="skip")
        
        # Clean column names (remove extra whitespace)
        df = df.dropna(axis=1, how="all")         # drop completely empty columns
        df = df.dropna(axis=0, how="all")         # drop completely empty rows
        df = df.fillna("")                        # replace NaN with empty string (or use df.fillna("N/A"))
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.error(f"Error details: {traceback.format_exc()}")
        return pd.DataFrame()


def summarize_df(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Return a compact textual summary of the dataframe to feed into LLMs."""
    if df.empty:
        return "No data available in the dataframe."
    
    try:
        buf = []
        buf.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        buf.append("Columns and dtypes:")
        for c, t in zip(df.columns, df.dtypes):
            buf.append(f"- {c}: {t}")

        # show first N rows (limited)
        buf.append("\nSample rows:")
        sample = df.head(max_rows).fillna("")
        buf.append(sample.to_csv(index=False))

        # numeric summary
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

# --------------------------- LLM Backend ---------------------------

def call_ollama(prompt: str) -> str:
    """Call local Ollama HTTP generate endpoint. Adjusted for better error handling."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": 800,
        "temperature": 0.2,
        "stream": False  # Disable streaming for simpler response handling
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if resp.status_code != 200:
            return f"ERROR: Ollama request failed: {resp.status_code} {resp.text}"
        
        # Try to parse as JSON first
        try:
            result = resp.json()
            if "response" in result:
                return result["response"].strip()
            else:
                return str(result)
        except json.JSONDecodeError:
            # If not JSON, return raw text
            return resp.text.strip()

    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure Ollama is running at " + OLLAMA_URL
    except requests.exceptions.Timeout:
        return "ERROR: Ollama request timed out. The model might be processing a large request."
    except Exception as e:
        return f"ERROR: Ollama call exception: {e}"

# --------------------------- Prompting ---------------------------

SYSTEM_PROMPT = (
    "You are an expert MIS analyst called MISInsight-Pro. "
    "You will ONLY answer questions or provide strategies that are directly supported by the provided MIS report. "
    "If the user asks anything outside the report, respond exactly with: \"I can only help with the MIS report you provided.\""
)

LLM_INSTRUCTION_TEMPLATE = (
    "You are given an MIS report summary below. Produce exactly a JSON object with two keys: 'insights' (a short list of 3-8 bullet insights), "
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

# --------------------------- Rule-based quick analyzer ---------------------------

def rule_based_analysis(df: pd.DataFrame) -> Dict[str, List[str]]:
    def basic_info(df):
        insights = [
            f"Dataset contains {len(df)} rows and {len(df.columns)} columns",
            f"Column names: {', '.join(df.columns.tolist())}"
        ]
        return insights

    def numeric_column_analysis(df):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        insights = []
        if numeric_cols:
            insights.append(f"Numeric columns found: {', '.join(numeric_cols)}")
        return insights, numeric_cols

    def sales_analysis(df, sales_cols):
        insights = []
        strategies = []
        for col in sales_cols:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    total_sales = df[col].sum()
                    avg_sales = df[col].mean()
                    insights.append(f"Total {col}={total_sales:.2f}, Average {col}={avg_sales:.2f}")
                    if avg_sales < 1000:
                        strategies.append(f"Increase targeted marketing and promotions to improve {col}.")
                    else:
                        strategies.append(f"Maintain current performance but optimize underperforming segments in {col}.")
            except Exception:
                continue
        return insights, strategies

    def expense_analysis(df, expense_cols):
        insights = []
        strategies = []
        for col in expense_cols:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    avg_exp = df[col].mean()
                    insights.append(f"Average {col}={avg_exp:.2f}")
                    if avg_exp > 5000:
                        strategies.append(f"Investigate top cost drivers in {col} and implement cost-reduction initiatives.")
            except Exception:
                continue
        return insights, strategies

    def regional_analysis(df, region_cols, sales_cols):
        insights = []
        strategies = []
        try:
            region_col = region_cols[0]
            sales_col = sales_cols[0]
            if pd.api.types.is_numeric_dtype(df[sales_col]):
                grouped = df.groupby(region_col)[sales_col].sum().sort_values()
                if len(grouped) > 0:
                    low_reg = grouped.index[0]
                    high_reg = grouped.index[-1]
                    insights.append(f"Performance by {region_col}: Lowest={low_reg}, Highest={high_reg}")
                    strategies.append(f"Focus improvement efforts on {low_reg} region and replicate success factors from {high_reg}.")
        except Exception:
            pass
        return insights, strategies

    def default_strategy_fillers(strategies):
        default_fillers = [
            "Implement weekly dashboard monitoring for key performance indicators.",
            "Conduct monthly review of top 10 transactions for anomalies and patterns.",
            "Establish benchmark comparisons with industry standards or historical performance.",
            "Create automated alerts for performance thresholds and unusual variations.",
            "Develop action plans with clear ownership for each identified improvement area."
        ]
        return ensure_min_strategies(strategies, default_fillers)

    insights = []
    strategies = []

    try:
        # Basic info
        insights.extend(basic_info(df))

        # Numeric column analysis
        numeric_insights, _ = numeric_column_analysis(df)
        insights.extend(numeric_insights)

        # Look for common business columns (case-insensitive)
        sales_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'income'])]
        expense_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['expense', 'cost', 'spend'])]
        region_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['region', 'location', 'area', 'territory'])]

        # Sales analysis
        sales_insights, sales_strategies = sales_analysis(df, sales_cols)
        insights.extend(sales_insights)
        strategies.extend(sales_strategies)

        # Expense analysis
        expense_insights, expense_strategies = expense_analysis(df, expense_cols)
        insights.extend(expense_insights)
        strategies.extend(expense_strategies)

        # Regional analysis
        if region_cols and sales_cols:
            region_insights, region_strategies = regional_analysis(df, region_cols, sales_cols)
            insights.extend(region_insights)
            strategies.extend(region_strategies)

        # Generic strategies if we don't have enough
        strategies = default_strategy_fillers(strategies)

        return {"insights": insights, "strategies": strategies}

    except Exception as e:
        # Fallback analysis if anything goes wrong
        return {
            "insights": [
                f"Dataset contains {len(df)} rows and {len(df.columns)} columns",
                f"Analysis error occurred: {str(e)}",
                "Manual review recommended for detailed insights"
            ],
            "strategies": [
                "Review data quality and completeness",
                "Validate key performance metrics manually",
                "Establish baseline measurements for tracking",
                "Create regular monitoring and reporting processes",
                "Implement data validation and quality checks"
            ]
        }

# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="MISInsight-Pro (Local)", layout="wide")
st.title("MISInsight-Pro — Local Edition")
st.write("Upload your MIS report (CSV exported from Excel). The app will analyze and suggest strategies based only on the uploaded report.")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload MIS report (CSV)", type=["csv"])    
    example = st.checkbox("Use example demo dataset (if you don't have a file)")

with col2:
    backend = st.selectbox("Analysis Mode / Backend", ["Rule-based (offline)", "Ollama (local LLM)"])
    st.markdown("**Strict scope behavior:** If the user asks anything outside the uploaded MIS report, the assistant will reply with the fallback message.")
    if backend == "Ollama (local LLM)":
        st.write(f"Ollama endpoint: {OLLAMA_URL}")
        st.write(f"Model: {OLLAMA_MODEL}")

# Load demo dataset if requested
df = pd.DataFrame()
if example and uploaded_file is None:
    # create a small demo dataframe
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
    st.info("No report loaded yet. Upload a CSV (exported from Excel) or use the demo dataset to analyze.")
else:
    st.subheader("Report Preview")
    st.dataframe(df, use_container_width=True)
    
    with st.expander("Show Dataset Summary"):
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.write("**Data Types:**")
        st.write(df.dtypes)
        if not df.select_dtypes(include=['number']).empty:
            st.write("**Numeric Summary:**")
            st.write(df.describe())

    st.subheader("Analysis & Strategies")
    
    # Add debug info
    with st.expander("Debug Information"):
        st.write("DataFrame empty:", df.empty)
        st.write("DataFrame shape:", df.shape if not df.empty else "N/A")
        st.write("Backend selected:", backend)
    
    run_button = st.button("🔍 Analyze Report", type="primary")

    if run_button:
        if df.empty:
            st.error("No data to analyze. Please upload a CSV file or use the demo dataset.")
        else:
            with st.spinner("Generating insights and strategies..."):
                try:
                    summary = summarize_df(df, max_rows=8)
                    st.write("**Data Summary Generated:**", len(summary), "characters")

                    if backend == "Rule-based (offline)":
                        st.info("Using rule-based analysis...")
                        result = rule_based_analysis(df)
                        insights = result["insights"]
                        strategies = result["strategies"]

                    else:  # Ollama
                        st.info("Connecting to Ollama...")
                        prompt = SYSTEM_PROMPT + "\n\n" + LLM_INSTRUCTION_TEMPLATE.format(summary=summary)
                        llm_output = call_ollama(prompt)
                        
                        if llm_output.startswith("ERROR:"):
                            st.error(llm_output)
                            # Fallback to rule-based
                            st.info("Falling back to rule-based analysis...")
                            result = rule_based_analysis(df)
                            insights = result["insights"]
                            strategies = result["strategies"]
                        else:
                            try:
                                # Try to parse JSON
                                parsed = json.loads(llm_output)
                                insights = parsed.get("insights", [])
                                strategies = parsed.get("strategies", [])
                            except json.JSONDecodeError:
                                st.warning("Ollama output couldn't be parsed as JSON. Using simple text parsing.")
                                lines = [l.strip() for l in llm_output.splitlines() if l.strip()]
                                insights = lines[:5] if len(lines) >= 5 else lines
                                strategies = lines[5:10] if len(lines) >= 10 else lines

                    # Ensure minimum strategies
                    default_fillers = [
                        "Implement regular performance monitoring and KPI tracking.",
                        "Establish data quality validation processes.",
                        "Create monthly business review meetings with stakeholders.",
                        "Develop automated reporting for key business metrics.",
                        "Set up alerts for performance threshold breaches."
                    ]
                    strategies = ensure_min_strategies(strategies, default_fillers)

                    # Store results in session state
                    st.session_state.analysis_results = {
                        "insights": insights,
                        "strategies": strategies
                    }

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error(f"Full error: {traceback.format_exc()}")

    # Display results if available
    if st.session_state.analysis_results:
        insights = st.session_state.analysis_results["insights"]
        strategies = st.session_state.analysis_results["strategies"]
        
        # Present results
        st.subheader("📊 Key Insights")
        if insights:
            for i, ins in enumerate(insights, 1):
                st.write(f"**{i}.** {ins}")
        else:
            st.write("No specific insights were generated from the data.")

        st.subheader("🎯 Actionable Strategies")
        for i, s in enumerate(strategies, 1):
            st.write(f"**{i}.** {s}")

        # Download functionality
        if st.button("📥 Download Results"):
            result_blob = {
                "insights": insights,
                "strategies": strategies,
                "analysis_timestamp": str(pd.Timestamp.now()),
                "data_shape": list(df.shape),
                "columns": list(df.columns)
            }
            
            # Create downloadable content
            download_content = "MIS ANALYSIS RESULTS\n"
            download_content += "=" * 50 + "\n\n"
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
                mime="text/plain"
            )
            
            st.download_button(
                "Download as JSON", 
                json.dumps(result_blob, indent=2), 
                file_name=f"mis_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

    # Chat-like Q/A about the report (scope-limited)
    if not df.empty:
        st.subheader("💬 Ask About Your Report")
        user_q = st.text_input("Ask a question about the uploaded MIS report only")
        if user_q:
            # Simple scope checking
            lower = user_q.lower()
            unrelated_keywords = ["weather", "movie", "football", "politics", "love", "recipe", "how to make", "what is python", "help me code"]
            if any(k in lower for k in unrelated_keywords):
                st.error(FALLBACK_MESSAGE)
            else:
                prompt_context = summarize_df(df, max_rows=6)
                if backend == "Rule-based (offline)":
                    st.info("Using rule-based Q&A...")
                    # Simple keyword matching
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
                        st.write("I can only provide information directly available in your uploaded report data.")

                else:  # Ollama
                    qa_prompt = SYSTEM_PROMPT + "\n\nReport summary:\n" + prompt_context + "\n\nQuestion: " + user_q + "\n\nProvide a concise answer based only on the report data."
                    with st.spinner("Generating answer..."):
                        ans = call_ollama(qa_prompt)
                        if ans.startswith("ERROR:"):
                            st.error(ans)
                        else:
                            st.write(ans)

# Sidebar
st.sidebar.header("About")
st.sidebar.write("**MISInsight-Pro — Local Edition**")
st.sidebar.write("Created by Shaun Mathew to replicate PartyRock MISInsight-Pro functionality locally.")
st.sidebar.write("This app strictly scopes responses to the uploaded MIS report.")

st.sidebar.header("Troubleshooting")
st.sidebar.write("**If the Analyze button doesn't work:**")
st.sidebar.write("1. Check if your CSV file uploaded correctly")
st.sidebar.write("2. Try the demo dataset first")
st.sidebar.write("3. For Ollama: ensure it's running at the specified URL")
st.sidebar.write("4. Check the Debug Information section")

st.sidebar.header("Configuration")
st.sidebar.write(f"**Ollama URL:** {OLLAMA_URL}")
st.sidebar.write(f"**Ollama Model:** {OLLAMA_MODEL}")
st.sidebar.write("Set environment variables OLLAMA_URL and OLLAMA_MODEL to customize.")