import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def extract_values(llm: ChatOpenAI, report: str) -> str:
    prompt = f"""
You are a medical data extraction assistant.

From the laboratory report, extract all test values and classify each one as high, low, or normal
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{report}
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else response.text


def generate_summary_and_diet(llm: ChatOpenAI, extraction_values: str) -> tuple[str, str]:
    prompt = f"""
You are a clinical nutritionist specializing in Ethiopian dietary habits.

Based on the lab results, write:
1. A short health summary in 4-5 lines explaining the patient's condition in simple language
2. A short, practical Ethiopian diet plan having only two sections (1) Foods to eat (2) Foods to avoid.
Do not include any other sections in the diet plan.

Use exactly these section headers:
### Health Summary
### Ethiopian Diet Plan

Blood Work Analysis:
{extraction_values}
"""
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else response.text
    return split_summary_and_diet(text)


def split_summary_and_diet(text: str) -> tuple[str, str]:
    summary = ""
    diet = ""

    if "### Health Summary" in text:
        after_summary = text.split("### Health Summary", 1)[1]
        if "### Ethiopian Diet Plan" in after_summary:
            summary_part, diet_part = after_summary.split("### Ethiopian Diet Plan", 1)
            summary = summary_part.strip()
            diet = diet_part.strip()
        else:
            summary = after_summary.strip()
    else:
        summary = text.strip()

    return summary, diet


def analyze_report(report: str) -> tuple[str, str, str]:
    llm = get_llm()
    extraction_values = extract_values(llm, report)
    health_summary, diet_plan = generate_summary_and_diet(llm, extraction_values)
    return extraction_values, health_summary, diet_plan


st.set_page_config(page_title="Lab Report Analyzer", layout="wide")

st.title("Healthcare Lab Report Analyzer")
st.caption(
    "Paste a laboratory report to extract values and generate a health summary "
    "with an Ethiopian diet plan."
)

if "extraction_values" not in st.session_state:
    st.session_state.extraction_values = ""
if "health_summary" not in st.session_state:
    st.session_state.health_summary = ""
if "diet_plan" not in st.session_state:
    st.session_state.diet_plan = ""

left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("Lab Report")
    report_text = st.text_area(
        "Paste lab work text",
        height=520,
        placeholder="Paste your laboratory report here...",
        label_visibility="collapsed",
    )
    analyze_clicked = st.button("Analyze Report", type="primary", use_container_width=True)

with right_col:
    st.subheader("Health Summary")
    st.text_area(
        "Health summary output",
        value=st.session_state.health_summary,
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

    st.subheader("Suggested Diet Plan")
    st.text_area(
        "Diet plan output",
        value=st.session_state.diet_plan,
        height=280,
        disabled=True,
        label_visibility="collapsed",
    )

if analyze_clicked:
    if not report_text.strip():
        st.warning("Please paste a lab report before analyzing.")
    else:
        with st.spinner("Analyzing lab report..."):
            try:
                extraction, summary, diet = analyze_report(report_text)
                st.session_state.extraction_values = extraction
                st.session_state.health_summary = summary
                st.session_state.diet_plan = diet
                st.rerun()
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

if st.session_state.extraction_values:
    with st.expander("Extracted test values"):
        st.text(st.session_state.extraction_values)
