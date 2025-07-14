import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from openai import OpenAI

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="AI Data Analyst Assistant", layout="wide")
st.title("🧠 AI Data Analyst Assistant")

# API Key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Fix duplicate columns
    if df.columns.duplicated().any():
        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
        st.warning("⚠️ Duplicate column names found. Renamed automatically.")

    # Rename Unnamed: 0 if it's likely an index
    if 'Unnamed: 0' in df.columns:
        if df['Unnamed: 0'].is_unique and df['Unnamed: 0'].is_monotonic_increasing:
            df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
        else:
            st.warning("⚠️ 'Unnamed: 0' detected but not a proper index. Kept as-is.")

    # =============================
    # PREVIEW
    # =============================
    st.subheader("👀 Preview of Your Data")
    st.dataframe(df.head(10), use_container_width=True)
    st.code(f"Columns loaded: {df.columns.tolist()}")

    # =============================
    # BASIC DATASET OVERVIEW
    # =============================
    st.subheader("📦 Basic Dataset Overview")
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    total_missing = df.isnull().sum().sum()
    type_counts = df.dtypes.value_counts()
    cols_by_type = {str(dtype): df.select_dtypes(include=[dtype]).columns.tolist() for dtype in df.dtypes.unique()}

    st.code(f"""Dataset Summary:
- Rows: {total_rows}
- Columns: {total_cols}
- Missing Values: {total_missing}
- Column Types: {type_counts.to_dict()}
""")
    for dtype, cols in cols_by_type.items():
        st.markdown(f"**🔹 {dtype} ({len(cols)} columns):** `{', '.join(cols)}`")

    # =============================
    # AI SUMMARY
    # =============================
    st.subheader("🤖 AI-Powered Insight Summary")
    if st.button("🔍 Generate Summary"):
        sample_df = df.sample(min(10, len(df)), random_state=1).to_csv(index=False)
        with st.spinner("Asking GPT..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst."},
                        {"role": "user", "content": f"Here's the data sample:\n\n{sample_df}\n\nPlease summarize the dataset in 3 bullet points."}
                    ]
                )
                summary = response.choices[0].message.content
                st.success("✅ Summary generated!")
                st.markdown(summary)
            except Exception as e:
                st.error(f"❌ Error generating summary: {e}")

    # =============================
    # ASK GPT ANYTHING
    # =============================
    st.subheader("💬 Ask Anything About Your Data")
    user_question = st.text_input("Enter your question (e.g., Which product is most profitable?)")
    if st.button("📊 Ask Now") and user_question:
        sample_df = df.sample(min(10, len(df)), random_state=2).to_csv(index=False)
        with st.spinner("Generating answer..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analyst assistant."},
                        {"role": "user", "content": f"Dataset sample:\n\n{sample_df}\n\nQuestion: {user_question}"}
                    ]
                )
                answer = response.choices[0].message.content
                st.success("✅ Answer generated!")
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")

    # =============================
    # AUTO CHART
    # =============================
    st.subheader("📈 Auto Chart Generator")
    x_col = st.selectbox("Select X-axis", df.columns, key="x")
    y_col = st.selectbox("Select Y-axis", df.select_dtypes(include=["number"]).columns, key="y")
    top_n = st.slider("Top N records to show (based on Y-axis)", 5, 100, 30)

    if x_col and y_col:
        if x_col == y_col:
            st.warning("⚠️ X and Y axis use the same column. Consider using different columns.")
        else:
            chart_df = df[[x_col, y_col]].dropna()

            if not chart_df[y_col].is_unique:
                try:
                    chart_df = chart_df.groupby(x_col, as_index=False).mean(numeric_only=True)
                except Exception as e:
                    st.error(f"❌ Error during groupby: {e}")
                    st.stop()

            try:
                chart_df = chart_df.sort_values(by=y_col, ascending=False).head(top_n)
                st.markdown(f"**Bar Chart: {y_col} by {x_col} (Top {top_n})**")
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.bar(chart_df[x_col].astype(str), chart_df[y_col])
                plt.xticks(rotation=90)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Chart failed to render: {e}")
