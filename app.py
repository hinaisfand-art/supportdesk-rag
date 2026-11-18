"""Streamlit interface for the SupportDesk policy assistant."""
import os
from pathlib import Path

import streamlit as st

from rag import PolicyIndex, draft, ingest

ROOT = Path(__file__).resolve().parent
EXAMPLE = "My order arrived 12 days ago. It’s unused and still in its original packaging. Can I return it?"

st.set_page_config(page_title="SupportDesk | Policy workspace", page_icon=":material/support_agent:", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 2.2rem;}
.hero {background: linear-gradient(115deg,#102830,#15584f); color: #fff;
       border-radius: 20px; padding: 32px 36px; margin-bottom: 24px;}
.hero p {color: #def5eb; margin: 8px 0 0;}
.hero strong {font-size: 2.5rem; letter-spacing: -.065em;}
[data-testid="stMetric"] {padding: 8px 0;}
</style>
<div class="hero"><strong>SupportDesk</strong>
<p>Find the right policy, check the evidence, and prepare a customer reply.</p></div>
""", unsafe_allow_html=True)
api_key = st.secrets["OPENAI_API_KEY"]
model = st.secrets["OPENAI_MODEL"]
os.environ["OPENAI_BASE_URL"] = st.secrets["OPENAI_BASE_URL"]

with st.sidebar:
    st.header("Knowledge base")
    mode = st.radio("Policy documents", ["Sample store policies", "Upload my policies"])
    uploads = []
    if mode == "Upload my policies":
        uploads = st.file_uploader("Add searchable PDFs or UTF-8 text files",
                                   type=["pdf", "txt", "md"], accept_multiple_files=True)
        if not uploads:
            st.info("Add at least one document to get started.")
    st.divider()
    st.caption("Drafting mode")
    st.write("OpenAI" if api_key else "Local policy excerpt")
    if api_key:
        st.caption("The ticket and retrieved passages are sent to OpenAI when you generate a draft.")
    else:
        st.caption("Works without an API key. Drafts quote the closest policy passage.")

if mode == "Sample store policies":
    documents = [("sample_policies.txt", (ROOT / "sample_policies.txt").read_bytes())]
else:
    documents = [(f.name, f.getvalue()) for f in uploads]

# Keep uploaded policy text and the index in this browser session.
if st.session_state.get("documents") != documents:
    passages, errors = [], []
    for number, (name, content) in enumerate(documents, 1):
        label = f"{name} (file {number})" if sum(n == name for n, _ in documents) > 1 else name
        try:
            extracted = ingest(name, content, label=label)
            if not extracted:
                errors.append(f"{label} has no extractable text. Try a text export or OCR.")
            passages.extend(extracted)
        except Exception:
            errors.append(f"Couldn’t read {label}. Check for encryption, invalid text, or a PDF over 100 pages.")
    try:
        index = PolicyIndex(passages) if passages else None
    except ValueError:
        index = None
        errors.append("The documents contain too little searchable text.")
    st.session_state.update(documents=documents, index=index, errors=errors, result=None)
    st.session_state.pop("edited_reply", None)

index = st.session_state.index
for message in st.session_state.errors:
    st.warning(message)

left, right = st.columns([1.05, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Customer ticket")
        if st.button("Load example ticket", disabled=mode != "Sample store policies"):
            st.session_state.ticket = EXAMPLE
        ticket = st.text_area("Customer message", key="ticket", height=165, max_chars=3000,
                              placeholder="Paste the customer’s message here.")
        if st.button("Find policy and draft reply", type="primary",
                     disabled=index is None or not ticket.strip()):
                    with st.spinner("Checking the policy documents…"):
                                    evidence = index.retrieve(ticket)
                                    try:
                                        result = draft(ticket, evidence, api_key, model)
                                        fallback = False
                                    except Exception as error:
                                        st.error(
                                            f"AI error: {type(error).__name__} | "
                                            f"Status: {getattr(error, 'status_code', 'unknown')} | "
                                            f"Code: {getattr(error, 'code', 'unknown')}"
                                        )
                                        result = draft(ticket, evidence)
                                        fallback = True
                                    st.session_state.result = (ticket, evidence, result, fallback)
                                    st.session_state.pop("edited_reply", None)  
    with st.container(border=True):
        docs_metric, passages_metric = st.columns(2)
        docs_metric.metric("Indexed documents", len({p.source for p in index.passages}) if index else 0)
        passages_metric.metric("Policy passages", len(index.passages) if index else 0)

with right:
    result = st.session_state.result
    with st.container(border=True):
        st.subheader("Resolution")
        if result is None or result[0] != ticket:
            st.caption("Run the current ticket to see a draft and its supporting policy.")
        else:
            _, evidence, (status, decision, reply), fallback = result
            if fallback:
                st.warning("OpenAI drafting failed. Showing a local draft. Check the key, model, and API access.")
            if status == "Needs review":
                st.warning(status)
            else:
                st.info(status)
            st.text(decision)
            edited = st.text_area("Edit customer reply", value=reply, key="edited_reply", height=220)
            st.download_button("Download reply", edited, "customer_reply.txt", "text/plain")
    if result is not None and result[0] == ticket and result[1]:
        st.subheader("Policy evidence")
        for rank, passage in enumerate(result[1], 1):
            with st.expander(f"[{rank}] {passage.label}", expanded=rank == 1):
                st.text(passage.text)
