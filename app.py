import logging

import streamlit as st

from src.input_validation import validate_query

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Customer Support",
    layout="centered"
)

try:
    from src.utils import run_customer_support
except Exception as e:
    logger.exception("Failed to load support system")
    st.error("The support system is temporarily unavailable. Please try again later.")
    st.stop()

st.title("Telecom Customer Support Agent")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="Type your question here..."
)

if st.button("Submit", type="primary", use_container_width=True):
    if query.strip():
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            st.warning(error_msg)
        else:
            with st.spinner("Processing..."):
                try:
                    result = run_customer_support(query)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Category", result["category"])
                    with col2:
                        st.metric("Sentiment", result["sentiment"])

                    if result["sentiment"].lower() == "negative":
                        st.warning("Negative sentiment detected")

                    st.subheader("Response:")

                    if "human agent will follow up" in result["response"].lower():
                        st.error(result["response"])
                    else:
                        st.info(result["response"])

                    st.session_state.history.append({
                        "query": query,
                        "category": result["category"],
                        "sentiment": result["sentiment"],
                        "response": result["response"]
                    })

                except Exception as e:
                    logger.exception("Error processing customer query")
                    st.error("Something went wrong processing your request. Please try again.")
    else:
        st.warning("Please enter a query")

if st.session_state.history:
    st.divider()
    st.subheader("Previous Questions")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Q{i}: {item['query'][:60]}..."):
            st.write(f"**Category:** {item['category']}")
            st.write(f"**Sentiment:** {item['sentiment']}")
            st.write(f"**Response:** {item['response']}")