import streamlit as st

st.set_page_config(
    page_title="AI Customer Support",
    layout="centered"
)

try:
    from src.utils import run_customer_support
except Exception as e:
    st.error(f"Failed to load support system: {e}")
    st.stop()

st.title("Customer Support System")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="Type your question here..."
)

if st.button("Submit", type="primary", use_container_width=True):
    if query.strip():
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
                st.error(f"Error: {str(e)}")
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