import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="centered"
)

# ---------------------------------------------------
# Load Saved Models
# ---------------------------------------------------
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('cluster_labels_map.pkl', 'rb') as f:
        labels_map = pickle.load(f)
    with open('product_similarity.pkl', 'rb') as f:
        product_similarity_df = pickle.load(f)
    return kmeans_model, scaler, labels_map, product_similarity_df

kmeans_model, scaler, labels_map, product_similarity_df = load_models()

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.title("🛒 Shopper Spectrum")
st.markdown("##### Customer Segmentation & Product Recommendations in E-Commerce")

module = st.sidebar.radio(
    "Choose a Module",
    ["🎯 Product Recommendation", "🔍 Customer Segmentation"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses RFM-based KMeans clustering to segment customers, "
    "and item-based collaborative filtering to recommend similar products."
)

# ---------------------------------------------------
# Recommendation Function
# ---------------------------------------------------
def recommend_products(product_name, similarity_df, top_n=5):
    if product_name not in similarity_df.index:
        matches = [p for p in similarity_df.index if product_name.upper() in p.upper()]
        if not matches:
            return None, None
        product_name = matches[0]

    similar_scores = similarity_df[product_name].sort_values(ascending=False)
    similar_scores = similar_scores.drop(product_name)
    top_products = similar_scores.head(top_n)
    return product_name, top_products

# ---------------------------------------------------
# MODULE 1: Product Recommendation
# ---------------------------------------------------
if module == "🎯 Product Recommendation":
    st.header("🎯 Product Recommendation Module")
    st.write("Enter a product name to get 5 similar product recommendations based on customer purchase patterns.")

    product_input = st.text_input("Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")

    if st.button("Get Recommendations"):
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            matched_name, recommendations = recommend_products(product_input, product_similarity_df, top_n=5)

            if recommendations is None:
                st.error(f"No matching product found for '{product_input}'. Try a different keyword.")
            else:
                st.success(f"Showing recommendations for: **{matched_name}**")
                st.markdown("### Recommended Products")

                for i, (prod, score) in enumerate(recommendations.items(), start=1):
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="
                                padding: 12px 16px;
                                margin-bottom: 8px;
                                border-radius: 10px;
                                background-color: #f5f7fa;
                                border-left: 5px solid #4a90d9;
                            ">
                                <b>{i}. {prod}</b><br>
                                <span style="color: gray; font-size: 0.85em;">Similarity Score: {score:.3f}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

# ---------------------------------------------------
# MODULE 2: Customer Segmentation
# ---------------------------------------------------
else:
    st.header("🔍 Customer Segmentation Module")
    st.write("Enter the customer's Recency, Frequency, and Monetary values to predict their segment.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (in days)", min_value=0, value=30, step=1)
    with col2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5, step=1)
    with col3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, step=10.0)

    if st.button("Predict Cluster"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans_model.predict(input_scaled)[0]
        segment = labels_map[cluster]

        segment_colors = {
            "High-Value": "#2ecc71",
            "Regular": "#3498db",
            "Occasional": "#f39c12",
            "At-Risk": "#e74c3c"
        }
        segment_descriptions = {
            "High-Value": "Recent, frequent, and big-spending customers. Prioritize retention and loyalty rewards.",
            "Regular": "Steady purchasers who are not yet premium. Good candidates for upselling.",
            "Occasional": "Rare, occasional buyers. Consider engagement campaigns to increase frequency.",
            "At-Risk": "Haven't purchased in a long time. Target with win-back offers."
        }

        color = segment_colors.get(segment, "#999999")
        description = segment_descriptions.get(segment, "")

        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 12px;
                background-color: {color}22;
                border: 2px solid {color};
                text-align: center;
            ">
                <h2 style="color: {color}; margin-bottom: 5px;">{segment}</h2>
                <p style="color: #333;">{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Shopper Spectrum | E-Commerce Customer Segmentation & Product Recommendation System")
