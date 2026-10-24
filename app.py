import streamlit as st

from mlp_model import SHELF_LIFE_DAYS, predict_risk, train_model


st.set_page_config(page_title="PantryPulse", page_icon="🥬", layout="centered")

st.title("PantryPulse")
st.caption("A small MLP-powered food waste risk demo")
st.write("Estimate whether an item may spoil before your household uses it.")


@st.cache_resource
def load_model():
    return train_model()


with st.form("item-details"):
    food_type = st.selectbox("Food", list(SHELF_LIFE_DAYS))
    days_since_purchase = st.slider("Days since purchase", 0, 14, 2)
    quantity = st.number_input("Quantity (servings)", min_value=1.0, max_value=12.0, value=4.0)
    household_size = st.number_input("People in household", min_value=1, max_value=6, value=2)
    servings_per_person_weekly = st.slider(
        "Typical servings per person each week", 0.2, 7.0, 3.0, step=0.2
    )
    submitted = st.form_submit_button("Estimate risk", type="primary")

if submitted:
    model = load_model()
    risk = predict_risk(
        model,
        food_type,
        days_since_purchase,
        quantity,
        household_size,
        servings_per_person_weekly,
    )
    st.metric("Estimated spoilage risk", f"{risk:.0%}")
    if risk >= 0.65:
        st.warning("Use this item soon, or plan a meal around it.")
    elif risk >= 0.35:
        st.info("Keep an eye on this item and include it in an upcoming meal.")
    else:
        st.success("This item is likely to be used before its estimated shelf life.")

st.caption(
    "Demo only: the model uses synthetic data and simplified shelf-life assumptions. "
    "It does not assess whether food is safe to eat."
)