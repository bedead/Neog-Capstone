import streamlit as st
import pandas as pd
import joblib  # To load your trained model

# Load your Excel data
excel_path = "Weather Data.xlsx"
weather_data = pd.read_excel(excel_path)

# Load the trained model and label encoder
model = joblib.load("model_rfc.pkl")
labelencoder_y = joblib.load("label_encoder.pkl")

# Apply a background style
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
        color: #333333;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(to right, #4CAF50, #81C784);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the web app
st.title("🌾 Crop Recommender System 🌾")

# Subheader for user guidance
st.subheader(
    "Provide soil and environmental details to get the best crop recommendations!"
)

# Advanced instructions
with st.expander("ℹ️ Instructions:", expanded=False):
    st.write(
        """
        - The month and state selection is optional but can auto-fill temperature, humidity, and rainfall.
        - Please fill in all required fields and press 'Predict Crop' to get a recommendation.
    """
    )

# Sidebar dropdowns for month and state selection
st.sidebar.header("Select Weather Data")

months = [
    "Select a Month",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
states = ["Select a State"] + list(weather_data["State"].unique())

selected_month = st.sidebar.selectbox("📅 Select Month:", months, index=0)
selected_state = st.sidebar.selectbox("🌍 Select State:", states, index=0)

# Initialize defaults
avg_temp = avg_humidity = avg_rainfall = 0.0

# Filter the weather data if selections are made
if selected_month != "Select a Month" and selected_state != "Select a State":
    filtered_data = weather_data[
        (weather_data["State"] == selected_state)
        & (weather_data["Month"].str.startswith(selected_month[:3]))
    ]

    if not filtered_data.empty:
        avg_temp = float(filtered_data["Daily Mean (°C)"].values[0])
        avg_humidity = float(filtered_data["Avg Relative Humidity (%)"].values[0])
        avg_rainfall = float(filtered_data["Avg Precipitation (mm)"].values[0])

# Soil and environmental inputs
st.header("Enter Soil and Environmental Details")
col1, col2, col3 = st.columns(3)

# Row 1: Nitrogen, Phosphorus, Potassium
with col1:
    N = st.number_input(
        "🧪 Nitrogen (N):", min_value=0, max_value=200, step=1, key="nitrogen"
    )
with col2:
    P = st.number_input(
        "🧪 Phosphorus (P):", min_value=0, max_value=200, step=1, key="phosphorus"
    )
with col3:
    K = st.number_input(
        "🧪 Potassium (K):", min_value=0, max_value=200, step=1, key="potassium"
    )

# Row 2: Temperature, Humidity, pH level
with col1:
    temperature = st.number_input(
        "🌡️ Temperature (°C):",
        value=avg_temp,
        max_value=50.0,
        step=0.1,
        key="temperature",
    )
with col2:
    humidity = st.number_input(
        "💧 Humidity (%):",
        value=avg_humidity,
        max_value=100.0,
        step=0.1,
        key="humidity",
    )
with col3:
    ph = st.number_input(
        "🧪 pH level:", min_value=0.0, max_value=14.0, step=0.1, key="ph"
    )

# Row 3: Rainfall
with col1:
    rainfall = st.number_input(
        "🌧️ Rainfall (mm):",
        value=avg_rainfall,
        max_value=500.0,
        step=1.0,
        key="rainfall",
    )

# Button to predict the crop recommendation
if st.button("🔍 Predict Crop"):
    # Fields to validate
    fields = {
        "Nitrogen (N)": N,
        "Phosphorus (P)": P,
        "Potassium (K)": K,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "pH level": ph,
        "Rainfall (mm)": rainfall,
    }

    # Check for empty or zero fields
    missing_fields = [
        field for field, value in fields.items() if value is None or value == 0
    ]

    if missing_fields:
        st.error(f"⚠️ Please fill in the following fields: {', '.join(missing_fields)}")
    else:
        # Create a DataFrame from user input
        user_data = pd.DataFrame(
            {
                "N": [N],
                "P": [P],
                "K": [K],
                "temperature": [temperature],
                "humidity": [humidity],
                "ph": [ph],
                "rainfall": [rainfall],
            }
        )

        # Make predictions
        prediction = model.predict(user_data)

        # Decode the predicted label back to the crop name
        crop_name = labelencoder_y.inverse_transform(prediction)

        # Display the result
        st.markdown("---")
        st.markdown("### 🎉 Recommended Crop:")
        st.success(f"**{crop_name[0]}**")
