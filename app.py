import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------
# Load Data
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_sales.csv")

    # Convert to standard column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Convert date column
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

    # Calculate sales = price * quantity
    df["sales"] = df["price"] * df["quantity"]

    return df


df = load_data()

# -----------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------
st.sidebar.title("Filters")

# Date range filter
start_date = df["date"].min()
end_date = df["date"].max()
date_range = st.sidebar.date_input("Date Range", (start_date, end_date))
df = df[(df["date"] >= pd.to_datetime(date_range[0])) &
        (df["date"] <= pd.to_datetime(date_range[1]))]

# City filter
cities = ["All"] + sorted(df["city"].unique().tolist())
selected_city = st.sidebar.selectbox("City", cities)
if selected_city != "All":
    df = df[df["city"] == selected_city]

# Product filter
products = ["All"] + sorted(df["product"].unique().tolist())
selected_product = st.sidebar.selectbox("Product", products)
if selected_product != "All":
    df = df[df["product"] == selected_product]

# -----------------------------------------------------------
# KPIs
# -----------------------------------------------------------
st.title("🍽️ Restaurant Sales Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{df['sales'].sum():,.2f}")
col2.metric("Total Orders", df.shape[0])
col3.metric("Avg Order Value", f"{df['sales'].mean():,.2f}")

st.markdown("---")

# -----------------------------------------------------------
# 1. Sales Over Time (Line Chart)
# -----------------------------------------------------------
daily_sales = df.groupby("date", as_index=False)["sales"].sum()
fig1 = px.line(daily_sales, x="date", y="sales", title="1️⃣ Sales Over Time")
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------------------------------------
# 2. Monthly Sales Trend (Bar Chart)
# -----------------------------------------------------------
df["month"] = df["date"].dt.to_period("M").astype(str)
monthly_sales = df.groupby("month", as_index=False)["sales"].sum()
fig2 = px.bar(monthly_sales, x="month", y="sales", title="2️⃣ Monthly Sales Trend")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------
# 3. Sales by City
# -----------------------------------------------------------
sales_city = df.groupby("city", as_index=False)["sales"].sum()
fig3 = px.bar(sales_city, x="city", y="sales", title="3️⃣ Sales by City")
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------------
# 4. Sales by Product Category
# -----------------------------------------------------------
sales_product = df.groupby("product", as_index=False)["sales"].sum()
fig4 = px.bar(sales_product, x="product", y="sales", title="4️⃣ Sales by Product")
st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------
# 5. Top 10 Best Selling Items (Quantity)
# -----------------------------------------------------------
top_items = df.groupby("product", as_index=False)["quantity"].sum().nlargest(10, "quantity")
fig5 = px.bar(top_items, x="quantity", y="product", orientation="h",
              title="5️⃣ Top 10 Best-Selling Items (Quantity)")
st.plotly_chart(fig5, use_container_width=True)

# -----------------------------------------------------------
# 6. Day of Week Sales
# -----------------------------------------------------------
df["day"] = df["date"].dt.day_name()
dow = df.groupby("day", as_index=False)["sales"].sum()

# Order days correctly
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow["day"] = pd.Categorical(dow["day"], categories=day_order, ordered=True)
dow = dow.sort_values("day")

fig6 = px.bar(dow, x="day", y="sales", title="6️⃣ Sales by Day of Week")
st.plotly_chart(fig6, use_container_width=True)

# -----------------------------------------------------------
# Raw Data
# -----------------------------------------------------------
with st.expander("Show Raw Data"):
    st.dataframe(df)
# -----------------------------------------------------------
# 📈 MACHINE LEARNING PREDICTIONS SECTION
# -----------------------------------------------------------
st.header("📈 Machine Learning Predictions")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Sales Forecasting (Prophet)",
    "💰 Price Optimization (Regression)",
    "🧠 Product Recommendation (Random Forest)",
    "🔥 Peak Sales Day Prediction"
])

# -----------------------------------------------------------
# 🔮 1. SALES FORECASTING (PROPHET)
# -----------------------------------------------------------
with tab1:
    st.subheader("30-Day Sales Forecast")

    try:
        from prophet import Prophet

        sales_daily = df.groupby("date", as_index=False)["sales"].sum()
        prophet_df = sales_daily.rename(columns={"date": "ds", "sales": "y"})

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig = px.line(forecast, x="ds", y="yhat",
                      title="📅 30-Day Sales Forecast")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prophet error: {e}")
        st.info("Install Prophet using: pip install prophet")


# -----------------------------------------------------------
# 💰 2. PRICE OPTIMIZATION (REGRESSION)
# -----------------------------------------------------------
with tab2:
    st.subheader("Price Optimization (Regression)")

    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = df[["quantity"]].values
    y = df["price"].values

    reg = LinearRegression()
    reg.fit(X, y)

    qty_range = np.linspace(df["quantity"].min(), df["quantity"].max(), 50).reshape(-1, 1)
    pred_line = reg.predict(qty_range)

    fig_reg = px.scatter(df, x="quantity", y="price", title="Price vs Quantity")
    fig_reg.add_scatter(x=qty_range.flatten(), y=pred_line, mode="lines", name="Predicted Line")

    st.plotly_chart(fig_reg, use_container_width=True)

    st.write(f"### Model: Price = {reg.intercept_:.2f} + {reg.coef_[0]:.2f} × Quantity")


# -----------------------------------------------------------
# 🧠 3. PRODUCT RECOMMENDATION (RANDOM FOREST)
# -----------------------------------------------------------
with tab3:
    st.subheader("Product Recommendation")

    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier

    rec_df = df.copy()
    rec_df["day"] = rec_df["date"].dt.day_name()

    features = ["day", "city", "purchase_type", "payment_method"]
    target = "product"

    # Encode categorical
    enc = LabelEncoder()
    for col in features + [target]:
        rec_df[col] = enc.fit_transform(rec_df[col].astype(str))

    X = rec_df[features]
    y = rec_df[target]

    # Train model
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # User Inputs
    day_input = st.selectbox("Select Day", sorted(df["date"].dt.day_name().unique()))
    city_input = st.selectbox("Select City", sorted(df["city"].unique()))
    purchase_input = st.selectbox("Purchase Type", sorted(df["purchase_type"].unique()))
    payment_input = st.selectbox("Payment Method", sorted(df["payment_method"].unique()))

    # Encode inputs
    inp = pd.DataFrame({
        "day": [day_input],
        "city": [city_input],
        "purchase_type": [purchase_input],
        "payment_method": [payment_input]
    })

    for col in inp.columns:
        inp[col] = enc.fit_transform(inp[col].astype(str))

    pred = clf.predict(inp)[0]

    # Decode prediction
    rec_product = df["product"].unique()[pred]

    st.success(f"Recommended Product: **{rec_product}**")


# -----------------------------------------------------------
# 🔥 4. PEAK SALES DAY PREDICTION (DECISION TREE)
# -----------------------------------------------------------
with tab4:
    st.subheader("Peak Sales Day Predictor")

    from sklearn.tree import DecisionTreeRegressor

    peak_df = df.copy()
    peak_df["day"] = peak_df["date"].dt.day_name()

    day_map = {"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
    peak_df["day_num"] = peak_df["day"].map(day_map)

    X = peak_df[["day_num"]]
    y = peak_df["sales"]

    tree = DecisionTreeRegressor()
    tree.fit(X, y)

    # Predict sales for each day
    pred_days = pd.DataFrame({
        "day": list(day_map.keys()),
        "day_num": list(day_map.values())
    })

    pred_days["predicted_sales"] = tree.predict(pred_days[["day_num"]])

    fig_peak = px.bar(pred_days, x="day", y="predicted_sales",
                      title="🔥 Predicted Busiest Sales Days (ML Based)")
    st.plotly_chart(fig_peak, use_container_width=True)

    max_day = pred_days.loc[pred_days["predicted_sales"].idxmax(), "day"]
    st.success(f"📈 Predicted Busiest Day: **{max_day}**")
