st.subheader("Numeric Column Distributions")
num_cols = df.select_dtypes("number").columns.tolist()
for col in num_cols[:10]:  # limit for speed
    st.write(f"Histogram: {col}")
    st.bar_chart(df[col].dropna().value_counts().sort_index())  # quick bin-free view

st.subheader("Correlation Heatmap")
import seaborn as sns, matplotlib.pyplot as plt
corr = df[num_cols].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, ax=ax)
st.pyplot(fig)