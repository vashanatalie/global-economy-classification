# global-economy-classification
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Sample data: GDP per capita, inflation, and trade balance for different countries
data = {
    "Country": ["USA", "China", "Germany", "India", "Brazil", "UK", "France", "Russia", "Japan", "South Africa"],
    "GDP_per_capita": [65000, 10000, 50000, 2500, 9000, 47000, 45000, 12000, 40000, 7000],
    "Inflation": [2.5, 1.8, 2.0, 5.2, 4.1, 1.9, 1.8, 5.0, 0.5, 5.7],
    "Trade_Balance": [-500, 300, 250, -100, -50, -150, 50, 200, 150, -80]
}

# Create DataFrame
df = pd.DataFrame(data)

# Feature selection and scaling
features = df[["GDP_per_capita", "Inflation", "Trade_Balance"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(df["GDP_per_capita"], df["Inflation"], c=df["Cluster"], cmap="viridis", edgecolors="k")
plt.xlabel("GDP per Capita ($)")
plt.ylabel("Inflation (%)")
plt.title("Classification of Global Economies")
plt.colorbar(label="Cluster Group")
for i, txt in enumerate(df["Country"]):
    plt.annotate(txt, (df["GDP_per_capita"][i], df["Inflation"][i]), fontsize=9, ha="right")
plt.show()

# Display classification results
print(df)
fre
