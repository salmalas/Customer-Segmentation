# Customer Segmentation

A K-Means clustering project that segments retail customers into five behavioral groups based on demographics, spending, and purchase-channel activity. The trained model is deployed as an interactive Streamlit app that predicts a customer's segment from a handful of manually entered details.

## Dataset

The data (`customer_segmentation.csv`) contains 2,240 customer records with 29 columns from a marketing campaign dataset, including:

- **Demographics** — `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`
- **Purchase history** — `Recency`, spending by category (`MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`), and purchases by channel (`NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`)
- **Engagement** — `NumWebVisitsMonth`, `Dt_Customer`
- **Campaign response** — `AcceptedCmp1`–`AcceptedCmp5`, `Response`, `Complain`

## Data Preprocessing

1. **Missing values** — 24 rows with a missing `Income` were dropped (2,240 → 2,216 rows).
2. **Column cleanup** — `ID` was dropped, as it carries no predictive signal.
3. **Feature engineering**:
   - `Total_Spendings` — sum of the six `Mnt*` spending categories and four purchase-channel counts
   - `Age` — derived from `Year_Birth`
   - `Customer_Since` — tenure in days, derived from `Dt_Customer`
   - `Accepted` — whether a customer accepted any of the five marketing campaigns (binary)
   - `AgeGroup` — binned age buckets, used only for exploratory analysis
4. **Exploratory analysis** — distribution plots for income, age, and spending; spending/income breakdowns by marital status and education; a correlation heatmap across the numeric features.
5. **Final feature set** — clustering was performed on seven features: `Age`, `Income`, `Total_Spendings`, `NumWebPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`, `Recency`.
6. **Scaling** — features were standardized with `StandardScaler` (zero mean, unit variance) before clustering. The fitted scaler is reused at inference time (`scaler.pkl`).

## Choosing the Number of Clusters

The elbow method was used to pick *k*: K-Means was fit for *k* = 2–9 on the scaled features, and inertia was recorded for each:

| k | Inertia |
|---|---------|
| 2 | 10,210.6 |
| 3 | 9,001.6 |
| 4 | 8,153.0 |
| 5 | 7,562.0 |
| 6 | 7,199.0 |
| 7 | 6,705.0 |
| 8 | 6,420.4 |
| 9 | 6,160.2 |

The inertia curve bends around **k = 5**, giving a good balance between model simplicity and cluster separation, so five clusters were used for the final model.

## Model

- **Algorithm**: K-Means (scikit-learn), `n_clusters=5`, fit on the seven standardized features
- **Dimensionality reduction**: PCA (2 components) was used only to visualize the clusters in 2D, not for training
- **Labeling**: each cluster's centroid was profiled (mean feature values) and manually mapped to a descriptive segment name
- **Artifacts**: the fitted model and scaler are exported with `joblib` as `kmeans_model.pkl` and `scaler.pkl`, which the Streamlit app loads directly

## Results

Cluster profiles (mean values per cluster):

| Cluster | Segment | Age | Income | Total Spending | Web Purchases | Store Purchases | Web Visits/mo | Recency | Customers |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Low Value Customers | 47.2 | $30,895 | $104 | 2.1 | 3.1 | 7.0 | 42.9 | 658 |
| 1 | Young High Value Customers | 46.1 | $77,805 | $1,305 | 4.7 | 8.5 | 2.7 | 50.2 | 342 |
| 2 | Budget Customers | 64.8 | $41,440 | $161 | 2.5 | 3.8 | 5.7 | 60.0 | 440 |
| 3 | Frequent Buyers | 58.9 | $59,127 | $877 | 7.7 | 7.8 | 6.4 | 43.5 | 450 |
| 4 | Old High Value Customers | 69.4 | $73,623 | $1,221 | 4.7 | 8.5 | 2.7 | 52.9 | 326 |

**Takeaways**:
- Income and total spending are strongly correlated (0.67) and both track closely with in-store purchase frequency.
- The two high-value segments (young and old) spend 8–12x more than the low-value and budget segments despite visiting the website far less often — they buy more per visit.
- Frequent Buyers stand out for their purchase volume across both web and store channels rather than for income alone.

## Technologies Used

- **Python 3**
- **pandas** / **numpy** — data loading and manipulation
- **matplotlib** / **seaborn** — exploratory data visualization
- **scikit-learn** — `StandardScaler`, `KMeans`, `PCA`
- **joblib** — model/scaler serialization
- **streamlit** — interactive prediction app
- **Jupyter Notebook** — exploratory analysis and model development (`segmentation.ipynb`)

## How to Run

1. **Clone the repository and enter the project folder**

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Create a virtual environment (recommended) and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app**

   ```bash
   streamlit run segmentation.py
   ```

4. **Use the app** — enter a customer's age, income, total spending, and purchase/engagement details, then click **Predict Segmentation** to see which of the five segments they fall into.

To explore or rerun the analysis and clustering from scratch, open `segmentation.ipynb` in Jupyter with `customer_segmentation.csv` in the same directory.

