# Customer-Segmentation
Customer segmentation using K-Means clustering with an interactive Streamlit prediction app.

##Project Overview
The main aim of this project is to help improve marketing strategies by identifying different types of customers and enabling more personalized marketing campaigns and offers.

The project analyzes customers features such as:
-Age
-Income
-Total spending
-Number of web purchases
-Number of store purchases
-Number of web visits
-Recency (days since the last purchase)

Using theses features, K-Means groups the customers into different segments:
The identified customer segments are:

-Low Value Customers
-Young High Value Customers
-Budget Customers
-Frequent Buyers
-Old High Value Customers


##Project Interpretation
Understanding different customer groups can help businesses make more targeted decisions, such as:
-Identifying which customers would benefit from specific marketing campaigns
-Identifying which customers should recieve offers first
-Creating more personalized promotions based on customer behavior
-Understanding differences in customer purchasing patterns

##Interactive Prediction App
The project includes a Streamlit application where users can enter customer information and receive a predicted customer segment.
The application uses the trained K-Means model and the same scaler used during training to assign the customer to the appropriate segment.
