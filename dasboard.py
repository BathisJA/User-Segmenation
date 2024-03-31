import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import numpy as np
import plotly.graph_objects as go

# Set page config for wide layout and expanded sidebar
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Load data
@st.cache_data()
def load_data():
    return pd.read_csv("data/user_profiles_for_ads.csv")

data = load_data()

# Sidebar
st.sidebar.title("Dataset Information")

# Display null values count
st.sidebar.write("Null Values Count:")
st.sidebar.write(data.isnull().sum())

# Main content
st.markdown("<h1 style='text-align: center;'>User Profiling and Segmentation</h1>", unsafe_allow_html=True)

# Display data
st.write("Data Preview:")
st.write(data)

# Demographic distributions
st.subheader("Distribution of Key Demographic Variables")
sns.set_style("whitegrid")

# subplot for the demographic distributions
fig, axes = plt.subplots(2, 2, figsize=(18,12))
fig.suptitle('Distribution of Key Demographic Variables', fontweight='bold')

# age distribution
sns.countplot(ax=axes[0, 0], x='Age', data=data, hue='Age', palette='coolwarm')
axes[0, 0].set_title('Age Distribution', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

# gender distribution
sns.countplot(ax=axes[0, 1], x='Gender', data=data, hue='Gender', palette='coolwarm')
axes[0, 1].set_title('Gender Distribution', fontweight='bold')

# education level distribution
sns.countplot(ax=axes[1, 0], x='Education Level', data=data, hue='Education Level', palette='coolwarm')
axes[1, 0].set_title('Education Level Distribution', fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)

# income level distribution
sns.countplot(ax=axes[1, 1], x='Income Level', data=data, hue='Income Level', palette='coolwarm')
axes[1, 1].set_title('Income Level Distribution', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(fig)

# Device usage distribution
st.subheader("Device Usage Distribution")
device_fig, device_ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='Device Usage', data=data, hue='Device Usage', palette='coolwarm', ax=device_ax)
device_ax.set_title('Device Usage Distribution')
st.pyplot(device_fig)

# User online behavior and ad interaction metrics
st.subheader("User Online Behavior and Ad Interaction Metrics")
behavior_fig, behavior_axes = plt.subplots(3, 2, figsize=(18, 15))
behavior_fig.suptitle('User Online Behavior and Ad Interaction Metrics')

# time spent online on weekdays
sns.histplot(ax=behavior_axes[0, 0], x='Time Spent Online (hrs/weekday)', data=data, bins=20, kde=True, color='skyblue')
behavior_axes[0, 0].set_title('Time Spent Online on Weekdays')

# time spent online on weekends
sns.histplot(ax=behavior_axes[0, 1], x='Time Spent Online (hrs/weekend)', data=data, bins=20, kde=True, color='orange')
behavior_axes[0, 1].set_title('Time Spent Online on Weekends')

# likes and reactions
sns.histplot(ax=behavior_axes[1, 0], x='Likes and Reactions', data=data, bins=20, kde=True, color='green')
behavior_axes[1, 0].set_title('Likes and Reactions')

# click-through rates
sns.histplot(ax=behavior_axes[1, 1], x='Click-Through Rates (CTR)', data=data, bins=20, kde=True, color='red')
behavior_axes[1, 1].set_title('Click-Through Rates (CTR)')

# conversion rates
sns.histplot(ax=behavior_axes[2, 0], x='Conversion Rates', data=data, bins=20, kde=True, color='purple')
behavior_axes[2, 0].set_title('Conversion Rates')

# ad interaction time
sns.histplot(ax=behavior_axes[2, 1], x='Ad Interaction Time (sec)', data=data, bins=20, kde=True, color='brown')
behavior_axes[2, 1].set_title('Ad Interaction Time (sec)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(behavior_fig)

# Top 10 User Interests
st.subheader("Top 10 User Interests")
interests_list = data['Top Interests'].str.split(', ').sum()
interests_counter = Counter(interests_list)
interests_df = pd.DataFrame(interests_counter.items(), columns=['Interest', 'Frequency']).sort_values(by='Frequency', ascending=False)

top_interests_fig, top_interests_ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='Frequency', y='Interest', data=interests_df.head(10), hue='Frequency', palette='coolwarm', legend=False, ax=top_interests_ax)
top_interests_ax.set_title('Top 10 User Interests')
top_interests_ax.set_xlabel('Frequency')
top_interests_ax.set_ylabel('Interest')
st.pyplot(top_interests_fig)

st.subheader("User Segmentation")

# selecting features for clustering
features = ['Age', 'Gender', 'Income Level', 'Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']

# separating the features we want to consider for clustering
X = data[features]

# defining preprocessing for numerical and categorical features
numeric_features = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']
numeric_transformer = StandardScaler()

categorical_features = ['Age', 'Gender', 'Income Level']
categorical_transformer = OneHotEncoder()

# combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# creating a preprocessing and clustering pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('cluster', KMeans(n_clusters=5, random_state=42))])

pipeline.fit(X)
cluster_labels = pipeline.named_steps['cluster'].labels_
data['Cluster'] = cluster_labels

st.write(data.head())

# computing the mean values of numerical features for each cluster
cluster_means = data.groupby('Cluster')[numeric_features].mean()

for feature in categorical_features:
    mode_series = data.groupby('Cluster')[feature].agg(lambda x: x.mode()[0])
    cluster_means[feature] = mode_series

st.write(cluster_means)

# Preparing data for radar chart
features_to_plot = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']
labels = np.array(features_to_plot)

# Creating a DataFrame for the radar chart
radar_df = cluster_means[features_to_plot].reset_index()

# Normalizing the data
radar_df_normalized = radar_df.copy()
for feature in features_to_plot:
    radar_df_normalized[feature] = (radar_df[feature] - radar_df[feature].min()) / (radar_df[feature].max() - radar_df[feature].min())

# Adding a full circle for plotting
radar_df_normalized = pd.concat([radar_df_normalized, radar_df_normalized.iloc[[0]]], ignore_index=True)

# Assigning names to segments
segment_names = ['Weekend Warriors', 'Engaged Professionals', 'Low-Key Users', 'Active Explorers', 'Budget Browsers']

fig = go.Figure()

# Loop through each segment to add to the radar chart
for i, segment in enumerate(segment_names):
    fig.add_trace(go.Scatterpolar(
        r=radar_df_normalized.iloc[i][features_to_plot].values.tolist() + [radar_df_normalized.iloc[i][features_to_plot].values[0]],  # Add the first value at the end to close the radar chart
        theta=labels.tolist() + [labels[0]],  # add the first label at the end to close the radar chart
        fill='toself',
        name=segment,
        hoverinfo='text',
        text=[f"{label}: {value:.2f}" for label, value in zip(features_to_plot, radar_df_normalized.iloc[i][features_to_plot].values)]+[f"{labels[0]}: {radar_df_normalized.iloc[i][features_to_plot].values[0]:.2f}"]  # Adding hover text for each feature
    ))

# finalize the radar chart
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='User Segments Profile'
)

st.plotly_chart(fig, use_container_width=True)

