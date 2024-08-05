import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def main():
    st.title('K-Means Clustering with Iris Dataset')

    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    st.subheader('Iris Dataset Overview')
    st.write("### Dataset Preview")
    st.write(df.head())

    # Sidebar for K-Means parameters
    st.sidebar.header('K-Means Parameters')
    n_clusters = st.sidebar.slider('Number of Clusters', 1, 10, 3)

    # Perform K-Means clustering
    km = KMeans(n_clusters=n_clusters, random_state=0)
    y_pred = km.fit_predict(df[['sepal length (cm)', 'sepal width (cm)']])
    df['Cluster'] = y_pred

    # Plot the clusters
    st.subheader('Cluster Plot')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", n_colors=n_clusters)
    for i in range(n_clusters):
        cluster_data = df[df.Cluster == i]
        ax.scatter(cluster_data['sepal length (cm)'], cluster_data['sepal width (cm)'],
                   color=colors[i], label=f'Cluster {i}')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.set_title('K-Means Clustering of Iris Dataset')
    ax.legend()
    st.pyplot(fig)

    # Compute the sum of squared distances for different values of k
    sse = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(df[['sepal length (cm)', 'sepal width (cm)']])
        sse.append(km.inertia_)

    # Plot the SSE for different values of k
    st.subheader('Elbow Method for Optimal k')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, sse, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Sum of Squared Distances (SSE)')
    ax.set_title('Elbow Method for Optimal k')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
