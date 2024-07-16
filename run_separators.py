import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import get_filtered_and_mean_file_bood, get_filtered_and_mean_file_fluid
from consts import FILENAME_CSV_BLOOD, PATH_TO_CSV_BLOOD, PATH_TO_CSV_FLUID, HOW_TO_COMBINE_EMBEDDINGS




# FILE_TO_BLOOD = "filtered_and_mean_filedata_20240625_173958.csv"


def apply_tsne(group):
    embeddings = group.filter(regex='embedding_')
    embeddings['category'] = group['category']
    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='random')
    tsne_results = tsne.fit_transform(embeddings.drop('category', axis=1))

    group['tsne-2d-one'] = tsne_results[:, 0]
    group['tsne-2d-two'] = tsne_results[:, 1]

    # Plotting
    plt.figure(figsize=(10, 8))
    colors = {'blood': 'red', 'fluid': 'blue'}
    for category, color in colors.items():
        subset = group[group['category'] == category]
        plt.scatter(subset['tsne-2d-one'], subset['tsne-2d-two'], c=color, label=category, alpha=0.5)

    plt.title('t-SNE Visualization of Two Categories')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    layer = group['esm2_layer'].unique()[0]
    plt.savefig(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/final_output/tsne_outputs/tsne_plot_with_{HOW_TO_COMBINE_EMBEDDINGS}_{FILENAME_CSV_BLOOD[:-4]}_layer_{layer}.png")
    plt.close()


def apply_pca(df_combined):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_combined.filter(regex='^embedding_'))

    df_combined['pca-one'] = principal_components[:, 0]
    df_combined['pca-two'] = principal_components[:, 1]

    colors = {'blood': 'red', 'fluid': 'blue'}
    plt.figure(figsize=(10, 8))
    for category, color in colors.items():
        subset = df_combined[df_combined['category'] == category]
        plt.scatter(subset['pca-one'], subset['pca-two'], c=color, label=category, alpha=0.5)

    plt.title('PCA of Protein Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    layer = df_combined['esm2_layer'].unique()[0]
    plt.savefig(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/final_output/pca_outputs/pca_plot_with_{HOW_TO_COMBINE_EMBEDDINGS}_{FILENAME_CSV_BLOOD[:-4]}_layer_{layer}.png")
    plt.close()

def apply_kmeans(df_combined):
    categories = df_combined['category'].map({'blood': 0, 'fluid': 1}).values
    df_combined = df_combined.drop(columns=['category'])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_combined.filter(regex='^embedding_'))

    kmeans = KMeans(n_clusters=2, random_state=0).fit(principal_components)
    labels = kmeans.labels_

    # Plotting the results
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=categories, cmap='viridis', alpha=0.5)
    plt.title('K-means Clusters after PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Actual Category')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.text(centers[0, 0], centers[0, 1], 'Cluster 1 Center', color='white')
    plt.text(centers[1, 0], centers[1, 1], 'Cluster 2 Center', color='white')
    # df_combined['cluster'] = kmeans.labels_
    # df_combined['pca-one'] = principal_components[:, 0]
    # df_combined['pca-two'] = principal_components[:, 1]

    # colors = {'0': 'red', '1': 'blue'}
    # plt.figure(figsize=(10, 8))
    # for cluster, color in colors.items():
    #     subset = df_combined[df_combined['cluster'] == int(cluster)]
    #     plt.scatter(subset['pca-one'], subset['pca-two'], c=color, label=f"Cluster {cluster}", \
    #                 alpha=0.3, s=50, edgecolors='none')

    # plt.title('PCA Colored by K-means Clusters')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.legend()
    layer = df_combined['esm2_layer'].unique()[0]
    plt.savefig(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/final_output/kmeans_outputs/kmeans_plot_with_{HOW_TO_COMBINE_EMBEDDINGS}_{FILENAME_CSV_BLOOD[:-4]}_layer_{layer}.png")
    plt.close()



def main():
    df_blood = pd.read_csv(get_filtered_and_mean_file_bood(PATH_TO_CSV_BLOOD, FILENAME_CSV_BLOOD))
    unnamed_cols_blood = [col for col in df_blood.columns if col.startswith('Unnamed')]
    df_blood = df_blood.drop(columns=unnamed_cols_blood)
    df_fluid = pd.read_csv(get_filtered_and_mean_file_fluid(PATH_TO_CSV_FLUID))
    unnamed_cols_fluid = [col for col in df_fluid.columns if col.startswith('Unnamed')]
    df_fluid = df_fluid.drop(columns=unnamed_cols_fluid)
    # df_fluid = df_fluid[len(df_fluid)//2:]

    df_blood['category'] = 'blood'
    df_fluid['category'] = 'fluid'
    df_combined = pd.concat([df_blood, df_fluid], ignore_index=True)
    df_combined = df_combined.sample(frac=1)
    df_combined = df_combined.reset_index(drop=True)
    grouped_df_comibed = df_combined.groupby('esm2_layer')
    grouped_df_comibed.apply(apply_tsne)
    grouped_df_comibed.apply(apply_pca)
    grouped_df_comibed.apply(apply_kmeans)
    ...


if __name__ == '__main__':
    main()