import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import get_filtered_and_mean_file_bood, get_filtered_and_mean_file_fluid, seq_identity
from consts import FILENAME_CSV_BLOOD, PATH_TO_CSV_BLOOD, PATH_TO_CSV_FLUID, \
    FILENAME_CSV_FLUID, HOW_TO_COMBINE_EMBEDDINGS, SAVE_FILE_SEPARATORS_NAME, PERPLEXITY, STATISTICS_IMAGES_PATH, \
    GRAPH_NAME_PERF, NUMBER_TYPE_T
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from Bio import pairwise2



# FILE_TO_BLOOD = "filtered_and_mean_filedata_20240625_173958.csv"


def apply_tsne(group_origin):
    for perplexity in PERPLEXITY:
        group = group_origin.copy(deep=True)
        embeddings = group.filter(regex='embedding_')
        # embeddings['category'] = group['category']
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='random')
        tsne_results = tsne.fit_transform(embeddings)
        # tsne_results = tsne.fit_transform(embeddings.drop('category', axis=1))

        group['tsne-2d-one'] = tsne_results[:, 0]
        group['tsne-2d-two'] = tsne_results[:, 1]

        # Plotting
        plt.figure(figsize=(10, 8))
        colors = {'blood': 'red', 'fluid': 'blue'}
        for category, color in colors.items():
            subset = group[group['category'] == category]
            plt.scatter(subset['tsne-2d-one'], subset['tsne-2d-two'], c=color, label=category, alpha=0.5)

        layer = group['esm2_layer'].unique()[0]
        plt.title(f't-SNE {SAVE_FILE_SEPARATORS_NAME} perplexity {perplexity} {HOW_TO_COMBINE_EMBEDDINGS} layer {layer} len {len(group)}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.savefig(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/final_output/tsne_outputs/tsne_plot_with_{HOW_TO_COMBINE_EMBEDDINGS}_perplexity_{perplexity}_{SAVE_FILE_SEPARATORS_NAME}_layer_{layer}.png")
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
    plt.savefig(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/final_output/pca_outputs/pca_plot_with_{HOW_TO_COMBINE_EMBEDDINGS}_{SAVE_FILE_SEPARATORS_NAME}_layer_{layer}.png")
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
    plt.savefig(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/final_output/kmeans_outputs/kmeans_plot_with_{HOW_TO_COMBINE_EMBEDDINGS}_{SAVE_FILE_SEPARATORS_NAME}_layer_{layer}.png")
    plt.close()


def apply_knn(df_combined_orig):
    for n_neighbor in [3, 5]:
        df_combined = df_combined_orig.copy(deep=True)
        X = df_combined.filter(regex='embedding_')
        y = df_combined['category']
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        accuracies = []
        layer = df_combined['esm2_layer'].unique()[0]
        counter = 0

        for train_index, test_index in kfold.split(X):
            counter += 1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            knn = KNeighborsClassifier(n_neighbors=n_neighbor, metric="minkowski")
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)

            draw_simlarity_graph(df_combined, test_index, y_test, predictions, f"knn_with_{n_neighbor}_neighbors", counter, accuracy, train_index)
        
        average_accuracy = np.mean(accuracies)
        print(f"KNN Accuracy for layer {layer} with k-fold CV and n_neighbor {n_neighbor} metric = Minkowski: {average_accuracy}")


def draw_the_3_graphs(prob_dist_all_seq, file_save_name_pref, layer):
    all_identities = [item for sublist in prob_dist_all_seq for item in sublist]
    plt.figure(figsize=(8, 6))
    plt.hist(all_identities, bins=20, color='blue', alpha=0.7)
    plt.title(f'Histogram of Sequence Identities {GRAPH_NAME_PERF} layer {layer}')
    plt.xlabel('Sequence Identity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(STATISTICS_IMAGES_PATH, f'histogram_sequence_identities_matplotlib_{file_save_name_pref}_{GRAPH_NAME_PERF}_layer_{layer}.png'))  # Save the figure

    means = [np.mean(sublist) for sublist in prob_dist_all_seq]
    plt.figure(figsize=(8, 6))
    plt.hist(means, bins=20, color='green', alpha=0.7)
    plt.title(f'Histogram of Mean Sequence Identities for Test Sequences {GRAPH_NAME_PERF} layer {layer}')
    plt.xlabel('Mean Sequence Identity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(STATISTICS_IMAGES_PATH, f'histogram_mean_sequence_identities_{file_save_name_pref}_{GRAPH_NAME_PERF}_layer_{layer}.png'))  # Save the figure

    maxs = [np.max(sublist) for sublist in prob_dist_all_seq]
    plt.figure(figsize=(8, 6))
    plt.hist(maxs, bins=20, color='blue', alpha=0.7)
    plt.title(f'Histogram of Maximum Sequence Identities for Test Sequences {GRAPH_NAME_PERF} layer {layer}')
    plt.xlabel('Maximum Sequence Identity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(STATISTICS_IMAGES_PATH, f'histogram_maximum_sequence_identities_{file_save_name_pref}_{GRAPH_NAME_PERF}_layer_{layer}.png'))  # Save the figure


def calculate_statistics_on_train_test(df_combined, train_index, test_index):
    seqs_train = df_combined.iloc[train_index]['seq'].tolist()
    seqs_test = df_combined.iloc[test_index]['seq'].tolist()

    prob_test_dist_all_max = {}
    for seq_test in seqs_test:
        prob_dist_one_seq = []
        for seq_train in seqs_train:
            seq_id = seq_identity(seq_test, seq_train)
            prob_dist_one_seq.append(seq_id)
            # print(f"seq_test {seq_test}, seq_train {seq_train}, score {pairwise2.align.globalxx(seq_test, seq_train)}")
            # print(f"final score is {seq_id}\n")
        # prob_dist_all_seq.append(prob_dist_one_seq)
        prob_test_dist_all_max[seq_test] = max(prob_dist_one_seq)

    return prob_test_dist_all_max

    # draw_the_3_graphs(prob_dist_all_seq, file_save_name_pref, layer)

def apply_random_forest_kfold(df_combined, n_splits=5):
    n_estimators = 100
    max_depth = None
    X = df_combined.filter(regex='embedding_')
    y = df_combined['category']

    # Setup KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    layer = df_combined['esm2_layer'].unique()[0]
    counter = 0
    # dist_all_tests = {}
    for train_index, test_index in kf.split(X):
        counter += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Create and train the Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)

        # Predict and evaluate the model
        predictions = rf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

        draw_simlarity_graph(df_combined, test_index, y_test, predictions, "random_forest_kfold", counter, accuracy, train_index)


    # Calculate the average accuracy across all folds
    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Random Forest Average Accuracy with {n_splits}-fold CV: {average_accuracy:.4f} and STD: {std_accuracy}")

def draw_simlarity_graph(df_combined, test_index, y_test, predictions, \
                         classifier, counter, accuracy, train_index):
    prob_test_dist_all_max = calculate_statistics_on_train_test(df_combined, train_index, test_index)

    assert df_combined.iloc[test_index].index.equals(y_test.index), "The order of df_combined must be the same as the order of y_test"
    seqs_test = df_combined.iloc[test_index]['seq'].tolist()  # We needs to make sure that df_cimibed(seq) is the same as the order of y_test. It should be
    scores = np.array([prob_test_dist_all_max[y] for y in seqs_test])
    correct = predictions == y_test

    correct_scores_mean = scores[correct].mean()
    incorrect_scores_mean = scores[~correct].mean()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(scores)), scores, color=np.where(correct, 'blue', 'red'), label='Prediction Correctness')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    plt.title(f'Scores by Prediction Correctness. Accuracy: {accuracy} for {len(seqs_test)} samples')
    
    # Add horizontal lines for the mean values
    plt.axhline(y=correct_scores_mean, color='blue', linestyle='--', label='Mean Correct Score')
    plt.axhline(y=incorrect_scores_mean, color='red', linestyle='--', label='Mean Incorrect Score')

    # Adding legend
    colors = {'Correct': 'blue', 'Incorrect': 'red'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    plt.savefig(f"/cs/labs/dina/sapir_amittai/code/spring_24/TCRep/final_output/outputs_images_distance_correctlly/type_{NUMBER_TYPE_T}/{classifier}_type_{NUMBER_TYPE_T}_iter_{counter}.png")  # Save the figure


def apply_random_forest(df_combined):
    n_estimators=100
    max_depth=None
    X = df_combined.filter(regex='embedding_')
    y = df_combined['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Random Forest Accuracy:", accuracy)


def apply_random_forest_cv(df_combined):
    n_estimators=100
    max_depth=None
    n_splits=5
    X = df_combined.filter(regex='embedding_')
    y = df_combined['category']

    # Create a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Perform k-fold cross-validation
    scores = cross_val_score(rf, X, y, cv=n_splits)
    average_accuracy = np.mean(scores)

    print(f"Random Forest Average Accuracy with {n_splits}-fold CV: {average_accuracy:.4f}")


def apply_qda(df_combined):
    X = df_combined.filter(regex='embedding_')
    y = df_combined['category']
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    layer = df_combined['esm2_layer'].unique()[0]
    
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        predictions = qda.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
    
    average_accuracy = np.mean(accuracies)
    print(f"QDA Accuracy for layer {layer} with k-fold CV: {average_accuracy}")


def apply_mog(df_combined_orig):
    for n_component in range(1, 10):
        df_combined = df_combined_orig.copy(deep=True)
        X = df_combined.filter(regex='embedding_')
        y = df_combined['category']
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        accuracies = []
        layer = df_combined['esm2_layer'].unique()[0]
        
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            mog = GaussianMixture(n_components=n_component)
            mog.fit(X_train)
            mog_labels = mog.predict(X_test)
            mog_labels = ['blood' if label == 0 else 'fluid' for label in mog_labels]
            accuracy = accuracy_score(y_test, mog_labels)
            accuracies.append(accuracy)
        
        average_accuracy = np.mean(accuracies)
        print(f"MOG Accuracy for layer {layer} with k-fold CV and n_component {n_component}: {average_accuracy}")


def main_on_esm2_native_way():
    """
    This function is doing pca and tsne for try to separate between blood and fluid examples
    """
    np.random.seed(0)
    df_blood = pd.read_csv(get_filtered_and_mean_file_bood(PATH_TO_CSV_BLOOD, FILENAME_CSV_BLOOD))
    unnamed_cols_blood = [col for col in df_blood.columns if col.startswith('Unnamed')]
    df_blood = df_blood.drop(columns=unnamed_cols_blood)
    df_fluid = pd.read_csv(get_filtered_and_mean_file_fluid(PATH_TO_CSV_FLUID))
    unnamed_cols_fluid = [col for col in df_fluid.columns if col.startswith('Unnamed')]
    df_fluid = df_fluid.drop(columns=unnamed_cols_fluid)

    df_blood = df_blood.sample(n=480, random_state=42)
    df_fluid = df_fluid.sample(n=480, random_state=42)

    df_blood['category'] = 'blood'
    df_fluid['category'] = 'fluid'
    df_combined = pd.concat([df_blood, df_fluid], ignore_index=True)
    df_combined = df_combined.sample(frac=1)
    df_combined = df_combined.reset_index(drop=True)
    grouped_df_comibed = df_combined.groupby('esm2_layer')
    grouped_df_comibed.apply(apply_tsne)
    # grouped_df_comibed.apply(apply_pca)
    # grouped_df_comibed.apply(apply_kmeans)
    grouped_df_comibed.apply(apply_knn)
    # grouped_df_comibed.apply(apply_qda)
    # grouped_df_comibed.apply(apply_mog)
    # grouped_df_comibed.apply(apply_random_forest_cv)
    grouped_df_comibed.apply(apply_random_forest_kfold)
    # grouped_df_comibed.apply(apply_random_forest)
    ...


def main_baseline_take_only_blood():
    """
    This function is doing pca and tsne for try to separate between blood and fluid examples
    """
    np.random.seed(0)
    df_blood = pd.read_csv(get_filtered_and_mean_file_bood(PATH_TO_CSV_BLOOD, FILENAME_CSV_BLOOD))
    unnamed_cols_blood = [col for col in df_blood.columns if col.startswith('Unnamed')]
    df_blood = df_blood.drop(columns=unnamed_cols_blood)
    # df_fluid = pd.read_csv(get_filtered_and_mean_file_fluid(PATH_TO_CSV_FLUID))
    # unnamed_cols_fluid = [col for col in df_fluid.columns if col.startswith('Unnamed')]
    # df_fluid = df_fluid.drop(columns=unnamed_cols_fluid)

    df_blood = df_blood.sample(n=1000, random_state=42)
    df_blood_1 = df_blood[:500]
    df_blood_2 = df_blood[500:]

    df_blood_1['category'] = 'blood'
    df_blood_2['category'] = 'fluid'
    df_combined = pd.concat([df_blood_1, df_blood_2], ignore_index=True)
    df_combined = df_combined.sample(frac=1)
    df_combined = df_combined.reset_index(drop=True)
    grouped_df_comibed = df_combined.groupby('esm2_layer')
    # grouped_df_comibed.apply(apply_tsne)
    # grouped_df_comibed.apply(apply_pca)
    # grouped_df_comibed.apply(apply_kmeans)
    grouped_df_comibed.apply(apply_knn)
    # grouped_df_comibed.apply(apply_qda)
    # grouped_df_comibed.apply(apply_mog)
    # grouped_df_comibed.apply(apply_random_forest_cv)
    grouped_df_comibed.apply(apply_random_forest_kfold)
    # grouped_df_comibed.apply(apply_random_forest)
    ...


def main():
    np.random.seed(0)
    df_blood = pd.read_csv(os.path.join(PATH_TO_CSV_BLOOD, FILENAME_CSV_BLOOD))
    unnamed_cols_blood = [col for col in df_blood.columns if col.startswith('Unnamed')]
    df_blood = df_blood.drop(columns=unnamed_cols_blood)
    df_fluid = pd.read_csv(os.path.join(PATH_TO_CSV_FLUID, FILENAME_CSV_FLUID))
    unnamed_cols_fluid = [col for col in df_fluid.columns if col.startswith('Unnamed')]
    df_fluid = df_fluid.drop(columns=unnamed_cols_fluid)

    # Delete same seq
    common_seq = set(df_blood['seq']).intersection(df_fluid['seq'])
    df_blood_filtered = df_blood[~df_blood['seq'].isin(common_seq)]
    df_fluid_filtered = df_fluid[~df_fluid['seq'].isin(common_seq)]

    # Sorted the longer df
    min_length = min(len(df_blood_filtered), len(df_fluid_filtered))
    if len(df_blood_filtered) > min_length:
        df_blood_filtered = df_blood_filtered.sample(n=min_length).reset_index(drop=True)
    elif len(df_fluid_filtered) > min_length:
        df_fluid_filtered = df_fluid_filtered.sample(n=min_length).reset_index(drop=True)

    df_blood_filtered['category'] = 'blood'
    df_fluid_filtered['category'] = 'fluid'
    df_combined = pd.concat([df_blood_filtered, df_fluid_filtered], ignore_index=True)
    df_combined = df_combined.sample(frac=1)
    df_combined = df_combined.reset_index(drop=True)
    grouped_df_comibed = df_combined.groupby('esm2_layer')
    # grouped_df_comibed.apply(apply_tsne)
    # grouped_df_comibed.apply(apply_pca)
    grouped_df_comibed.apply(apply_knn)
    # grouped_df_comibed.apply(apply_qda)
    # grouped_df_comibed.apply(apply_mog)
    # grouped_df_comibed.apply(apply_random_forest_cv)
    grouped_df_comibed.apply(apply_random_forest_kfold)
    # grouped_df_comibed.apply(apply_random_forest)
    ...


if __name__ == '__main__':
    # main_baseline_take_only_blood()
    # main_on_esm2_native_way()
    main()