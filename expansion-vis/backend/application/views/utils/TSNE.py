import numpy as np

from sklearn.manifold import TSNE

def normalize_coordinates(coordinates):
    x_min = min(coordinates[:, 0])
    x_max = max(coordinates[:, 0])
    y_min = min(coordinates[:, 1])
    y_max = max(coordinates[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    for i in range(coordinates.shape[0]):
        coordinates[i][0] = (coordinates[i][0] - x_min) / x_range
        coordinates[i][1] = (coordinates[i][1] - y_min) / y_range
    return coordinates 


def tsne_projection(num, get_expansion_data, select_images_list, idx_to_labels, features, labels, labels1, length_original_dataset, n_iter=1000):
    if features.shape[0] < 10:
        perplexity = features.shape[0] // 2
    else:
        perplexity = 10
    tsne = TSNE(n_components=2, n_iter=n_iter, perplexity=perplexity)
    X_tsne_2d = tsne.fit_transform(features)
    
    X_tsne_2d = normalize_coordinates(X_tsne_2d)
    
    if get_expansion_data:
        unique_labels = np.unique(labels1)
    else:
        unique_labels = np.unique(labels)
    label_dict = {idx_to_labels[label]: {} for label in unique_labels}
    
    original_X_tsne_2d = X_tsne_2d[:length_original_dataset]
    print("original_X_tsne_2d.shape", original_X_tsne_2d.shape)
    
    count = 0
    if not get_expansion_data:
        for i, label in enumerate(labels):
            coordinates = original_X_tsne_2d[i]
            img_path = select_images_list[count]
            label_name = idx_to_labels[label]
            label_dict[label_name][count] = [float(coord) for coord in coordinates] + [img_path] + [0]
            count += 1
    if num == -1:
        return label_dict

    expansion_X_tsne_2d = X_tsne_2d[length_original_dataset:]
    print("expansion_X_tsne_2d.shape", expansion_X_tsne_2d.shape)
    for i, label in enumerate(labels1):
        coordinates = expansion_X_tsne_2d[i]
        img_path = select_images_list[count]
        label_name = idx_to_labels[label]
        label_dict[label_name][count] = [float(coord) for coord in coordinates] + [img_path] + [1]
        count += 1
    return label_dict
