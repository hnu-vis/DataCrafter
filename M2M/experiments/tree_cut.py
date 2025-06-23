import os
import sys
import torch
import numpy as np
import json
from sklearn.cluster import KMeans
import string
from openai import OpenAI
import requests
import base64
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.probability import FreqDist
import clip
from PIL import Image


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []
        self.isleaf = False
        self.h = 0

    def add_child(self, child_node):
        child_node.parent = self
        child_node.h = self.h + 1
        self.children.append(child_node)
        self.isleaf = False
        child_node.isleaf = True

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.name) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def get_all_node_names(self):
        node_names = [self.name]
        for child in self.children:
            node_names.extend(child.get_all_node_names())
        return node_names

    def get_all_children(self):
        children = [self]
        for child in self.children:
            children.extend(child.get_all_children())
        return children

    def get_node_names_which_height_equals_1(self):
        node_names = []
        for child in self.children:
            if child.h == 1:
                node_names.append(child.name)
        return node_names

    def find_path(self, root, path, k):
        if root is None:
            return False
        path.append(root)
        if root.name == k:
            return True
        for child in root.children:
            if self.find_path(child, path, k):
                return True
        path.pop()
        return False

    def find_distance(self, root, n1, n2):
        path1 = []
        path2 = []
        if not self.find_path(root, path1, n1) or not self.find_path(root, path2, n2):
            return -1
        i = 0
        while i < len(path1) and i < len(path2):
            if path1[i].name != path2[i].name:
                break
            i += 1
        lca_distance = (len(path1) - i) + (len(path2) - i)
        return lca_distance


class HierarchicalTree:
    def __init__(self, base_path,root_name,api_key):
        self.root = TreeNode(root_name)
        self.high_level_features = None
        self.tree_dict = {}
        self.image_features = None
        self.text_features = None
        self.image_info = None
        self.path_name = None
        self.words = None
        self.clusters = None
        self.node_features = {}
        self.word_frequency = None
        self.node_frequency = {}
        self.base_path = base_path
        self.api_key = api_key
        

    def create_tree(self, data, parent):
        if isinstance(data, dict):
            for key, value in data.items():
                node = TreeNode(key)
                parent.add_child(node)
                self.create_tree(value, node)
        elif isinstance(data, list):
            for item in data:
                node = TreeNode(item)
                parent.add_child(node)

    def create_hierarchy(self):
        self.create_tree(self.tree_dict, self.root)
        return self.root

    def prepare_info(self, path_name):
        self.path_name = path_name
        image_path = "{}/{}/image_features.npy".format(self.base_path, path_name)
        text_path = "{}/{}/text_features.pt".format(self.base_path, path_name)
        image_info_path = "{}/{}/images_info.json".format(self.base_path, path_name)
        words_path = "{}/{}/frequency.json".format(self.base_path, path_name)
        self.image_features = torch.from_numpy(np.load(image_path))
        self.text_features = torch.load(text_path)
        with open(image_info_path) as f:
            self.image_info = json.load(f)
        self.words = list(dict(json.load(open(words_path))).keys())
        self.word_frequency = dict(json.load(open(words_path)))

    def combine_text_image_features(self):
        image_info = self.image_info
        # words = list(self.text_features.keys())
        combined_features = {}
        for word in self.words:
            temp_features = torch.empty([0, 768])
            for i, item in enumerate(image_info):
                if word in item["nouns"]:
                    temp_features = torch.cat(
                        (temp_features, self.image_features[i].unsqueeze(0)), dim=0
                    )
            temp_features = torch.mean(temp_features, dim=0)
            # print(temp_features.shape)
            combined_features[word] = torch.cat(
                (self.text_features[word].unsqueeze(0), temp_features.unsqueeze(0)),
                dim=1,
            )
            if torch.isnan(combined_features[word]).any():
                print(word)
        torch.save(
            combined_features,
            os.path.join(
                "{}/{}/combined_features.pt".format(self.base_path, self.path_name)
            ),
        )
        return combined_features

    def kmeans_clustering(self, num):
        word_images_features = self.combine_text_image_features()
        word_images_features = [
            word_images_features[word].detach().numpy() for word in word_images_features
        ]  
        word_images_features = np.vstack(
            word_images_features
        )  
        np.save(
            "{}/{}/word_images_features.npy".format(self.base_path, self.path_name),
            word_images_features,
        )
        
        clusters_num = num  
        kmeans = KMeans(n_clusters=clusters_num, random_state=0).fit(
            word_images_features
        )  
        labels = kmeans.labels_  
        words = self.words

        initial_clusters = {label: [] for label in np.unique(labels)}
        for word, label in zip(words, labels):
            initial_clusters[label].append(word)

        final_clusters = {}
        for label, cluster_words in initial_clusters.items():
            cluster_features = [
                word_images_features[words.index(word)] for word in cluster_words
            ]
            refined_cluster = self.refine_clusters(
                np.array(cluster_features), cluster_words, 3, 8, f"cluster-{label+1}"
            )
            final_clusters[f"cluster-{label+1}"] = refined_cluster

        output_file_path = "{}/{}/clustered_words.json".format(
            self.base_path, self.path_name
        )
        with open(output_file_path, "w", encoding="utf-8") as json_file:
            json.dump(final_clusters, json_file, indent=2, ensure_ascii=False)
        self.clusters = final_clusters
        print(f"Clustered words saved to {output_file_path}")
        return None

    def refine_clusters(
        self, features, words, min_words, max_words, cluster_name="cluster-1"
    ):
        if len(words) <= max_words:
            return words  
        sse = []
        for clusters_num in range(2, len(words) // min_words):
            kmeans = KMeans(n_clusters=clusters_num, random_state=0).fit(features)
            labels = kmeans.labels_
            sse.append(kmeans.inertia_)
        best_k = self.optimal_number_of_clusters(sse)
        if best_k == 1:
            return words
        kmeans = KMeans(n_clusters=best_k, random_state=0).fit(features)
        labels = kmeans.labels_
        refined_clusters = {label: [] for label in np.unique(labels)}
        for word, label in zip(words, labels):
            refined_clusters[label].append(word)
        result = {}
        for label, cluster_words in refined_clusters.items():
            cluster_features = [features[words.index(word)] for word in cluster_words]
            sub_cluster_name = f"{cluster_name}-{label+1}"
            sub_cluster = self.refine_clusters(
                np.array(cluster_features),
                cluster_words,
                min_words,
                max_words,
                sub_cluster_name,
            )
            result[sub_cluster_name] = sub_cluster
        return result

    def optimal_number_of_clusters(self, sse):
        x = range(1, len(sse) + 1)
        y = sse
        slopes = [(y[i] - y[i + 1]) / (x[i] - x[i + 1]) for i in range(len(x) - 1)]
        max_slope_change = 0
        best_k = 1
        for i in range(1, len(slopes)):
            slope_change = abs(slopes[i] - slopes[i - 1])
            if slope_change > max_slope_change:
                max_slope_change = slope_change
                best_k = i + 1
        return best_k

    def get_high_level_word(self, words):
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.openai.com/v1",
        )
        prompt = f"Here are some words: {', '.join(words)}. Please only provide a high-level word, preferably a noun, to better summarize the overall meaning of these words. No additional answers are required."
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Let me know if you have any questions."},
                {"role": "user", "content": prompt},
            ],
        )
        high_level_word = str(completion.choices[0].message.content).strip()
        high_level_word = high_level_word.translate(
            str.maketrans("", "", string.punctuation)
        )
        return high_level_word.lower()

    def process_clusters(self, clusters):
        high_level_words = {}
        for cluster, content in clusters.items():
            if isinstance(content, list):
                high_level_words[cluster] = self.get_high_level_word(content)
            elif isinstance(content, dict):
                sub_high_level_words = self.process_clusters(content)
                high_level_word_list = list(sub_high_level_words.values())
                high_level_words[cluster] = self.get_high_level_word(
                    high_level_word_list
                )
                high_level_words.update(sub_high_level_words)
        return high_level_words

    def replace_clusters_with_high_level_words(self, clusters, high_level_descriptions):
        new_clusters = {}
        for cluster, content in clusters.items():
            high_level_word = high_level_descriptions.get(cluster)
            if isinstance(content, list):
                new_clusters[high_level_word] = content
            elif isinstance(content, dict):
                new_clusters[high_level_word] = (
                    self.replace_clusters_with_high_level_words(
                        content, high_level_descriptions
                    )
                )
        return new_clusters

    def wirte_down_high_level_clusters(self):
        clusters = self.clusters
        high_level_descriptions = self.process_clusters(clusters)
        final_clusters = self.replace_clusters_with_high_level_words(
            clusters, high_level_descriptions
        )
        output_file_path = "{}/{}/high_level_clusters.json".format(
            self.base_path, self.path_name
        )
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(final_clusters, f, ensure_ascii=False, indent=4)
        print("high-level words have been saved in high_level_clusters.json")
        self.tree_dict = final_clusters

    def load_high_level_clusters(self):
        with open(
            "{}/{}/high_level_clusters.json".format(self.base_path, self.path_name)
        ) as f:
            self.tree_dict = json.load(f)
        return self.tree_dict

    def compute_average_feature(self, node, node_name="root"):
        # only compute the average feature for high-level nodes
        if isinstance(node, list):
            # Leaf nodes (word lists)
            features = [
                self.text_features[word] for word in node if word in self.text_features
            ]
            if features:
                avg_feature = torch.mean(torch.stack(features), dim=0)
                self.node_features[node_name] = avg_feature
                for word in node:
                    if word in self.text_features:
                        self.node_features[word] = self.text_features[word]
                return avg_feature
            else:
                return None

        elif isinstance(node, dict):
            # Internal nodes (dictionaries)
            features = []
            for key, value in node.items():
                child_feature = self.compute_average_feature(value, node_name=key)
                if child_feature is not None:
                    features.append(child_feature)
            if features:
                avg_feature = torch.mean(torch.stack(features), dim=0)
                self.node_features[node_name] = avg_feature
                return avg_feature
            else:
                return None
        else:
            raise ValueError("Unsupported node type")

    def get_node_features(self):
        self.compute_average_feature(self.tree_dict)
        path = "{}/{}/node_features.pt".format(self.base_path, self.path_name)
        torch.save(self.node_features, path)
        print(f"Node features saved to {path}")
        return self.node_features

    def make_treecut(self):
        treecut_result = []
        nodes = self.root.get_all_children()
        for node in nodes:
            click_word_result = self.get_selected_words(node.name, [])
            treecut_result.append(
                {"click_word": node.name, "selected_words": click_word_result}
            )
        path = "{}/{}/treecut.json".format(self.base_path, self.path_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(treecut_result, f, ensure_ascii=False, indent=4)
        return treecut_result

    def calculate_and_merge_distance(self, cur_node, root, node_features):
        node_names = self.root.get_all_node_names()
        merged_list = []

        for node in node_names:
            tree_distance = self.root.find_distance(root, cur_node, node)
            cur_node_feature = node_features[cur_node]
            cur_node_feature = cur_node_feature / torch.norm(cur_node_feature)
            node_feature = node_features[node]
            node_feature = node_feature / torch.norm(node_feature)
            cos_similarity = torch.cosine_similarity(
                cur_node_feature.float(), node_feature.float(), dim=0
            ).item()
            merged_list.append(
                {
                    "name": node,
                    "tree_distance": tree_distance,
                    "cos_similarity": cos_similarity,
                    "degree of interest": cos_similarity - tree_distance,
                }
            )

        sorted_result = sorted(
            merged_list, key=lambda x: x["degree of interest"], reverse=True
        )
        return sorted_result

    def get_selected_words(self, click_word, final_nodes=[]):
        sorted_result = self.calculate_and_merge_distance(
            click_word, self.root, self.node_features
        )
        # if click_word == "tiger":
        #     print(sorted_result)
        nodes = self.root.get_all_children()
        # print(sorted_result)
        # print(nodes)
        final_nodes.append(click_word)
        max_size = 20
        for node in nodes:
            temp_node = node
            if node.name == click_word:
                while (
                    node.name != "root"
                    and node.name not in final_nodes
                    and len(final_nodes) < 10
                ):
                    if node.isleaf:
                        final_nodes.append(node.name)
                    node = node.parent
                node = temp_node
                if (
                    node.isleaf
                    and node.name not in final_nodes
                    and len(final_nodes) < 10
                ):
                    final_nodes.append(node.name)
                else:
                    children = node.get_all_children()
                    for child in children:
                        if child.name not in final_nodes and len(final_nodes) < 10:
                            if node.isleaf:
                                final_nodes.append(child.name)
        for entry in sorted_result:
            cur_node_name = entry["name"]
            if cur_node_name == "root":
                continue
            for node in nodes:
                if node.name == cur_node_name:
                    cur_node = node
                    if (
                        cur_node.name not in final_nodes
                        and len(final_nodes) < max_size
                        and cur_node.isleaf
                    ):
                        final_nodes.append(cur_node.name)

                # final_nodes.append(cur_node_name)
        select_list = []
        for node in final_nodes:
            select_list.append(node)
        return select_list

    def construct_word_postive_samples(self):
        tree_dict = self.tree_dict
        nodes = self.root.get_all_children()
        all_clusters = []
        for node in nodes:
            if node.h == 1:
                clusters = []
                children_names = node.get_all_node_names()
                for child in children_names:
                    clusters.append(child)
                all_clusters.append(clusters)
        path = "{}/{}/word_positive_samples.json".format(self.base_path, self.path_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_clusters, f, ensure_ascii=False, indent=4)
        return all_clusters

    def recalculate_word_frequency(self, tree_dict):
        total_frequency = 0
        if isinstance(tree_dict, list):
            # Leaf nodes (word lists)
            for word in tree_dict:
                word_frequency = self.word_frequency.get(word, 0)
                total_frequency += word_frequency
                self.node_frequency[word] = word_frequency

        elif isinstance(tree_dict, dict):
            # Internal nodes (dictionaries)
            for key, value in tree_dict.items():
                child_total_frequency = self.recalculate_word_frequency(value)
                total_frequency += child_total_frequency
                self.node_frequency[key] = child_total_frequency

        return total_frequency

    def re_claculate_word_frequency(self):
        total_frequency = self.recalculate_word_frequency(self.tree_dict)
        path = "{}/{}/node_frequency.json".format(self.base_path, self.path_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.node_frequency, f, ensure_ascii=False, indent=4)
        print(f"Node frequency saved to {path}")


class Files:
    def __init__(self, name, device, base_path,api_key):
        self.name = name
        self.device = device
        self.base_path = base_path
        self.api_key = api_key

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def extract_and_lemmatize_nouns(self, sentence):
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        nouns = [word for word, pos in pos_tags if pos.startswith("NN")]
        lemmatized_nouns = [lemmatizer.lemmatize(noun, wordnet.NOUN) for noun in nouns]
        return lemmatized_nouns

    def lemmatize_sentence(self, sentence):
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(sentence)
        lemmatized_words = [
            lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in words
        ]
        return " ".join(lemmatized_words)

    def get_frequency_and_images_info(self):
        file_path = "{}/{}/captions.json".format(self.base_path, self.name)
        with open(file_path, "r") as file:
            data = json.load(file)
        all_nouns = []
        for item in data:
            original_caption = item.get("caption", "")
            lemmatized_nouns = self.extract_and_lemmatize_nouns(original_caption)
            lemmatized_nouns = [
                noun for noun in lemmatized_nouns if noun not in ["cat", "dog"]
            ]
            all_nouns.extend(lemmatized_nouns)
        freq_dist = FreqDist(all_nouns)
        # generate the frequency file
        freq_dist_json = {word: frequency for word, frequency in freq_dist.items()}
        sorted_freq_dist = dict(
            sorted(freq_dist_json.items(), key=lambda item: item[1], reverse=True)
        )
        freq_dist_file_path = "{}/{}/noun_frequency.json".format(
            self.base_path, self.name
        )
        with open(freq_dist_file_path, "w") as file:
            json.dump(sorted_freq_dist, file, indent=4)

        # filter the frequency less than 7
        noun_frequency = json.load(
            open("{}/{}/noun_frequency.json".format(self.base_path, self.name))
        )
        filtered_noun_frequency = {
            word: frequency
            for word, frequency in noun_frequency.items()
            if frequency >= 7
        }
        filtered_noun_frequency_file_path = "{}/{}/frequency.json".format(
            self.base_path, self.name
        )
        with open(filtered_noun_frequency_file_path, "w") as file:
            json.dump(filtered_noun_frequency, file, indent=4)
        print("Frequency file saved to frequency.json")
        # read the new frequency file and get the words
        file_path = "{}/{}/captions.json".format(self.base_path, self.name)
        with open(file_path, "r") as file:
            data = json.load(file)
        frequency_7_file_path = "{}/{}/frequency.json".format(
            self.base_path, self.name
        )  
        with open(frequency_7_file_path, "r") as file:
            word_list = json.load(file)

        words = list(word_list.keys())
        word_dict = {}
        # handle the captions
        for item in data:
            original_caption = item.get("caption", "")
            lemmatized_caption = self.lemmatize_sentence(original_caption)
            nouns = self.extract_and_lemmatize_nouns(original_caption)
            item["lemmatized_caption"] = lemmatized_caption
            nouns_list = [noun for noun in nouns if noun in words]
            nouns_list = list(set(nouns_list))
            item["nouns"] = nouns_list
            for word in words:
                if word in words:
                    word_dict[word] = word_dict.get(word, 0) + 1

        # generate the images info file
        new_file_path = "{}/{}/images_info.json".format(self.base_path, self.name)
        with open(new_file_path, "w") as file:
            json.dump(data, file, indent=4)
        print("Images info file saved to images_info.json")

    def get_captions(self, image_folder):

        # OpenAI API Key
        api_key = self.api_key

        # Send requests to the OpenAI API for each image and save responses
        save_path = "{}/{}/captions.json".format(self.base_path, self.name)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        # Initialize an empty list to store image data
        image_data_list = []
        # Walk through the image folder and process each image
        for root, dirs, files in os.walk(image_folder):
            for image_file in files:
                if image_file.endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(root, image_file)
                    image_data = {"image_path": image_path}
                    image_data_list.append(image_data)

        # Function to generate payload for OpenAI API
        def generate_payload(base64_image):
            return {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please use a caption of about 15 words to describe the content of this picture. Note that the caption should be a complete sentence. No additional answers are required.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # Load existing data if the file exists
        if os.path.exists(save_path):
            with open(save_path, "r") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []
        # Merge existing data with new data
        existing_paths = {data["image_path"] for data in existing_data}
        for image_data in image_data_list:
            if image_data["image_path"] not in existing_paths:
                existing_data.append(image_data)
        for idx, image_data in tqdm(
            enumerate(existing_data), desc="Processing images", unit="image"
        ):
            if "caption" not in image_data:  # Skip images that already have captions
                image_path = image_data["image_path"]
                base64_image = encode_image(image_path)
                payload = generate_payload(base64_image)
                response = requests.post(
                    "https://api.openai.com/v1",
                    headers=headers,
                    json=payload,
                )
                image_data["caption"] = response.json()["choices"][0]["message"][
                    "content"
                ]
                image_data["index"] = idx + 1
                # print(image_data["caption"])
                # Save the updated image data with responses to a JSON file
                with open(save_path, "w") as json_file:
                    json.dump(existing_data, json_file, indent=4)
        print("Image data with responses has been saved to captions.json")
        
        
    def get_words_per_class(self):
        path = "{}/{}/images_info.json".format(self.base_path, self.name)
        word_path = "{}/{}/node_frequency.json".format(self.base_path, self.name)
        with open(word_path, "r") as f:
            word_data = json.load(f)
        with open(path, "r") as f:
            data = json.load(f)
        word_num_per_class = {}
        # print(len(word_data))
        # Get unique class names
        # class_list  = [
        #         'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird'
        #     ]
        class_list = ["Beagle","Bengal","Birman","Bombay","Havanese","Persian","Pug","Russian_blue","Samoyed","Shiba_inu"]
        unique_classes = set()
        for item in data:
            image_path = item["image_path"]
            classname = image_path.split("/")[-2]
            unique_classes.add("train_" + classname)
        for item in data:
            image_path = item["image_path"]
            classname = image_path.split("/")[-2]
            unique_classes.add("gene_" + classname)
        if self.name == "coco":
            unique_classes = class_list

        for word in word_data:
            word_num_per_class[word] = {} 
            for class_name in unique_classes:
                word_num_per_class[word][class_name] = 0
            for i, item in enumerate(data):
                image_path = item["image_path"]
                classname = image_path.split("/")[-2]
                if i < 1000:
                    new_name = "train_"+classname
                elif i < 3000:
                    new_name = "gene_"+classname
                if self.name == "coco":
                    new_name = class_list[i // 1000]
                if word in item["nouns"]:
                    word_num_per_class[word][new_name] = word_num_per_class[word].get(new_name, 0) + 1
        save_path = "{}/{}/word_num_per_class.json".format(self.base_path, self.name)
        with open(save_path, "w") as f:
            json.dump(word_num_per_class, f, indent=4)
            
        
    def extract_features(self):
        image_folder = "/root/DatasetExpansion/GIF_SD/Pets/data/pets/train/"
        save_path = "{}/{}/image_features.npy".format(self.base_path, self.name)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Load the images
        image_files = []
        for root, dirs, files in os.walk(image_folder):
            for image_file in files:
                if image_file.endswith((".png", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, image_file))
        # Load the model
        device = self.device
        model, preprocess = clip.load("ViT-L/14", device=device)
        image_features = []
        for image_file in tqdm(image_files, desc="Processing images"):
            image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(image)
                image_features.append(feature.cpu().numpy())
        image_features = np.concatenate(image_features, axis=0)
        np.save(save_path, image_features)
        print(f"Image features saved to {save_path}")
        frequency_file_path = "{}/{}/frequency.json".format(self.base_path, self.name)
        with open(frequency_file_path, "r") as file:
            word_list = json.load(file)
        text_features = {}
        for word in word_list:
            text_description = f"a photo of a {word}"
            text = clip.tokenize([text_description]).to(device)
            with torch.no_grad():
                feature = model.encode_text(text).cpu()
                text_features[word] = feature.squeeze()
        pt_file_path = "{}/{}/text_features.pt".format(self.base_path, self.name)
        torch.save(text_features, pt_file_path)
        print(f"Text features saved to {pt_file_path}")


if __name__ == "__main__":
    base_path = "/root/M2M/backend/data"
    api_key = "your_api_key"
    image_folder= "your_image_folder"
    # get the captions and frequency
    files = Files("pets", "cuda", base_path, api_key)
    files.get_captions("pets",image_folder)
    files.get_frequency_and_images_info()
    files.get_words_per_class()
    files.extract_features()
    
    
    tree = HierarchicalTree(base_path,"root",api_key)
    tree.prepare_info("pets")
    # only conduct cluster once
    tree.kmeans_clustering(8)
    tree.wirte_down_high_level_clusters()
    # if you have already conducted the cluster, you can exegesis the top two lines and run the following code
    tree_dict = tree.load_high_level_clusters()
    root = tree.create_hierarchy()
    tree.get_node_features()
    all_clusters = tree.construct_word_postive_samples()
    tree.get_node_features()
    treecut_result = tree.make_treecut()
    tree.re_claculate_word_frequency()
