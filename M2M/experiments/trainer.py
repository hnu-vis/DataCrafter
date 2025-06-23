#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import random
import torch
from torch import nn
from dataset.warppers import DataSetWrapper
from utils.common_utils import check_path_exists, time_stamp_to_date_time_adjoin
from utils.math_utils import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import MultiStepLR
import shutil
from utils.constant_pool import ConfigInfo
from multiprocessing import Queue
from utils.logger import InfoLogger, LogWriter
import seaborn as sns
import time
import json
from experiments.tree_cut import TreeNode, HierarchicalTree


def draw_loss(training_loss, idx, save_path=None):
    plt.figure()
    plt.plot(idx, training_loss, color="blue", label="training loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()


def draw_projections(
    embeddings, text_embeddings, labels, vis_save_path, words, fig_type
):
    if embeddings.shape[0] % 100 != 0:
        x_img = embeddings[:1000, 0]
        y_img = embeddings[:1000, 1]
        labels = labels[:1000]
    else:
        x_img = embeddings[:, 0]
        y_img = embeddings[:, 1]
        labels = labels[:]
    if embeddings.shape[0] > 8000:
        class_list = [
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "bird",
        ]
    else:
        class_list = [
            "Beagle",
            "Bengal",
            "Birman",
            "Bombay",
            "Havanese",
            "Persian",
            "Pug",
            "Russian_blue",
            "Samoyed",
            "Shiba_inu",
        ]
    # class_list = self.class_list
    labels = [
        class_list[int(label) - 1] if label <= 10 else class_list[int((label - 1) % 10)]
        for label in labels
    ]
    plt.figure(figsize=(8, 8))
    num_classes = 10
    palette = "tab10" if num_classes <= 10 else "tab20"
    scatter = sns.scatterplot(
        x=x_img, y=y_img, s=8, hue=labels, palette=palette, alpha=0.8
    )
    if text_embeddings is not None:
        x_text = text_embeddings[:, 0]
        y_text = text_embeddings[:, 1]
        sns.scatterplot(
            x=x_text,
            y=y_text,
            s=8,
            color="red",
            marker="X",
            legend=False,
            alpha=0.8,
            label="Text Embeddings",
        )
        for i, word in enumerate(words):
            plt.text(x_text[i], y_text[i], word, fontsize=8, ha="right")
    plt.xticks([])
    plt.yticks([])
    # plt.legend()
    if vis_save_path is not None:
        name = vis_save_path.split("/")[-1]
        new_path = vis_save_path.replace(name, fig_type + "_" + name)
        plt.savefig(new_path, dpi=600, bbox_inches="tight", pad_inches=0.1)


class M2MTrainer:
    def __init__(
        self,
        model,
        dataset_name,
        configs,
        result_save_dir,
        config_path,
        device="cuda",
        log_path="log.txt",
    ):
        self.model = model
        self.config_path = config_path
        self.configs = configs
        self.device = device
        self.result_save_dir = result_save_dir
        self.dataset_name = dataset_name
        self.base_path = configs.exp_params.base_path
        self.a = configs.exp_params.image_image_loss
        self.b = configs.exp_params.image_text_loss
        self.c = configs.exp_params.text_image_cluster_loss
        self.d = configs.exp_params.text_text_loss
        self.e = configs.exp_params.stability_loss
        self.init_epoch = configs.exp_params.init_epoch
        self.batch_size = configs.exp_params.batch_size
        self.epoch_num = configs.training_params.epoch_nums
        self.n_neighbors = configs.exp_params.n_neighbors
        self.print_iter = int(
            self.configs.training_params.epoch_print_inter_ratio * self.epoch_num
        )
        self.is_image = not isinstance(self.configs.exp_params.input_dims, int)
        self.lr = configs.exp_params.LR
        self.ckp_save_dir = self.result_save_dir
        self.high_list = []
        self.image_image_loss = []
        self.image_text_loss = []
        self.text_image_cluster_loss = []
        self.batch_num = 0
        self.val_inter = 0
        self.start_epoch = 0
        self.train_loader = None
        self.launch_date_time = None
        self.optimizer = None
        self.scheduler = None
        self.tmp_log_path = log_path
        self.log_process = None
        self.log_path = None
        self.message_queue = Queue()
        self.fixed_k = 15
        self.texts = None
        self.words = None
        self.path = None
        self.class_list = (
            [
                "Beagle",
                "Bengal",
                "Birman",
                "Bombay",
                "Havanese",
                "Persian",
                "Pug",
                "Russian_blue",
                "Samoyed",
                "Shiba_inu",
            ]
            if self.dataset_name != "coco"
            else [
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "bird",
            ]
        )

        self.high_level_words = None
        self.word_frequency = None
        self.image_features = None
        self.image_info = None
        self.word_features_dict = None
        self.pre_embeddings = None
        self.text_postive_samples = None
        self.tree = None
        self.word_num_per_class = None
        self.clr_dataset = None
        self.resume_epochs = 0

        self.model.to(self.device)
        self.steps = 0
        self.resume_start_epoch = (
            self.resume_epochs if self.resume_epochs > 0 else self.epoch_num
        )
        self.gradient_redefine = configs.exp_params.gradient_redefine
        self.warmup_epochs = 0
        self.separation_epochs = 0
        if self.gradient_redefine:
            self.warmup_epochs = int(
                self.epoch_num * configs.exp_params.separation_begin_ratio
            )
            self.separation_epochs = int(
                self.epoch_num * configs.exp_params.steady_begin_ratio
            )

    def update_configs(self, configs):
        self.configs = configs
        self.dataset_name = configs.exp_params.dataset
        self.epoch_num = configs.training_params.epoch_nums

    def encode(self, x):
        return self.model.encode(x)

    def _train_begin(self, launch_time_stamp=None):
        self.sta_time = time.time() if launch_time_stamp is None else launch_time_stamp

        InfoLogger.info(
            "Start Training for {} Epochs".format(self.epoch_num - self.start_epoch)
        )

        param_template = (
            "Experiment Configurations: \nDataset: %s Epochs: %d Batch Size: %d \n"
            "Learning rate: %4f Optimizer: %s\n"
        )

        param_str = param_template % (
            self.dataset_name,
            self.epoch_num,
            self.batch_size,
            self.lr,
            self.configs.exp_params.optimizer,
        )

        InfoLogger.info(param_str)
        self.message_queue.put(param_str)

        InfoLogger.info("Start Training for {} Epochs".format(self.epoch_num))
        if self.launch_date_time is None:
            if launch_time_stamp is None:
                launch_time_stamp = int(time.time())
            self.launch_date_time = time_stamp_to_date_time_adjoin(launch_time_stamp)

        self.result_save_dir = os.path.join(
            self.result_save_dir,
            "{}_{}".format(self.dataset_name, self.launch_date_time),
        )

        self.log_path = os.path.join(self.result_save_dir, "log.txt")
        self.ckp_save_dir = self.result_save_dir

        if self.optimizer is None:
            self.init_optimizer()
            self.init_scheduler(cur_epochs=self.epoch_num)

        val_inter = math.ceil(
            self.epoch_num * self.configs.training_params.val_inter_ratio
        )
        ckp_save_inter = math.ceil(
            self.epoch_num * self.configs.training_params.ckp_inter_ratio
        )

        return val_inter, ckp_save_inter

    def init_optimizer(self):
        if self.configs.exp_params.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=0.0001
            )
        elif self.configs.exp_params.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001
            )
        else:
            raise RuntimeError(
                "Unsupported optimizer! Please check the configuration and ensure the param "
                "name is one of 'adam/sgd'"
            )

    def init_scheduler(self, cur_epochs, base=0, gamma=0.1, milestones=None):
        if milestones is None:
            milestones = [0.8]
        if self.configs.exp_params.scheduler == "multi_step":
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=[int(base + p * cur_epochs) for p in milestones],
                gamma=gamma,
                last_epoch=-1,
            )
        elif self.configs.exp_params.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_loader),
                eta_min=0.00001,
                last_epoch=-1,
            )
        else:
            raise RuntimeError(
                "Unsupported learning scheduler! Please check the configuration and ensure the param "
                "name is one of 'multi_step/cosine'"
            )

    def _before_epoch(self, epoch):
        self.model = self.model.to(self.device)
        if self.gradient_redefine:
            if epoch == self.warmup_epochs:
                self.train_loader.dataset.transform.build_neighbor_repo(
                    self.separation_epochs - self.warmup_epochs, self.n_neighbors
                )
            elif epoch == self.separation_epochs:
                self.train_loader.dataset.transform.build_neighbor_repo(
                    self.epoch_num - self.separation_epochs, self.n_neighbors
                )
        train_iterator = iter(self.train_loader)
        return train_iterator, 0

    def _step_prepare(self, *args):
        data, epoch = args
        x, x_sim, indices, sim_indices = data[0]
        x = x.to(self.device, non_blocking=True)
        x_sim = x_sim.to(self.device, non_blocking=True)
        return x, x_sim, epoch, indices, sim_indices

    def compute_stability_loss(self, x_embeddings, x_indices):
        size = self.pre_embeddings.shape[0]
        x_indices = x_indices.to(x_embeddings.device)  
        valid_mask = x_indices < size
        valid_indices = x_indices[valid_mask]
        pre_embeddings = self.pre_embeddings[valid_indices]
        cur_embeddings = x_embeddings[valid_mask]
        distances = torch.norm(cur_embeddings - pre_embeddings, dim=1)
        loss = torch.mean(distances)
        return loss

    def compute_loss_text_and_image(self, images, texts, images_emb, texts_emb):
        images = images / images.norm(dim=-1, keepdim=True)
        texts = texts / texts.norm(dim=-1, keepdim=True)
        (
            img_loss,
            txt_loss,
            cross_loss,
            all_loss,
            _,
            _,
            _,
            cross_dis,
            _,
            _,
            _,
            low_cross_dis,
        ) = self.model.data_context_map(images, texts, images_emb, texts_emb)
        loss = 0
        if self.steps > 100:
            loss = (
                loss + 10 * all_loss + 2 * cross_loss
            )  # +img_loss+txt_loss+2*cross_loss
        else:
            loss = loss + img_loss
        if self.steps > 800:
            for i in range(texts.shape[0]):
                loss = loss + 0.05 * torch.norm(
                    self.rank_loss(
                        images, texts[i], images_emb, texts_emb[i], self.batch_size
                    )
                )
        return loss

    def rank_loss(self, img_high, txt_high, img_low, txt_low, num):
        matrix = torch.norm(txt_low - img_low, dim=1).repeat(num, 1)
        # dis=torch.norm(matrix - matrix.transpose(0, 1), dim=-1)
        dis = matrix - matrix.T
        hmatrix = torch.norm(txt_high - img_high, dim=1).repeat(num, 1)
        hdis = hmatrix - hmatrix.T
        return nn.ReLU()(-dis * hdis) / (torch.norm(txt_low - img_low) + 0.0001)

    def get_word_indices_of_image_caption(self, image_indices, words, image_info):
        image_pos_words = {}
        image_pos_words_features = {}
        for idx in image_indices:
            item = image_info[idx]
            nouns = item["nouns"]
            image_pos_words[idx] = []
            image_pos_words_features[idx] = []
            for noun in nouns:
                if noun in words:
                    image_pos_words[idx].append(words.index(noun))
                    # image_pos_words_features[idx].append(self.word_features_dict[noun][:768])
        for k, v in image_pos_words.items():
            if len(v) == 0:
                image_pos_words[k].append(
                    self.compute_cosine_distance_and_get_topk_texts(
                        self.image_features[k], self.texts
                    )[0]
                )
        return image_pos_words

    def compute_cosine_distance_and_get_topk_texts(self, images, texts):
        if len(images.shape) == 1:
            images = images.unsqueeze(0)
        images_norm = images / images.norm(dim=-1, keepdim=True).to(torch.float32)
        # print(images_norm.shape)
        texts_norm = texts / texts.norm(dim=-1, keepdim=True).to(torch.float32)
        cosine_similarity = torch.mm(images_norm, texts_norm.T)  # [images * texts]
        topk_indices = torch.topk(
            cosine_similarity, k=1, dim=1, largest=True, sorted=True
        ).indices
        random_idx = []
        # text_count = {}
        for indices in topk_indices:
            random_index = random.choice(indices)
            random_idx.append(random_index)
        return random_idx

    def compute_word_pos_words_by_cosine_similarity(self):
        word_features = self.texts
        word_features = word_features / word_features.norm(dim=-1, keepdim=True).to(
            torch.float32
        )
        cosine_similarity = torch.mm(word_features, word_features.T)
        topk_indices = torch.topk(
            cosine_similarity, k=1, dim=1, largest=True, sorted=True
        ).indices
        random_idx = []
        for indices in topk_indices:
            random_index = random.choice(indices)
            # print(random_index)
            random_idx.append(random_index)
        return random_idx

    def get_image_word_by_highest_freq(
        self, image_indices, words, image_info, word_num_per_class
    ):
        image_pos_words = {}
        unique_classes = set()
        for item in self.class_list:
            unique_classes.add("train_" + item)
            unique_classes.add("gene_" + item)
        if self.dataset_name == "coco":
            unique_classes = set(self.class_list)
        max_class_for_word = {class_name: [] for class_name in unique_classes}
        for word, counts in word_num_per_class.items():
            max_class = max(counts, key=counts.get)
            if sum(counts.values()) == 0:
                continue
            max_class_for_word[max_class].append(word)
        # print(max_class_for_word)
        for idx in image_indices:
            image_pos_words[idx] = []
            if self.dataset_name == "coco":
                t = idx // 1000
                classname = self.class_list[t]
            else:
                if idx < 1000:
                    t = idx // 100
                    classname = "train_" + self.class_list[t]
                elif idx < 3000:
                    t = (idx - 1000) // 200
                    classname = "gene_" + self.class_list[t]
            image_pos_words[idx] = [
                self.words.index(name) for name in max_class_for_word[classname]
            ]
        return image_pos_words

    def _train_step(self, *args, texts, words):
        x, x_sim, epoch, indices, sim_indices = args
        self.optimizer.zero_grad()
        _, x_embeddings, _, x_sim_embeddings = self.forward(x, x_sim)
        class_list = self.class_list
        train_loss_image_image = torch.zeros(1).to(self.device)
        train_loss_image_text = torch.zeros(1).to(self.device)
        train_loss_text_text = torch.zeros(1).to(self.device)
        train_loss_text_image_cluster = torch.zeros(1).to(self.device)
        train_loss = torch.zeros(1).to(self.device)
        loss_stability = 0
        image_embeddings, text_embeddings = None, None
        if self.pre_embeddings is not None:
            loss_stability = self.compute_stability_loss(x_embeddings, indices)
        if self.configs.exp_params.method == "MFM":
            _, x_text_embeddings = self.text_forward(texts)
            train_loss = self.compute_loss_text_and_image(
                x, texts, x_embeddings, x_text_embeddings
            )
        else:
            if epoch < self.init_epoch:
                train_loss_image_image = self.model.compute_loss(
                    x_embeddings, x_sim_embeddings, epoch, None, None, None, 1
                )
                train_loss = self.a * train_loss_image_image + self.e * loss_stability
            else:
                _, x_text_embeddings = self.text_forward(texts)
                if epoch == self.init_epoch:
                    data = (
                        torch.tensor(self.train_loader.dataset.get_all_data())
                        .to(self.device)
                        .float()
                    )
                    image_embeddings, text_embeddings_pre = self.cal_lower_embeddings(
                        data
                    )
                    image_embeddings = torch.tensor(image_embeddings).to(self.device)
                    text_embeddings_pre = torch.tensor(text_embeddings_pre).to(
                        self.device
                    )
                    word_num_per_class = self.word_num_per_class
                    text_embeddings = torch.empty(len(self.texts), 2).to(self.device)
                    for word in self.words:
                        if sum(word_num_per_class[word].values()) == 0:
                            self.high_list.append(self.words.index(word))
                        for i, item in enumerate(word_num_per_class):
                            if word == item:
                                weight = {
                                    k: v
                                    for k, v in word_num_per_class[item].items()
                                    if sum(word_num_per_class[item].values()) != 0
                                }
                                sorted_weight = sorted(
                                    weight.items(), key=lambda x: x[1], reverse=True
                                )
                                max_names, current_sum, half_sum = (
                                    [],
                                    0,
                                    sum(weight.values()) / 2,
                                )
                                for name, value in sorted_weight:
                                    max_names.append(name)
                                    current_sum += value
                                    if current_sum > half_sum:
                                        break
                                if len(max_names) == 0:
                                    continue
                                if word_num_per_class[item][max_names[0]] == 0:
                                    text_embeddings[i] = text_embeddings_pre[i]
                                    continue
                                max_name = max_names[0]
                                if self.dataset_name == "coco":
                                    idx = class_list.index(max_name)
                                    text_embeddings[i] = image_embeddings[
                                        idx * 1000 : idx * 1000 + 1000
                                    ].mean(dim=0)
                                else:
                                    if max_name.split("_")[0] == "train":
                                        classname = "_".join(max_name.split("_")[1:])
                                        idx = class_list.index(classname)
                                        text_embeddings[i] = image_embeddings[
                                            idx * 100 : (idx + 1) * 100
                                        ].mean(dim=0)
                                    else:
                                        length = len(max_name.split("_"))
                                        classname = "_".join(max_name.split("_")[1:])
                                        if classname in class_list:
                                            idx = class_list.index(classname)
                                            text_embeddings[i] = image_embeddings[
                                                idx * 200
                                                + 1000 : (idx + 1) * 200 + 1000
                                            ].mean(dim=0)
                                        
                    self.high_list = list(set(self.high_list))
                train_loss_image_image = self.model.compute_loss(
                    x_embeddings, x_sim_embeddings, epoch, None, None, None, 1
                )
                image_pos_words_idx = self.get_image_word_by_highest_freq(
                    indices, words, self.image_info, self.word_num_per_class
                )
                train_loss_image_text = self.model.compute_loss(
                    x_embeddings,
                    x_text_embeddings,
                    epoch,
                    texts,
                    image_pos_words_idx,
                    self.word_frequency,
                    2,
                )
                train_loss_text_image_cluster = self.model.compute_loss(
                    x_embeddings,
                    x_text_embeddings,
                    epoch,
                    texts,
                    indices,
                    self.word_num_per_class,
                    4,
                    self.word_frequency,
                )
                train_loss_text_text = self.model.compute_loss(
                    texts,
                    x_text_embeddings,
                    epoch,
                    self.word_frequency,
                    self.text_postive_samples,
                    None,
                    3,
                )
                train_loss = (
                    self.a * train_loss_image_image
                    + self.b * train_loss_image_text
                    + self.c * train_loss_text_image_cluster
                    + self.d * train_loss_text_text
                    + 0 * loss_stability
                )
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=2.0, norm_type=2
        )
        self.optimizer.step()
        return (
            train_loss,
            image_embeddings,
            text_embeddings,
            train_loss_image_image.item(),
            train_loss_image_text.item(),
            train_loss_text_text.item(),
        )

    def model_prepare(self):
        self.model.preprocess()

    def _after_epoch(
        self, ckp_save_inter, epoch, training_loss, training_loss_history, val_inter
    ):

        if self.configs.exp_params.scheduler == "cosine" and epoch >= 10:
            self.scheduler.step()
        elif self.configs.exp_params.scheduler == "multi_step":
            self.scheduler.step()
        # print("Epoch: ", epoch)
        train_loss = training_loss / self.batch_num
        if epoch % self.print_iter == 0:
            epoch_template = "Epoch %d/%d, Train Loss: %.5f, "
            epoch_output = epoch_template % (epoch, self.epoch_num, train_loss)
            InfoLogger.info(epoch_output)
            self.message_queue.put(epoch_output)

        training_loss_history.append(train_loss.detach().cpu().numpy())
        embeddings = self.post_epoch(ckp_save_inter, epoch, val_inter)

        return embeddings

    def _train_end(self, training_loss_history, image_embeddings, text_embeddings):
        np.save(
            os.path.join(
                self.result_save_dir, "image_embeddings_{}.npy".format(self.epoch_num)
            ),
            image_embeddings,
        )
        np.save(
            os.path.join(
                self.result_save_dir, "text_embeddings_{}.npy".format(self.epoch_num)
            ),
            text_embeddings,
        )
        self.message_queue.put("end")
        self.save_weights(self.epoch_num)
        x_idx = np.linspace(
            self.start_epoch, self.epoch_num, self.epoch_num - self.start_epoch
        )
        save_path = os.path.join(
            self.result_save_dir, "loss_{}.jpg".format(self.epoch_num)
        )
        # print(len(training_loss_history), x_idx.shape)
        draw_loss(training_loss_history, x_idx, save_path)
        self.log_process.join(timeout=5)
        shutil.copyfile(self.tmp_log_path, self.log_path)
        InfoLogger.info("Training process logging to {}".format(self.log_path))
        if self.dataset_name not in [
            "coco"
        ]:
            self.normalize_embeddings_and_get_result(
                image_embeddings, text_embeddings, self.result_save_dir, type="m2m"
            )

    def normalize_embeddings_and_get_result(
        self, image_embeddings, text_embeddings, save_path, type=None
    ):
        class_list = self.class_list
        self.texts, self.words = self.get_all_texts()
        words = self.words
        self.prepare_treecut()
        self.word_frequency, self.image_info = self.prepare_json()
        images_info = self.image_info
        words_count = self.word_frequency

        max_val_x = np.max(
            np.concatenate((image_embeddings[:, 0], text_embeddings[:, 0]))
        )
        max_val_y = np.max(
            np.concatenate((image_embeddings[:, 1], text_embeddings[:, 1]))
        )
        min_val_x = np.min(
            np.concatenate((image_embeddings[:, 0], text_embeddings[:, 0]))
        )
        min_val_y = np.min(
            np.concatenate((image_embeddings[:, 1], text_embeddings[:, 1]))
        )

        normalized_image_embeddings = np.zeros_like(image_embeddings)
        normalized_image_embeddings[:, 0] = (image_embeddings[:, 0] - min_val_x) / (
            max_val_x - min_val_x
        )
        normalized_image_embeddings[:, 1] = (image_embeddings[:, 1] - min_val_y) / (
            max_val_y - min_val_y
        )

        normalized_text_embeddings = np.zeros_like(text_embeddings)
        normalized_text_embeddings[:, 0] = (text_embeddings[:, 0] - min_val_x) / (
            max_val_x - min_val_x
        )
        normalized_text_embeddings[:, 1] = (text_embeddings[:, 1] - min_val_y) / (
            max_val_y - min_val_y
        )
        result = {}
        image_index = 0
        if image_embeddings.shape[0] > 2000:
            for class_name in class_list:
                result[class_name] = {}
                for i in range(100):
                    result[class_name][str(image_index)] = [
                        float(normalized_image_embeddings[image_index][0]),
                        float(normalized_image_embeddings[image_index][1]),
                        images_info[image_index]["image_path"],
                        0,
                    ]
                    image_index += 1

            for class_name in class_list:
                for i in range(200):
                    if image_index < 1000:
                        result[class_name][str(image_index)] = [
                            float(normalized_image_embeddings[image_index][0]),
                            float(normalized_image_embeddings[image_index][1]),
                            images_info[image_index]["image_path"],
                            0,
                        ]
                    else:
                        result[class_name][str(image_index)] = [
                            float(normalized_image_embeddings[image_index][0]),
                            float(normalized_image_embeddings[image_index][1]),
                            images_info[image_index]["image_path"],
                            1,
                        ]
                        image_index += 1
        else:
            for class_name in class_list:
                result[class_name] = {}
                for i in range(int(image_embeddings.shape[0] / 10)):
                    result[class_name][str(image_index)] = [
                        float(normalized_image_embeddings[image_index][0]),
                        float(normalized_image_embeddings[image_index][1]),
                        images_info[image_index]["image_path"],
                        0,
                    ]
                    image_index += 1


    def train(self, launch_time_stamp=None):
        self.val_inter, ckp_save_inter = self._train_begin(launch_time_stamp)
        self.texts, self.words = self.get_all_texts()
        self.word_frequency, self.image_info = self.prepare_json()
        self.prepare_treecut()
        text_embeddings = None
        image_embeddings = None
        net = self.model
        net.batch_num = self.batch_num
        training_loss_history = []
        vis_text_emb = None
        for epoch in range(self.start_epoch, self.epoch_num):
            loss_image_text_sum = 0
            loss_image_image_sum = 0
            loss_text_image_cluster_sum = 0
            print("Epoch: ", epoch)
            train_iterator, training_loss = self._before_epoch(epoch)
            for idx, data in enumerate(train_iterator):
                self.steps += 1
                train_data = self._step_prepare(data, epoch)
                (
                    loss,
                    vis_image_emb,
                    vis_text_emb,
                    loss_image_image,
                    loss_image_text,
                    loss_text_image_cluster,
                ) = self._train_step(*train_data, texts=self.texts, words=self.words)
                training_loss += loss
                loss_image_text_sum += loss_image_text
                loss_image_image_sum += loss_image_image
                loss_text_image_cluster_sum += loss_text_image_cluster
            image_embeddings, text_embeddings = self._after_epoch(
                ckp_save_inter,
                epoch + 1,
                training_loss,
                training_loss_history,
                self.val_inter,
            )
            self.image_text_loss.append(loss_image_text_sum)
            self.image_image_loss.append(loss_image_image_sum)
            self.text_image_cluster_loss.append(loss_text_image_cluster_sum)
        self._train_end(training_loss_history, image_embeddings, text_embeddings)
        return image_embeddings, text_embeddings

    def resume_train(self, resume_epoch):
        self.resume_start_epoch = self.epoch_num
        self.start_epoch = self.epoch_num
        self.epoch_num = self.resume_start_epoch + resume_epoch
        self.optimizer.param_groups[0]["lr"] = self.lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(self.train_loader), eta_min=0.00001, last_epoch=-1
        )
        return self.train()

    def save_weights(self, epoch, prefix_name=None):
        if prefix_name is None:
            prefix_name = epoch
        if not os.path.exists(self.ckp_save_dir):
            os.mkdir(self.ckp_save_dir)
        weight_save_path = os.path.join(
            self.ckp_save_dir, "{}.pth.tar".format(prefix_name)
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr": self.lr,
                "launch_time": self.launch_date_time,
            },
            weight_save_path,
        )
        InfoLogger.info(
            "model weights successfully saved to {}".format(weight_save_path)
        )

    def forward(self, x, x_sim):
        return self.model.forward(x, x_sim)

    def text_forward(self, texts):
        return self.model.text_forward(texts)

    def load_weights(self, checkpoint_path, train=True):
        self.preprocess(train)
        model_ckpt = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_ckpt["state_dict"])
        self.init_optimizer()
        self.optimizer.load_state_dict(model_ckpt["optimizer"])
        self.optimizer.param_groups[0]["lr"] = self.lr
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        return model_ckpt

    def load_weights_train(self, checkpoint_path):
        model_ckpt = self.load_weights(checkpoint_path)
        self.start_epoch = model_ckpt["epoch"]
        self.launch_date_time = model_ckpt["launch_time"]
        self.train()

    def load_weights_visualization(self, checkpoint_path, vis_save_path, device="cuda"):
        self.load_weights(checkpoint_path, train=False)
        image_embeddings, text_embeddings = self.visualize(vis_save_path, device=device)
        return image_embeddings, text_embeddings

    def train_for_visualize(self):
        InfoLogger.info("Start train for Visualize")
        launch_time_stamp = int(time.time())
        self.preprocess()
        image_embeddings, text_embeddings = self.train(launch_time_stamp)
        return image_embeddings, text_embeddings

    def cal_lower_embeddings(self, data):
        if self.is_image:
            data = data / 255.0
        image_embeddings, text_embeddings = self.acquire_latent_code_allin(data)
        return image_embeddings, text_embeddings

    def visualize(self, vis_save_path=None, device="cuda:3"):
        self.model.to(device)
        data = torch.tensor(self.train_loader.dataset.get_all_data()).to(device).float()
        image_embeddings, text_embeddings = self.cal_lower_embeddings(data)
        draw_projections(
            image_embeddings,
            text_embeddings,
            self.train_loader.dataset.targets,
            vis_save_path,
            words=self.words,
            fig_type="m2m",
        )

        image_features = torch.from_numpy(
            np.load("{}/{}/image_features.npy".format(self.base_path,self.path))
        ).to(self.device, dtype=torch.float32)
        text_features = self.texts
        tsne_image, tsne_text = self.tsne(image_features, text_features)
        print(tsne_image.shape, tsne_text.shape)
        # draw_projections(tsne_image,tsne_text, self.train_loader.dataset.targets, vis_save_path,words = self.words,fig_type="tsne")
        # self.normalize_embeddings_and_get_result(tsne_image,tsne_text,self.result_save_dir,type="tsne")
        # pca_image,pca_text = self.pca(image_features,text_features)
        # draw_projections(pca_image,pca_text, self.train_loader.dataset.targets, vis_save_path,words = self.words,fig_type="pca")
        # umap_image,umap_text = self.umap(image_features,text_features)
        # draw_projections(umap_image,umap_text, self.train_loader.dataset.targets, vis_save_path,words = self.words,fig_type="umap")
        # mds_image,mds_text = self.mds(image_features,text_features)
        # draw_projections(mds_image,mds_text, self.train_loader.dataset.targets, vis_save_path,words = self.words,fig_type="mds")
        # dcm_image,dcm_text = self.dcm(image_features,text_features)
        # draw_projections(dcm_image,dcm_text, self.train_loader.dataset.targets, vis_save_path,words = self.words,fig_type="dcm")
        # self.normalize_embeddings_and_get_result(dcm_image,dcm_text,self.result_save_dir,type="dcm")

        return image_embeddings, text_embeddings

    def acquire_latent_code(self, inputs):
        return self.model.acquire_latent_code(inputs)

    def acquire_text_latent_code(self, texts):
        return self.model.acquire_text_latent_code(texts)

    def acquire_latent_code_allin(self, data):
        texts, _ = self.get_all_texts()
        with torch.no_grad():
            self.model.eval()
            image_embeddings = self.model.acquire_latent_code(data).cpu().numpy()
            text_embeddings = self.model.acquire_text_latent_code(texts).cpu().numpy()
            self.model.train()
        return image_embeddings, text_embeddings

    def preprocess(self, train=True):
        self.build_dataset()
        if train:
            self.log_process = LogWriter(
                self.tmp_log_path, self.log_path, self.message_queue
            )
            self.log_process.start()
        self.model_prepare()

    def build_dataset(self):
        knn_cache_path = os.path.join(
            ConfigInfo.NEIGHBORS_CACHE_DIR,
            "{}_k{}.npy".format(self.dataset_name, self.n_neighbors),
        )
        pairwise_cache_path = os.path.join(
            ConfigInfo.PAIRWISE_DISTANCE_DIR, "{}.npy".format(self.dataset_name)
        )
        check_path_exists(ConfigInfo.NEIGHBORS_CACHE_DIR)
        check_path_exists(ConfigInfo.PAIRWISE_DISTANCE_DIR)

        cdr_dataset = DataSetWrapper(self.batch_size)
        resume_start_epoch = self.resume_start_epoch
        if self.gradient_redefine:
            resume_start_epoch = self.warmup_epochs

        self.train_loader, self.n_samples = cdr_dataset.get_data_loaders(
            resume_start_epoch,
            self.dataset_name,
            ConfigInfo.DATASET_CACHE_DIR,
            self.n_neighbors,
            knn_cache_path,
            pairwise_cache_path,
            self.is_image,
        )

        self.batch_num = cdr_dataset.batch_num
        self.model.batch_num = self.batch_num

    def post_epoch(self, ckp_save_inter, epoch, val_inter):
        image_embeddings, text_embeddings = None, None
        vis_save_path = os.path.join(
            self.result_save_dir, "{}_vis_{}.jpg".format(self.dataset_name, epoch)
        )

        if epoch % val_inter == 0:
            if not os.path.exists(self.result_save_dir):
                os.makedirs(self.result_save_dir)
                if self.config_path is not None:
                    shutil.copyfile(
                        self.config_path,
                        os.path.join(self.result_save_dir, "config.yaml"),
                    )

            image_embeddings, text_embeddings = self.visualize(
                vis_save_path, device=self.device
            )

        # save model
        if epoch % ckp_save_inter == 0:
            if not os.path.exists(self.ckp_save_dir):
                os.makedirs(self.ckp_save_dir)
            self.save_weights(epoch)

        return image_embeddings, text_embeddings

    def get_all_texts(self):
        if self.dataset_name == "coco":
            self.path = "coco"
        else:
            self.path = self.dataset_name
        word_path = "{}/{}/node_frequency.json".format(self.base_path,self.path)
        words_frequency = dict(json.load(open(word_path)))
        new_word_features_dict = torch.load(
            "{}/{}/node_features.pt".format(self.base_path,self.path)
        ) 
        word_list = list(words_frequency.keys())
        word_features = []
        words = []
        if self.path == "coco":
            for word in word_list:
                for k, v in new_word_features_dict.items():
                    if word == k:
                        words.append(k)
                        word_features.append(v[:768])
        else:
            for word in word_list:
                for k, v in new_word_features_dict.items():
                    if word == k and word not in ["dog", "cat"]:
                        words.append(k)
                        word_features.append(v[:768])

        self.word_features_dict = new_word_features_dict
        word_features = torch.vstack(word_features)
        self.texts = word_features.to(self.device, dtype=torch.float32)
        self.words = words
        # print(self.texts.shape)
        # print(len(self.words))
        return self.texts, self.words

    def get_pre_embeddings(self):
        if self.dataset_name in {
            "step3"
        }:
            path = "/root/M2M/results/M2M/15/pets_20241113_13h40m45s/image_embeddings_1000.npy"
        self.pre_embeddings = torch.from_numpy(np.load(path)).to(self.device)
        print(self.pre_embeddings.shape)

    def prepare_json(self):
        word_path = "{}/{}/node_frequency.json".format(self.base_path,self.path)
        words_frequency = dict(json.load(open(word_path)))
        # words = self.words
        image_info_path = "{}/{}/images_info.json".format(self.base_path,self.path)
        with open(image_info_path, "r") as f:
            image_info = json.load(f)
        self.image_features = torch.from_numpy(
            np.load("{}/{}/image_features.npy".format(self.base_path,self.path))
        ).to(self.device, dtype=torch.float32)
        word_num_per_class_path = (
            "{}/{}/word_num_per_class.json".format(self.base_path,self.path)
        )
        with open(word_num_per_class_path, "r") as f:
            word_num_per_class = json.load(f)
        self.word_num_per_class = word_num_per_class
        return words_frequency, image_info

    def prepare_treecut(self):
        tree = HierarchicalTree(self.base_path,"root","")
        tree.prepare_info(self.path)
        tree_dict = tree.load_high_level_clusters()
        root = tree.create_hierarchy()
        clusters = tree.construct_word_postive_samples()
        text_pos = {}
        self.tree = tree
        # print(len(clusters))
        if self.path == "coco":
            for item in clusters:
                for word in item:
                    text_pos[word] = [
                        self.words.index(v) for v in item if v not in [word]
                    ]
        else:
            for item in clusters:
                for word in item:
                    if word not in ["dog", "cat"]:
                        text_pos[word] = [
                            self.words.index(v)
                            for v in item
                            if v not in [word, "dog", "cat"]
                        ]
        self.text_postive_samples = text_pos
        self.high_level_words = root.get_node_names_which_height_equals_1()

    def tsne(self, image_features, text_features):
        from sklearn.manifold import TSNE

        bz = image_features.shape[0]
        tsne = TSNE(n_components=2, random_state=0)
        if text_features is None:
            combined_matrix_2d = tsne.fit_transform(image_features.cpu().numpy())
            return combined_matrix_2d, None
        else:
            combined_matrix_2d = tsne.fit_transform(
                torch.cat((image_features, text_features), dim=0).cpu().numpy()
            )
            return combined_matrix_2d[:bz], combined_matrix_2d[bz:]

    def mds(self, image_features, text_features):
        from sklearn.manifold import MDS, smacof

        mds = MDS(n_components=2, random_state=0)
        bz = image_features.shape[0]
        if text_features is None:
            combined_matrix_2d = mds.fit_transform(image_features.cpu().numpy())
            return combined_matrix_2d, None
        else:
            concat_data = torch.cat((image_features, text_features), dim=0).to(
                torch.float32
            )
            dist_matrix = torch.cdist(concat_data, concat_data, p=2).cpu().numpy()
            data, stress = smacof(
                dist_matrix, n_components=2, metric=True, n_init=1, max_iter=300
            )
            return data[:bz], data[bz:]

    def dcm(self, image_features, text_features):
        from sklearn.manifold import MDS, smacof

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        distance_i_i = torch.cdist(image_features, image_features, p=2)
        distance_t_t = torch.cdist(text_features, text_features, p=2)
        cosine_similarity_matrix_i_t = torch.mm(image_features, text_features.T)
        distance_i_t = 1 - cosine_similarity_matrix_i_t
        distance_i_i /= distance_i_i.mean()
        distance_t_t /= distance_t_t.mean()
        distance_i_t /= distance_i_t.mean()

        combined_distance = (
            torch.cat(
                [
                    torch.cat([distance_i_i, distance_i_t], dim=1),
                    torch.cat([distance_i_t.T, distance_t_t], dim=1),
                ],
                dim=0,
            )
            .cpu()
            .numpy()
        )

        mds = MDS(n_components=2, random_state=0)
        data = mds.fit_transform(combined_distance)
        bz = image_features.shape[0]
        return data[:bz], data[bz:]

    def pca(self, image_features, text_features):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        bz = image_features.shape[0]
        if text_features is None:
            combined_matrix_2d = pca.fit_transform(image_features.cpu().numpy())
            return combined_matrix_2d, None
        else:
            combined_matrix_2d = pca.fit_transform(
                torch.cat((image_features, text_features), dim=0).cpu().numpy()
            )
            return combined_matrix_2d[:bz], combined_matrix_2d[bz:]

    def umap(self, image_features, text_features):
        import umap.umap_ as umap

        reducer = umap.UMAP()
        bz = image_features.shape[0]
        if text_features is None:
            combined_matrix_2d = reducer.fit_transform(image_features.cpu().numpy())
            return combined_matrix_2d, None
        else:
            combined_matrix_2d = reducer.fit_transform(
                torch.cat((image_features, text_features), dim=0).cpu().numpy()
            )
            return combined_matrix_2d[:bz], combined_matrix_2d[bz:]
