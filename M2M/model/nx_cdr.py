import numpy as np
from torch.nn import Module
from model.nce_loss import torch_app_skewnorm_func
from utils.math_utils import get_similarity_function, get_correlated_mask
from utils.umap_utils import find_ab_params
from model.baseline_encoder import *
from audtorch.metrics.functional import pearsonr
import random


def exp_ce(data_matrix, t, data_labels, accumulation="MEAN"):
    exp_data = torch.exp(data_matrix / t)
    return ce(exp_data, data_labels, accumulation)


def skewnorm_ce(data_matrix, ratio, data_labels, accumulation="MEAN"):
    sn_data = torch_app_skewnorm_func(data_matrix, ratio)
    return ce(sn_data, data_labels, accumulation)


def ce(data_matrix, data_labels, accumulation="MEAN"):
    softmax_data = data_matrix / torch.sum(data_matrix, dim=1).unsqueeze(1)
    loss = -torch.log(softmax_data[torch.arange(0, data_matrix.shape[0]), data_labels])
    if accumulation == "MEAN":
        return torch.mean(loss)
    elif accumulation == "SUM":
        return torch.sum(loss)
    else:
        return loss


class NX_CDRModel(Module):
    def __init__(self, cfg, device="cuda"):
        Module.__init__(self)
        self.device = device
        self.config = cfg
        self.input_dims = cfg.exp_params.input_dims
        self.encoder_name = cfg.exp_params.method
        self.in_channels = (
            1 if isinstance(self.input_dims, int) else self.input_dims[-1]
        )

        self.input_size = int(np.sqrt(self.input_dims / self.in_channels))
        self.latent_dim = 2
        self.batch_size = cfg.exp_params.batch_size
        self.similarity_method = "umap"
        self.temperature = cfg.exp_params.temperature
        self.temperature_text = cfg.exp_params.temperature_text
        self.text_batch_size = cfg.exp_params.text_batch_size
        self.batch_num = 0
        self.max_neighbors = 0
        self.encoder = None
        self.text_encoder = None
        self.pro_head = None
        self.datasets_name = cfg.exp_params.dataset
        self.criterion = None
        self.correlated_mask = get_correlated_mask(2 * self.batch_size)
        self.min_dist = 0.1

        self._a, self._b = find_ab_params(1, self.min_dist)
        self.similarity_func = get_similarity_function(self.similarity_method)
        self.similarity_func_text = get_similarity_function("cosine")

        self.reduction = "mean"
        self.epoch_num = self.config.training_params.epoch_nums
        self.batch_count = 0

    def build_model(self):
        encoder, encoder_out_dims = get_encoder(
            self.encoder_name, self.input_size, self.input_dims, self.in_channels
        )
        print(encoder_out_dims)
        self.encoder = encoder
        pro_dim = 512
        if self.encoder_name == "M2M":
            self.pro_head = nn.Sequential(
                nn.Linear(encoder_out_dims, pro_dim),
                nn.ReLU(),
                nn.Linear(pro_dim, self.latent_dim),
            )
        else:
            self.pro_head = nn.Sequential(
                nn.Linear(encoder_out_dims, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.latent_dim),
            )

    def preprocess(self):
        self.build_model()
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)

    def encode(self, x):
        if x is None:
            return None, None
        reps = self.encoder(x)
        reps = reps.squeeze()
        embeddings = self.pro_head(reps)
        return reps, embeddings

    def text_encode(self, texts):
        if texts is None:
            return None, None
        # text_reps = self.text_encoder(texts)
        text_reps = self.encoder(text_reps)  # 让text和image过同一个网络
        # print(text_reps.shape)
        text_reps = text_reps.squeeze()
        text_embeddings = self.pro_head(text_reps)
        # print(text_embeddings.shape)
        return text_reps, text_embeddings

    def forward(self, x, x_sim):
        # get the representations and the projections
        x_reps, x_embeddings = self.encode(x)  # [N,C]

        # get the representations and the projections
        x_sim_reps, x_sim_embeddings = self.encode(x_sim)  # [N,C]

        return x_reps, x_embeddings, x_sim_reps, x_sim_embeddings

    def text_forward(self, texts):
        texts_reps, texts_embeddings = self.encode(texts)
        # print(texts_embeddings.shape)
        return texts_reps, texts_embeddings

    def acquire_latent_code(self, inputs):
        reps, embeddings = self.encode(inputs)
        return embeddings

    def acquire_text_latent_code(self, texts):
        text_reps, text_embeddings = self.encode(texts)
        return text_embeddings

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        loss = 0
        epoch = args[0]
        logits = self.batch_logits(x_embeddings, x_sim_embeddings, *args)
        if args[1] is None:
            loss += self._post_loss(logits, x_embeddings, epoch, None, *args)
        else:
            loss_ = self._post_loss(logits, x_embeddings, epoch, None, *args)
            loss += loss_
            # if args[4] == 4:
            #     print("loss: ",loss_)
            if np.isnan(loss.cpu().detach().numpy()):
                # print(logits)
                raise ValueError("Loss is NaN")
        return loss

    def _post_loss(self, logits, x_embeddings, epoch, item_weights, *args):
        labels = torch.zeros(logits.shape[0]).to(self.device).long()
        if args[1] != None:
            loss = self.criterion(logits / self.temperature_text, labels)
        else:
            loss = self.criterion(logits / self.temperature, labels)
        return loss

    def batch_logits(self, x_embeddings, x_sim_embeddings, *args):
        self.batch_count += 1
        device = self.device
        flag = args[4]
        if flag == 2:  # 计算image和text之间的loss
            # print("_________________")
            image_pos_words_idx = args[2]
            word_frequency_json = args[3]
            text_embeddings = x_sim_embeddings
            # print(text_embeddings.shape)
            all_embeddings = torch.cat([x_embeddings, text_embeddings], dim=0)
            representations = all_embeddings.unsqueeze(0).repeat(
                all_embeddings.shape[0], 1, 1
            )
            similarity_matrix, pairwise_dist = self.similarity_func(
                representations.transpose(0, 1), representations, self.min_dist
            )
            batch_size = self.batch_size
            # Compute positive indices and mask for all samples
            pos_indices_list = [
                torch.tensor(indices, dtype=torch.int).clone().detach() + batch_size
                for indices in image_pos_words_idx.values()
            ]
            pos_similarities = torch.stack(
                [
                    (
                        similarity_matrix[i, indices].mean()
                        if len(indices) > 0
                        else torch.tensor(0.0, device=device)
                    )
                    for i, indices in enumerate(pos_indices_list)
                ]
            )
            mask = torch.ones(
                (batch_size, len(text_embeddings)), dtype=torch.bool, device=device
            )
            for i in range(batch_size):
                mask[i, pos_indices_list[i] - batch_size] = (
                    False  # Exclude positive samples
                )
            neg_samples_per_image = self.text_batch_size - len(pos_indices_list[0])
            neg_similarities = torch.empty(
                (batch_size, neg_samples_per_image), device=device
            )

            word_frequency = list(word_frequency_json.values())
            for i in range(batch_size):
                # Get remaining word frequencies for negative sampling
                remaining_indices = torch.where(mask[i])[0]
                remaining_frequencies = torch.tensor([word_frequency[idx] for idx in remaining_indices.tolist()], dtype=torch.float, device=device) ** 0.75
                remaining_probabilities = remaining_frequencies / remaining_frequencies.sum()
                neg_indices = remaining_indices[torch.multinomial(remaining_probabilities, neg_samples_per_image, replacement=False)]
                # neg_indices = remaining_indices[
                #     torch.randperm(len(remaining_indices))[:neg_samples_per_image]
                # ]
                neg_similarities[i] = similarity_matrix[i, neg_indices + batch_size]
            pos_similarities = pos_similarities.view(batch_size, 1)
            logits = torch.cat((pos_similarities, neg_similarities), dim=1)
            # Pad the logits to the batch size
            pad_width = (0, batch_size - logits.shape[1])
            logits = F.pad(logits, pad_width)
            return logits
        elif flag == 4:
            indices = args[2]
            word_num_per_class = args[3]
            word_frequency = args[5]
            text_embeddings = x_sim_embeddings
            bz = x_embeddings.shape[0]
            all_embeddings = torch.cat([x_embeddings, text_embeddings], dim=0)
            representations = all_embeddings.unsqueeze(0).repeat(
                all_embeddings.shape[0], 1, 1
            )
            similarity_matrix, pairwise_dist = self.similarity_func(
                representations.transpose(0, 1), representations, self.min_dist
            )
            if self.datasets_name == "coco":
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

            # 优化 image_index_per_class 的创建
            if self.datasets_name == "coco":
                image_index_per_class = {class_list[i]: [] for i in range(10)}
            else:
                image_index_per_class = {
                    f"train_{class_list[i]}": [] for i in range(10)
                }
                image_index_per_class.update(
                    {f"gene_{class_list[i]}": [] for i in range(10)}
                )
            # image_index_per_class.update({f"gene_Bengal_1": []})
            # image_index_per_class.update({f"gene_Bengal_2": []}) # step 51

            # step 6
            # image_index_per_class.update({f"gene_Bengal_1": []})
            # image_index_per_class.update({f"gene_Bengal_2": []})
            # image_index_per_class.update({f"gene_Bengal_3": []})
            # image_index_per_class.update({f"gene_Birman_1": []})
            # image_index_per_class.update({f"gene_Bombay_1": []})
            # image_index_per_class.update({f"gene_Persian_1": []})
            # image_index_per_class.update({f"gene_Russian_blue_1": []})
            # image_index_per_class.update({f"gene_Bombay_1": []})
            for idx, indice in enumerate(indices):
                if self.datasets_name == "coco":
                    new_name = class_list[indice.item() // 1000]
                else:
                    if indice < 1000:
                        classidx = indice.item() // 100
                        new_name = f"train_{class_list[classidx]}"
                    elif indice < 3000:
                        classidx = (indice.item() - 1000) // 200
                        new_name = f"gene_{class_list[classidx]}"
                image_index_per_class[new_name].append(idx)
            text_size = text_embeddings.shape[0]
            pos_logits_list = []
            neg_logits_list = []
            zero_count = 0
            for i, item in enumerate(word_num_per_class):
                weight = {
                    k: v
                    for k, v in word_num_per_class[item].items()
                    if sum(word_num_per_class[item].values()) != 0
                }
                sorted_weight = sorted(weight.items(), key=lambda x: x[1], reverse=True)
                max_names, current_sum, half_sum = [], 0, sum(weight.values()) / 2
                for name, value in sorted_weight:
                    max_names.append(name)
                    current_sum += value
                    if current_sum > half_sum:
                        break
                lens = len(max_names)
                if lens == 0 or word_num_per_class[item][max_names[0]] == 0:
                    zero_count += 1
                    continue
                pos_similarities = torch.tensor(0.0).to(device).unsqueeze(0)
                max_name = max_names[0]  # 控制是否只取最大的类别
                pos_indices = image_index_per_class[max_name]
                if pos_indices:
                    pos_similarities = torch.concat(
                        [
                            pos_similarities,
                            (similarity_matrix[i + bz, pos_indices].mean()).unsqueeze(
                                0
                            ),
                        ],
                        dim=0,
                    )
                pos_similarities = pos_similarities.mean().unsqueeze(0)
                pos_logits_list.append(pos_similarities)
                neg_indices = [
                    indices
                    for name, indices in image_index_per_class.items()
                    if name != max_name and len(indices) > 0
                ]
                neg_similarities = [
                    (
                        similarity_matrix[i + bz, idx].mean()
                        if len(idx) > 0
                        else torch.tensor(0.0).to(device)
                    )
                    for idx in neg_indices
                ]
                neg_similarities = torch.stack(neg_similarities)
                neg_logits_list.append(neg_similarities)
            pos_logits_tensor = torch.cat(pos_logits_list, dim=0)
            neg_logits_tensor = torch.cat(neg_logits_list, dim=0)
            pos_logits = pos_logits_tensor.view(text_size - zero_count, -1)
            neg_logits = neg_logits_tensor.view(text_size - zero_count, -1)
            logits = torch.cat((pos_logits, neg_logits), dim=1)
            return logits

        elif flag == 1:  # 计算image和image之间的loss
            all_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
            representations = all_embeddings.unsqueeze(0).repeat(
                all_embeddings.shape[0], 1, 1
            )
            similarity_matrix, pairwise_dist = self.similarity_func(
                representations.transpose(0, 1), representations, self.min_dist
            )
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            positives = torch.cat([l_pos, r_pos]).view(all_embeddings.shape[0], 1)
            negatives = similarity_matrix[self.correlated_mask].view(
                all_embeddings.shape[0], -1
            )
            logits = torch.cat((positives, negatives), dim=1)
            # print(logits.shape)
            return logits
        else:
            text_postive_samples = args[2].values()
            texts = x_embeddings
            text_embeddings = x_sim_embeddings
            batch_size = texts.shape[0]
            all_embeddings = torch.cat([text_embeddings, text_embeddings], dim=0)
            representations = all_embeddings.unsqueeze(0).repeat(
                all_embeddings.shape[0], 1, 1
            )
            similarity_matrix, pairwise_dist = self.similarity_func(
                representations.transpose(0, 1), representations, self.min_dist
            )
            pos_indices_list = [
                torch.tensor(indices, dtype=torch.int).clone().detach() + batch_size
                for indices in text_postive_samples
            ]
            pos_similarities = torch.stack(
                [
                    similarity_matrix[i, pos_indices_list[i]].mean()
                    for i in range(batch_size)
                ]
            )
            mask = torch.ones(
                (batch_size, len(text_embeddings)), dtype=torch.bool, device=device
            )
            for i in range(batch_size):
                mask[i, pos_indices_list[i] - batch_size] = False
            # max_neg_samples_per_image = batch_size
            # neg_similarities = torch.empty((batch_size, max_neg_samples_per_image), device=device)
            for i in range(batch_size):
                remaining_indices = torch.where(mask[i])[0]
                count = remaining_indices.size(0)
                neg_indices = remaining_indices[
                    torch.randperm(len(remaining_indices))[:count]
                ]
                neg_similarity = similarity_matrix[i, neg_indices + batch_size]
                neg_similarity = F.pad(
                    neg_similarity, (0, batch_size - count - 1)
                ).unsqueeze(0)
                # print(neg_similarity.shape)
                neg_similarities = (
                    neg_similarity
                    if i == 0
                    else torch.cat((neg_similarities, neg_similarity), dim=0)
                )
            pos_similarities = pos_similarities.view(batch_size, -1)
            # print(pos_similarities.shape,neg_similarities.shape)
            logits = torch.cat((pos_similarities, neg_similarities), dim=1)
            pad_width = (0, batch_size - logits.shape[1])
            logits = F.pad(logits, pad_width)
            return logits

    def pearson_loss(self, M, P):
        M_mean = M.mean()
        P_mean = P.mean()
        M_centered = M - M_mean
        P_centered = P - P_mean
        # print(M_centered * P_centered)
        numerator = (M_centered * P_centered).sum()
        M_squared_sum = (M_centered**2).sum()
        P_squared_sum = (P_centered**2).sum()
        denominator = torch.sqrt(M_squared_sum) * torch.sqrt(P_squared_sum)
        correlation = numerator / denominator
        return -correlation

    def f(self, x):
        return torch.where(x >= 0, torch.zeros_like(x), -x)

    def L2_loss(self, TI, P_TI):
        n, m = TI.size(0), TI.size(1)
        # Create all pair combinations for j and k
        j_indices, k_indices = torch.tril_indices(m, m, offset=-1)

        # Broadcast to create the differences
        TI_diff = TI[:, j_indices] - TI[:, k_indices]
        P_TI_diff = P_TI[:, j_indices] - P_TI[:, k_indices]
        # print(TI_diff,P_TI_diff)
        # Compute the term-wise loss
        term_product = TI_diff * P_TI_diff
        loss_sum = self.f(term_product).sum()

        # Normalize by the L2 norm of P_TI
        P_TI_norm = torch.norm(P_TI)
        loss = loss_sum / (P_TI_norm)

        return -loss.mean()

    def data_context_map(
        self, img_embedding, txt_embedding, img_projection, txt_projection
    ):
        img0 = img_embedding
        txt0 = txt_embedding
        img_embedding = torch.flatten(img_embedding, start_dim=1)
        num = img_embedding.shape[0]
        img_matrix = img_embedding.repeat(num, 1, 1)
        txt_embedding = torch.flatten(txt_embedding, start_dim=1)
        num1 = txt_embedding.shape[0]
        txt_matrix = txt_embedding.repeat(num1, 1, 1)
        txt_matrix0 = txt_embedding.repeat(num, 1, 1)
        img_matrix0 = img_embedding.repeat(num1, 1, 1)
        img_dis = torch.norm(img_matrix - img_matrix.transpose(0, 1), dim=-1)
        txt_dis = torch.norm(txt_matrix - txt_matrix.transpose(0, 1), dim=-1)
        cross_dis = torch.norm(txt_matrix0 - img_matrix0.transpose(0, 1), dim=-1)
        #     cross_dis=1-torch.matmul(img0,txt0.T)
        img_dis = img_dis / torch.mean(img_dis)
        txt_dis = txt_dis / torch.mean(txt_dis)
        cross_dis = cross_dis / torch.mean(cross_dis)
        merge_dis0 = torch.cat((img_dis, cross_dis), dim=1)
        merge_dis1 = torch.cat((cross_dis.T, txt_dis), dim=1)
        merge_dis = torch.cat((merge_dis0, merge_dis1), dim=0)
        low_img0 = img_projection
        low_txt0 = txt_projection
        low_merge = torch.cat((img_projection, txt_projection), dim=0)
        # print(low_merge.shape)
        img_projection = torch.flatten(img_projection, start_dim=1)
        num = img_projection.shape[0]
        low_img_matrix = img_projection.repeat(num, 1, 1)
        txt_projection = torch.flatten(txt_projection, start_dim=1)
        num1 = txt_projection.shape[0]
        low_txt_matrix = txt_projection.repeat(num1, 1, 1)
        low_txt_matrix0 = txt_projection.repeat(num, 1, 1)
        low_img_matrix0 = img_projection.repeat(num1, 1, 1)
        low_img_dis = torch.norm(
            low_img_matrix - low_img_matrix.transpose(0, 1), dim=-1
        )
        low_txt_dis = torch.norm(
            low_txt_matrix - low_txt_matrix.transpose(0, 1), dim=-1
        )
        low_merge = torch.flatten(low_merge, start_dim=1)
        num2 = low_merge.shape[0]
        low_merge_matrix = low_merge.repeat(num2, 1, 1)
        low_merge_dis = torch.norm(
            low_merge_matrix - low_merge_matrix.transpose(0, 1), dim=-1
        )
        low_cross_dis = torch.norm(
            low_txt_matrix0 - low_img_matrix0.transpose(0, 1), dim=-1
        )
        img_corr = torch.mean(
            pearsonr(
                img_dis,
                low_img_dis,
            )[:, 0]
        )
        txt_corr = torch.mean(
            pearsonr(
                txt_dis,
                low_txt_dis,
            )[:, 0]
        )
        cross_corr = torch.mean(
            pearsonr(
                cross_dis,
                low_cross_dis,
            )[:, 0]
        )
        all_corr = torch.mean(
            pearsonr(
                merge_dis,
                low_merge_dis,
            )[:, 0]
        )
        return (
            -img_corr,
            -txt_corr,
            -cross_corr,
            -all_corr,
            merge_dis,
            img_dis,
            txt_dis,
            cross_dis,
            low_merge_dis,
            low_img_dis,
            low_txt_dis,
            low_cross_dis,
        )
