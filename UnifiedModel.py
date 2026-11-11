# Includes adaptations from POMO, Copyright (c) 2021 Yeong-Dae Kwon.
# Upstream MIT license: licenses/POMO-MIT.txt.
"""Shared attention generator for text-attributed COP instances."""

import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from TextEncoder import TextEncoder, ENCODER_DIM_DICT
from utils.utils import tag_prompt
from lncs.problems import TASKS


class UnifiedModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        if not model_params.get("text_encoder", True):
            raise ValueError("This release supports text-attributed inputs only.")
        self.model_params = model_params
        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.text_encoder = TextEncoder(
            model_params["llm"],
            device=model_params.get("device", "cpu"),
            root=model_params.get("cache_dir"),
            local_files_only=model_params.get("local_files_only", False),
            pooling=model_params.get("pooling", "legacy_mean"),
            dtype=model_params.get("llm_dtype", "float32"),
        )
        self.text_encoder.get_model()
        self.task_description = {}
        self.encoded_nodes = None

    def _instance_prompts(self, state, problem):
        if problem in ("CVRP", "VRPB"):
            instances = zip(state.depot_xy, state.node_xy, state.node_demand)
        elif problem == "SMTWTP":
            instances = zip(state.due_time, state.weights, state.processing_time)
        else:
            instances = state.problems
        return [
            tag_prompt(instance, problem=problem, task_description=True)
            for instance in instances
        ]

    def pre_forward(self, reset_state, problem):
        if problem not in TASKS:
            raise ValueError(f"Unsupported problem: {problem}")
        self.problem = problem
        prompts = self._instance_prompts(reset_state, problem)
        batch = len(prompts)
        node_count = len(prompts[0]) - 1
        self.problem_size = (
            node_count - 1 if problem in ("CVRP", "VRPB") else node_count
        )
        self.last_step_nodes = None
        self.text_encoder.batch_size = max(1, self.problem_size // 2)
        dimension = ENCODER_DIM_DICT[self.model_params["llm"]]
        task_rows = []
        node_prompts = []
        for row in prompts:
            task_text = row[0]
            # A graph's task description can include its edge count; cache the
            # exact text rather than borrowing another instance's metadata.
            key = problem + ":" + hashlib.sha256(task_text.encode()).hexdigest()
            if key not in self.task_description:
                self.task_description[key] = (
                    self.text_encoder.encode([task_text])
                    .reshape(1, 1, dimension)
                    .detach()
                )
            task_rows.append(self.task_description[key])
            node_prompts.extend(row[1:])
        self.task_embed = torch.cat(task_rows, dim=0)
        node_embed = self.text_encoder.encode(node_prompts).reshape(
            batch, node_count, dimension
        )
        self.encoded_nodes = self.encoder(node_embed.to(self.encoder.embedding.weight))
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch, pomo = state.BATCH_IDX.shape
        routing = self.problem in ("CVRP", "VRPB")
        if routing and state.selected_count == 0:
            selected = torch.zeros(
                (batch, pomo), dtype=torch.long, device=self.encoded_nodes.device
            )
            self.decoder.set_q1(self.encoded_nodes[:, [0], :], self.task_embed)
            return selected, self.encoded_nodes.new_ones(batch, pomo)
        if routing and state.selected_count == 1:
            selected = (
                state.START_NODE
                if self.problem == "VRPB"
                else torch.arange(1, pomo + 1, device=self.encoded_nodes.device)[
                    None, :
                ].expand(batch, pomo)
            )
            return selected, self.encoded_nodes.new_ones(batch, pomo)
        if not routing and state.current_node is None:
            selected = torch.arange(pomo, device=self.encoded_nodes.device)[
                None, :
            ].expand(batch, pomo)
            self.decoder.set_q1(
                _get_encoding(self.encoded_nodes, selected), self.task_embed
            )
            self.last_step_nodes = selected
            return selected, self.encoded_nodes.new_ones(batch, pomo)
        current = state.current_node
        if self.problem == "KP":
            current = torch.where(
                current >= self.problem_size, self.last_step_nodes, current
            )
            self.last_step_nodes = current
        last = _get_encoding(self.encoded_nodes, current)
        capacity = (
            state.load if routing else state.capacity if self.problem == "KP" else None
        )
        probs = self.decoder(last, capacity, state.ninf_mask)
        if not torch.isfinite(probs).all() or (probs.sum(-1) <= 0).any():
            raise FloatingPointError(f"Invalid action probabilities for {self.problem}")
        if (
            not self.training
            and self.model_params.get("eval_type", "argmax") == "argmax"
        ):
            return probs.argmax(2), None
        # Bound the retry retained for historical multinomial zero-probability
        # selections; an invalid distribution must fail rather than hang.
        for _ in range(3):
            selected = (
                probs.reshape(batch * pomo, -1).multinomial(1).reshape(batch, pomo)
            )
            probability = probs[state.BATCH_IDX, state.POMO_IDX, selected]
            if (probability > 0).all():
                return selected, probability
        raise FloatingPointError("Sampling selected a masked action repeatedly")


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding = nn.Linear(
            ENCODER_DIM_DICT[model_params["llm"]], model_params["embedding_dim"]
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(**model_params)
                for _ in range(model_params["encoder_layer_num"])
            ]
        )

    def forward(self, data):
        out = self.embedding(data)
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params["head_num"]

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        dim = model_params["embedding_dim"]
        width = model_params["head_num"] * model_params["qkv_dim"]
        self.Wq_first = nn.Linear(dim, width, bias=False)
        self.Wq_last = nn.Linear(dim + 1, width, bias=False)
        self.Wq_task = nn.Linear(
            ENCODER_DIM_DICT[model_params["llm"]], width, bias=False
        )
        self.Wk = nn.Linear(dim, width, bias=False)
        self.Wv = nn.Linear(dim, width, bias=False)
        self.multi_head_combine = nn.Linear(width, dim)

    def set_kv(self, encoded_nodes):
        heads = self.model_params["head_num"]
        self.k = reshape_by_heads(self.Wk(encoded_nodes), heads)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), heads)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, first, task_embed):
        heads = self.model_params["head_num"]
        self.q_first = reshape_by_heads(self.Wq_first(first), heads)
        self.q_task = reshape_by_heads(
            self.Wq_task(task_embed.to(self.Wq_task.weight)), heads
        )

    def forward(self, last, capacity, ninf_mask):
        dynamic = (
            last.new_zeros(*last.shape[:2], 1)
            if capacity is None
            else capacity.unsqueeze(-1)
        )
        query = reshape_by_heads(
            self.Wq_last(torch.cat((last, dynamic), dim=-1)),
            self.model_params["head_num"],
        )
        query = self.q_first + query + self.q_task
        attended = multi_head_attention(query, self.k, self.v, ninf_mask)
        pointer = self.multi_head_combine(attended)
        score = (
            torch.matmul(pointer, self.single_head_key)
            / self.model_params["sqrt_embedding_dim"]
        )
        score = self.model_params["logit_clipping"] * torch.tanh(score)
        return F.softmax(score + ninf_mask, dim=-1)


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(
        batch_size, pomo_size, embedding_dim
    )
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch, head_num, n, TSP_SIZE)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(
            batch_s, head_num, n, problem_s
        )

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
