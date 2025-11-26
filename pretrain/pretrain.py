import networkx as nx
import numpy as np
import random
from collections import defaultdict
import os, json
import copy
import torch
import transformers
import matplotlib.pyplot as plt
import itertools
from transformers import Trainer, TrainingArguments
from torch.utils.data import IterableDataset, get_worker_info, Dataset
from typing import Dict, Optional, Sequence
from sklearn.utils import shuffle
from dataclasses import dataclass
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union
import os
import argparse
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def add_edge(G, h, t, r):
    num_edges = 0
    if G.has_edge(h, t):
        if r not in G[h][t]['id']:
            G[h][t]['id'].append(r)
            num_edges += 1
        else:
            print('edge already exists')
    else:
        G.add_edge(h, t, id=[r])
        num_edges += 1
    print('add edge: ', (h, r, t), 'num edges: ', num_edges)
    return num_edges


def generate_rules(relations, num_rules, L_min, L_max, weighted=False, temperature=0.25):
    # Generate K acyclic logic rules with varying lengths
    dependency_graph = defaultdict(set)
    rules = []
    weights = []
    if weighted:
        for l in range(L_min, L_max + 1):
            weights.append(np.exp(-temperature*l))
        probs = np.array([w / sum(weights) for w in weights])
    else:
        weights = [1] * (L_max - L_min + 1)

    def has_cycle(start, visited, stack):
        """Detects if adding a new dependency introduces a cycle."""
        if start not in visited:
            visited.add(start)
            stack.add(start)
            print('visited: ', visited)
            print('stack: ', stack)
            for neighbor in dependency_graph[start]:
                if neighbor in stack:
                    return True
                elif has_cycle(neighbor, visited, stack):
                    return True
        if start in stack:
            stack.remove(start)
        return False

    for _ in range(num_rules):
        while True:
            if weighted:
                length = random.choices(range(L_min, L_max + 1), weights=weights)[0]
            else:
                length = random.randint(L_min, L_max)
            rule_relations = random.choices(relations, k = length + 1) # the first element is the implied relation
            valid_rule = True
            for i in range(1, len(rule_relations)):
                dependency_graph[rule_relations[0]].add(rule_relations[i])

                # Check for cycles
                if has_cycle(rule_relations[i], set(), set()):
                    valid_rule = False
                    for j in range(1, i + 1):
                        dependency_graph[rule_relations[0]].remove(rule_relations[j])
                    break

            if valid_rule:
                rules.append(tuple(rule_relations))
                break

    print('rules: ', rules)
    return rules

def get_node_types(rules, max_num_relations_per_node=3):
    # map node types to out relations
    node_types = {}
    # map out relations to node types
    r2node_types = defaultdict(list)
    for rule in rules:
        for i in range(len(rule)):
            node_type = len(node_types)
            if i == 0:
                node_types[node_type] = [rule[i], rule[1]]
                r2node_types[rule[i]].append(node_type)
                r2node_types[rule[1]].append(node_type)
            elif i == len(rule) - 1:
                node_types[node_type] = ['-' + rule[i], '-' + rule[0]]
                r2node_types['-' + rule[i]].append(node_type)
                r2node_types['-' + rule[0]].append(node_type)
            else:
                node_types[node_type] = ['-' + rule[i], rule[i+1]]
                r2node_types['-' + rule[i]].append(node_type)
                r2node_types[rule[i+1]].append(node_type)

    print(node_types)
    print(r2node_types)

    for num_rs in range(2, max_num_relations_per_node):
        possible_new_node_types = []
        for r in r2node_types:
            alt_rs = []
            for node_type in r2node_types[r]:
                for _r in node_types[node_type]:
                    if _r != r:
                        alt_rs.append(_r)
            alt_rs = list(set(alt_rs))
            for node_type in r2node_types[r]:
                if len(node_types[node_type]) == num_rs:
                    for _r in alt_rs:
                        if _r not in node_types[node_type]:
                            possible_new_node_types.append(tuple(sorted([_r] + list(node_types[node_type]))))
            print(possible_new_node_types)
            possible_new_node_types += list(set(possible_new_node_types))
        possible_new_node_types = list(set(possible_new_node_types))
        print(possible_new_node_types)

        for rs in possible_new_node_types:
            new_node_type = len(node_types)
            node_types[new_node_type] = list(rs)
            for _r in rs:
                r2node_types[_r].append(new_node_type)

    return node_types

def get_adj_out_relations(rules):
    adj = defaultdict(list)
    for rule in rules:
        for i in range(len(rule)):
            if i == 0:
                adj[rule[i]].append(rule[1])
                adj[rule[1]].append(rule[i])
            elif i == len(rule) - 1:
                adj['-' + rule[i]].append('-' + rule[0])
                adj['-' + rule[0]].append('-' + rule[i])
            else:
                adj['-' + rule[i]].append(rule[i+1])
                adj[rule[i+1]].append('-' + rule[i])
    return adj


def latent_rule_graph(num_rules=50, L_min=2, L_max=4, n=10000, m=10, n_r=200,
                      num_test=1000, num_train=150000, check_frequency=100,
                      power_law=False, initial_graph=None,
                      length_weighted=False, mcmc=0.2, temperature=0.25,
                      deductible_ratio=0.5):

    relations = ['P' + str(i) for i in range(n_r)]
    all_rules = generate_rules(relations, max(n_r//L_min, num_rules), L_min, L_max)
    r2rules = {}
    for rule in all_rules:
        if rule[0] not in r2rules:
            r2rules[rule[0]] = []
        r2rules[rule[0]].append(rule[1:])
    num_triples = 0
    repeated_entities = defaultdict(list) # map in relation to entities
    child_relations = []
    for rule in all_rules:
        child_relations += rule[1:]
    child_relations = list(set(child_relations))
    child_relations += ['-' + r for r in child_relations]
    deductible_rules = random.sample(all_rules, num_rules)
    if length_weighted:
        weights = [int(100*np.exp(-temperature*len(rule))) for rule in all_rules]
    else:
        weights = [1 for _ in all_rules]
    repeated_rules = []
    for rule, weight in zip(all_rules, weights):
        for _ in range(weight):
            repeated_rules.append(rule)
    random.shuffle(repeated_rules)
    adj = get_adj_out_relations(repeated_rules)
    all_deductibles = {}

    if initial_graph is None:
        # Default initial graph
        G = nx.DiGraph()
        node_id = 0
        min_repeated_entities = 0
        while min_repeated_entities < m:
            for rule in all_rules:
                source = 'Q' + str(node_id)
                node_id += 1
                h = source
                for r in rule[1:]:
                    t = 'Q' + str(node_id)
                    node_id += 1
                    num_triples += add_edge(G, h, t, r)
                    repeated_entities[r].append(t)
                    repeated_entities['-' + r].append(h)
                    h = t
                num_triples += add_edge(G, source, t, rule[0])
                repeated_entities[rule[0]].append(t)
                repeated_entities['-' + rule[0]].append(source)

            min_repeated_entities = min([len(set(repeated_entities[r])) for r in child_relations])
    else:
        if len(initial_graph) < m or len(initial_graph) > n:
            raise nx.NetworkXError(
                f"Initial graph needs between m={m} and n={n} nodes"
            )
        G = initial_graph.copy()
        node_id = len(G)

    if not power_law:
        repeated_entities = {r: list(set(repeated_entities[r])) for r in repeated_entities}

    # adding nodes
    while node_id < n:
        source = 'Q' + str(node_id)
        node_id += 1
        possible_relations = [_r for _r in adj if _r in child_relations]
        if len(possible_relations) == 0:
            print('no adj relations')
            break
        print('add child edge')
        chosen_edges = []
        stop = False
        for _ in range(m):
            it = 0
            while (r, t) in chosen_edges:
                r = random.choice(possible_relations)
                t = random.choice(repeated_entities[r])
                it += 1
                if it > 100:
                    print('failed to find edge')
                    stop = True
                    break
            if stop or len(possible_relations) == 0:
                break

            possible_relations = [_r for _r in adj[r] if _r in child_relations]
            chosen_edges.append((r, t))
            if r[0] == '-':
                num_triples += add_edge(G, t, source, r[1:])
                repeated_entities[r[1:]].append(source)
            else:
                num_triples += add_edge(G, source, t, r)
                repeated_entities['-' + r].append(source)
            repeated_entities[r].append(t)
            if len(possible_relations) == 0:
                print('no adj relations')
                break

        if not power_law:
            repeated_entities = {r: list(set(repeated_entities[r])) for r in repeated_entities}

        if node_id % check_frequency == 0 or node_id == n-1:
            # add deductibles
            all_nodes = list(G.nodes)
            random.shuffle(all_nodes)
            for h in all_nodes:
                for rule in deductible_rules:
                    head_list = [h]
                    r = rule[0]

                    for _r in rule[1:]:
                        next_head_list = []
                        for e_h in head_list:
                            if e_h not in G.nodes:
                                continue
                            for e_t in G[e_h]:
                                if _r in G[e_h][e_t]['id']:
                                    if random.random() < mcmc:
                                        next_head_list.append(e_t)
                        head_list = next_head_list

                    for t in head_list:
                        if (h, r, t) not in all_deductibles:
                            all_deductibles[(h, r, t)] = [rule]
                        elif rule not in all_deductibles[(h, r, t)]:
                            all_deductibles[(h, r, t)].append(rule)
                        if not G.has_edge(h, t) or r not in G[h][t]['id']:
                            print('add deductible edge')
                            add_edge(G, h, t, r)
                            num_triples += 1
                            repeated_entities[r].append(t)
                            repeated_entities['-' + r].append(h)

    atomic_triples = []
    deductible_triples = []
    for h, t in G.edges:
        for r in G[h][t]['id']:
            if (h, r, t) not in all_deductibles:
                atomic_triples.append((h, r, t))
            else:
                deductible_triples.append((h, r, t))
    random.shuffle(atomic_triples)
    random.shuffle(deductible_triples)
    assert len(atomic_triples) >= int(num_train * (1-deductible_ratio))
    # 需要足够的三元组：训练集 + uniform_test
    assert len(deductible_triples) >= int(num_train * deductible_ratio) + num_test, \
        f"需要至少 {int(num_train * deductible_ratio) + num_test} 个可推理三元组，实际只有 {len(deductible_triples)}"
    
    # 按照标准设置：先删除不在训练集中的边，然后确定测试集
    remove_triples = []
    train_atomic_triples = atomic_triples[:int(num_train * (1-deductible_ratio))]
    remove_triples += atomic_triples[int(num_train * (1-deductible_ratio)):]
    train_deductible_triples = deductible_triples[:int(num_train * deductible_ratio)]
    remove_triples += deductible_triples[int(num_train * deductible_ratio):]

    # Remove edges not in training set
    print(f"\nRemoving {len(remove_triples)} edges not in training set...")
    for h, r, t in remove_triples:
        _t = t
        if G.has_edge(h, _t):
            rs = G[h][_t]['id']
            if r in rs:
                if len(rs) == 1:
                    G.remove_edge(h, _t)
                else:
                    G[h][_t]['id'].remove(r)

    train_triples = train_deductible_triples + train_atomic_triples
    random.shuffle(train_triples)
    print("num train triples: ", len(train_triples))

    # 构建规则映射（用于 check_deductible）
    r2rule = {}
    for rule in deductible_rules:
        if rule[0] in r2rule:
            r2rule[rule[0]].append(rule[1:])
        else:
            r2rule[rule[0]] = [rule[1:]]

    def check_deductible(triple):
        """检查三元组是否可以通过规则推理得出（在当前图上）"""
        h, r, t = triple
        if r not in r2rule:
            return False
        alt_ts = []
        for rule in r2rule[r]:
            head_list = [h]
            for _r in rule:
                next_head_list = []
                for e_h in head_list:
                    if e_h not in G.nodes:
                        continue
                    for e_t in G[e_h]:
                        if _r in G[e_h][e_t]['id']:
                            next_head_list.append(e_t)
                head_list = next_head_list
            alt_ts += head_list
        if t in alt_ts:
            return True
        return False

    # Determine test sets after edge removal
    print("\nDetermining test sets after edge removal...")
    
    # Collect all deductible test candidates with their RULE lengths (true inference difficulty)
    test_start_idx = int(num_train * deductible_ratio)
    deductible_candidates = []
    
    for triple in deductible_triples[test_start_idx:]:
        if check_deductible(triple):
            if triple in all_deductibles:
                # Get the minimum rule length for this triple (number of inference steps)
                # Rule format: (implied_relation, step1, step2, ...) -> length = len(rule) - 1
                min_rule_len = min([len(rule) - 1 for rule in all_deductibles[triple]])
                deductible_candidates.append((triple, min_rule_len))
    
    print(f"Total deductible candidates: {len(deductible_candidates)}")
    
    # Split by RULE length (true inference difficulty) into two difficulty levels
    # Short: 1-2 inference steps, Long: 3+ inference steps
    short_path_candidates = [(t, l) for t, l in deductible_candidates if 1 <= l <= 2]
    long_path_candidates = [(t, l) for t, l in deductible_candidates if l >= 3]
    
    print(f"  Short (1-2 rule steps): {len(short_path_candidates)}")
    print(f"  Long (3+ rule steps): {len(long_path_candidates)}")
    
    # OOD-Long: long inference paths (Medium difficulty - empirically easier)
    random.shuffle(long_path_candidates)
    ood_medium_triples = [t for t, l in long_path_candidates[:num_test]]
    ood_medium_rules = [all_deductibles[triple] for triple in ood_medium_triples]
    
    # OOD-Short: short inference paths (Hard difficulty - empirically harder)
    random.shuffle(short_path_candidates)
    ood_hard_triples = [t for t, l in short_path_candidates[:num_test]]
    ood_hard_rules = [all_deductibles[triple] for triple in ood_hard_triples]
    
    # ID (Easy): memory test from training set
    id_triples = random.sample(train_triples, k=min(num_test, len(train_triples)))
    
    print(f"\nTest set sizes:")
    print(f"  ID (Easy, from training): {len(id_triples)}")
    print(f"  OOD-Long (Medium, 3+ steps): {len(ood_medium_triples)}")
    print(f"  OOD-Short (Hard, 1-2 steps): {len(ood_hard_triples)}")
    
    # Print average rule lengths
    if ood_medium_triples:
        avg_medium = np.mean([l for t, l in long_path_candidates[:len(ood_medium_triples)]])
        print(f"  Avg rule length (Medium): {avg_medium:.2f}")
    if ood_hard_triples:
        avg_hard = np.mean([l for t, l in short_path_candidates[:len(ood_hard_triples)]])
        print(f"  Avg rule length (Hard): {avg_hard:.2f}")
    
    return G, deductible_rules, train_triples, id_triples, ood_medium_triples, ood_medium_rules, ood_hard_triples, ood_hard_rules

class LatentRuleGraph:
    def __init__(self,
                 n=1000, n_r=40, m=5, n_rules=30, n_triples=10000,
                 num_test=1000, L_min=2, L_max=4, power_law=False,
                 length_weighted=False, mcmc=1.0,
                 temperature=0.25, deductible_ratio=0.5, seed=42):
        self.n = n
        self.n_r = n_r
        self.n_triples = n_triples
        self.n_rules = n_rules
        self.num_test = num_test
        self.L_min = L_min
        self.L_max = L_max
        self.power_law = power_law
        self.m = m
        self.length_weighted = length_weighted
        self.mcmc = mcmc
        self.temperature = temperature
        self.deductible_ratio = deductible_ratio
        self.seed = seed
        set_seed(seed)  # Set all random seeds for reproducibility
        self.G = nx.DiGraph()
        self.load_data()
        self.all_es = list(self.G.nodes)
        self.all_rs = set()
        for h, t, r_dict in self.G.edges(data=True):
            for r in r_dict['id']:
                self.all_rs.add(r)
        self.triple_complet_file = None

    def load_data(self):
        self.triples = []
        self.id_triples = []
        self.ood_medium_triples = []
        self.ood_hard_triples = []
        self.ood_medium_alt_ts = []
        self.ood_hard_alt_ts = []
        self.rules = []
        self.ood_medium_rules = []
        self.ood_hard_rules = []

        self.G, self.rules, self.triples, \
        self.id_triples, \
        self.ood_medium_triples, self.ood_medium_rules, \
        self.ood_hard_triples, self.ood_hard_rules = latent_rule_graph(
            num_rules=self.n_rules, L_min=self.L_min, L_max=self.L_max,
            n=self.n, n_r=self.n_r, m=self.m,
            num_test=self.num_test, num_train=self.n_triples,
            power_law=self.power_law,
            length_weighted=self.length_weighted, mcmc=self.mcmc,
            deductible_ratio=self.deductible_ratio, temperature=self.temperature)

        r2rule = {}
        for rule in self.rules:
            if rule[0] in r2rule:
                r2rule[rule[0]].append(rule[1:])
            else:
                r2rule[rule[0]] = [rule[1:]]

        def get_alt_ts(h, r, t):
            alt_ts = []
            if r not in r2rule:
                return alt_ts
            for rule in r2rule[r]:
                head_list = [h]
                for _r in rule:
                    next_head_list = []
                    for e_h in head_list:
                        if e_h not in self.G.nodes:
                            continue
                        for e_t in self.G[e_h]:
                            if _r in self.G[e_h][e_t]['id']:
                                next_head_list.append(e_t)
                    head_list = next_head_list
                alt_ts += head_list
            return alt_ts

        for h, r, t in self.ood_medium_triples:
            alt_ts = get_alt_ts(h, r, t)
            self.ood_medium_alt_ts.append(alt_ts)

        for h, r, t in self.ood_hard_triples:
            alt_ts = get_alt_ts(h, r, t)
            self.ood_hard_alt_ts.append(alt_ts)







class TrainDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
        self,
        graph, # generated graph
        tokenizer,
        seq_length=256,
        num_of_sequences=1024,
        chars_per_token=3.6,
        seed=42,
    ):
        super(TrainDataset, self).__init__()

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.epoch = 0
        self.current_size = 0
        self.num_buffer_sequences = num_of_sequences
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.seed = seed
        self.data = graph

        print("max buffer size: ", self.max_buffer_size)

    def set_epoch(self, worker_id):
        set_seed(self.seed + self.epoch + worker_id) # int(time.time())

    def triple2str(self, triple):
        if type(triple[0]) == int or type(triple[1]) == int or type(triple[2]) == int:
            return f'Q{triple[0]} P{triple[1]} Q{triple[2]}'
        else:
            return ' '.join(list(triple))

    def iter_fun(self, worker_id=0):
        num_sents = len(self.data.triples)
        while True:
            i = random.randint(0, num_sents-1)
            triple = self.data.triples[i]
            text = self.triple2str(triple) + '\n'
            if text is None:
                print("cannot translate ", triple, " into text.")
                continue
            yield text

    def __len__(self):
        return len(self.data.triples)

    def __iter__(self):
        more_examples = True
        try:
            worker_info = get_worker_info()
            print(worker_info)
            worker_id = worker_info.id
        except:
            worker_id = 0
        self.set_epoch(worker_id)
        iterator = self.iter_fun(worker_id=worker_id)
        print("worker id: ", )

        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    print("data buffer full")
                    break
                try:
                    buffer.append(next(iterator))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    self.epoch += 1
                    self.set_epoch(worker_id)
                    iterator = self.iter_fun()
                    print(f"Dataset epoch: {self.epoch}")
            # print(buffer[:3])

            input_lens = []
            random.shuffle(buffer)
            tokenized_inputs = self.tokenizer(buffer,
                                padding=False,
                                max_length=self.seq_length,
                                truncation=True)["input_ids"]
            for tokenized_input in tokenized_inputs:
                input_ids = tokenized_input + [self.tokenizer.eos_token_id]
                input_lens.append(len(input_ids))
                self.current_size += 1
                yield dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(input_ids))            
                
                
def model_path_map(model_name):
    return '../llms/' + model_name

def count_params(model):
    params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params

def compute_llama_param(l, h, v):
    d = 64 * h
    embd = d * v
    atten = 4*d*d
    mlp = 2*d*d*3
    ln = d
    return l * (atten + mlp + 2*ln) + ln + 2*embd

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                 padding_value=IGNORE_INDEX)

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # print("input_ids: ", input_ids)
        # print("labels: ", labels)
        # print("atten mask: ", attn_mask)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attn_mask,
        )

class BaseTokenizer:
    def __init__(self, n=1, vocab=None, padding_side='right', add_special_tokens=False):
        self.n = n
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.padding_side = padding_side
        self.add_special_tokens = add_special_tokens
        self.bos_token = '<BOS>'
        self.bos_token_id = self.vocab['<BOS>']
        self.eos_token = '<EOS>'
        self.eos_token_id = self.vocab['<EOS>']
        self.pad_token = '<PAD>'
        self.pad_token_id = self.vocab['<PAD>']
        self.unk_token = '<UNK>'
        self.unk_token_id = self.vocab['<UNK>']
        self.all_special_ids = [self.bos_token_id, self.eos_token_id,
                                self.pad_token_id, self.unk_token_id]
        self.all_special_tokens = self.all_special_tokens_extended = [
            self.bos_token, self.eos_token,
            self.pad_token, self.unk_token]

    def build_vocab(self):
        pass

    def tokenize(self, text: str, max_length: int):
        pass

    def encode(self, text, padding=False, max_length=1024, return_tensors=None, truncation=True):
        if type(text) == str:
            ids = [self.tokenize(text, max_length)]
        else:
            ids = []
            lens = []
            for t in text:
                _ids = self.tokenize(t, max_length)
                ids.append(_ids)
                lens.append(len(_ids))

            if padding:
                max_length = max(lens)
                for _ids in ids:
                    if len(_ids) < max_length:
                        if self.padding_side == 'left':
                            _ids = [self.pad_token_id] * (max_length - len(_ids)) + _ids
                        elif self.padding_side == 'right':
                            _ids += [self.pad_token_id] * (max_length - len(_ids))
                        else:
                            raise NotImplementedError

        if return_tensors == 'pt':
            ids = torch.tensor(ids)

        return ids

    def __call__(self, text, padding=False, max_length=1024, return_tensors=None, truncation=True, device='cpu'):
        if type(text) == str:
            ids = [self.tokenize(text, max_length)]
            attns = [[1] * len(ids[0])]
        else:
            ids = []
            attns = []
            lens = []
            for t in text:
                _ids = self.tokenize(t, max_length)
                ids.append(_ids)
                lens.append(len(_ids))
                attns.append([1] * len(_ids))

            if padding:
                max_length = max(lens)
                padded_ids = []
                padded_attns = []
                for _ids, attn in zip(ids, attns):
                    num_pad = max_length - len(_ids)
                    if self.padding_side == 'left':
                        padded_ids.append([self.pad_token_id] * num_pad + _ids)
                        padded_attns.append([0] * num_pad + attn)
                    elif self.padding_side == 'right':
                        padded_ids.append(_ids + [self.pad_token_id] * num_pad)
                        padded_attns.append(attn + [0] * num_pad)
                    else:
                        raise NotImplementedError
                ids = padded_ids
                attns = padded_attns

        if return_tensors == 'pt':
            ids = torch.tensor(ids).to(device)
            attns = torch.tensor(attns).to(device)

        return {"input_ids": ids, 'attention_mask': attns}

    def __len__(self):
        return len(self.vocab)

    def decode(self, token_ids, skip_special_tokens=False):
        if type(token_ids) == int:
            return self.rev_vocab[token_ids]
        else:
            out = ''
            for i in token_ids:
                if i == self.eos_token_id:
                    if not skip_special_tokens:
                        out += self.eos_token
                    break
                if skip_special_tokens and i in self.all_special_ids:
                    continue
                out += self.rev_vocab[i]
            return out

    def batch_decode(self, sequences, skip_special_tokens=False):
        out = []
        for token_ids in sequences:
            out.append(self.decode(token_ids, skip_special_tokens))
        return out

    def save_pretrained(self, output_dir):
        with open(f'{output_dir}/tokenizer.json', 'w') as wf:
            json.dump(self.vocab, wf, indent = 4)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, padding_side='right', trust_remote_code=False, revision=None):
        vocab_path = f"{pretrained_model_name_or_path}/tokenizer.json"
        if os.path.exists(vocab_path):
            vocab = json.load(open(vocab_path))
            n = 1
            for token in vocab:
                if token not in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']:
                    if '_' in token:
                        n = max(n, int(token.split('_')[1]) + 1)
                    else:
                        n = max(n, len(token))

            return cls(n, vocab, padding_side=padding_side)
        else:
            return cls(padding_side=padding_side)

class CharTokenizer(BaseTokenizer):
    def __init__(self, n=1, vocab=None, padding_side='right', add_special_tokens=False):
        super().__init__(n, vocab, padding_side, add_special_tokens)

    def build_vocab(self):
        vocab = {'Q':0, 'P':1}
        for i in range(10):
            vocab[str(i)] = i+2
        vocab_size = 12
        vocab['\n'] = vocab_size
        vocab_size += 1
        vocab[' '] = vocab_size
        vocab_size += 1
        vocab['-'] = vocab_size
        vocab_size += 1
        vocab['?'] = vocab_size
        vocab_size += 1
        vocab['<BOS>'] = vocab_size
        vocab_size += 1
        vocab['<EOS>'] = vocab_size
        vocab_size += 1
        vocab['<PAD>'] = vocab_size
        vocab_size += 1
        vocab['<UNK>'] = vocab_size

        return vocab

    def tokenize(self, text: str, max_length: int):
        ids = []
        for l in text.split('\n'):
            if len(l) == 0:
                continue
            for w in l.split():
                for c in w.strip():
                    if c not in self.vocab:
                        ids.append(self.unk_token_id)
                    else:
                        ids.append(self.vocab[c])
                ids.append(self.vocab[' '])
            ids.append(self.vocab['\n'])

        if self.add_special_tokens:
            ids.append(self.vocab['<EOS>'])
        else:
            ids = ids[:-2]
        # print(ids)
        if max_length < len(ids):
            return ids[:max_length]
        else:
            return ids
        
        
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
            # pad_to_multiple_of=8,
        )["input_ids"]
        for text in strings
    ]
    input_ids = labels = [tokenized[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def prepare_data(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length)
                                             for strings in (examples, sources)]
    eos = torch.tensor([tokenizer.eos_token_id])
    input_ids = [torch.cat((ids, eos)) for ids in examples_tokenized["input_ids"]]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train(train_dataset, model_name_or_path='llama-2-2', random_initialize=True,
          output_dir='.', bf16=False, GPU_ID=0, max_steps=1000):

    set_seed(42) # make sure use the same model initialization
    l, h, v = None, None, None

    if random_initialize:
        print("Random initializing...")
        model_name, l, h = model_name_or_path.split('-')
        l, h = int(l), int(h)
        d = 64 * h
        if model_name == 'llama':
            config = transformers.LlamaConfig(hidden_size=d,
                                            intermediate_size=2*d,
                                            num_attention_heads=h,
                                            num_hidden_layers=l)
        else:
            raise NotImplemented


        tokenizer = CharTokenizer()
        config.vocab_size = len(tokenizer.vocab)
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        print("vocab size: ", len(tokenizer.vocab))
        print("new config: ", config)

        v = config.vocab_size
        model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        print("embedding size: ", model.get_input_embeddings().weight.data.shape)
    else:
        print("Using pre-trained model weights...")

        tokenizer = CharTokenizer.from_pretrained(model_name_or_path)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path
        )

    if l is not None and h is not None and v is not None:
        print("theoretical # params: ", compute_llama_param(l, h, v))
    print("actual # params: ", count_params(model))

    train_dataset.tokenizer = tokenizer

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_args = TrainingArguments(bf16=bf16, max_steps=max_steps,
                      per_device_train_batch_size=32, eval_strategy="no",
                      save_steps=max_steps, save_total_limit=1, learning_rate=1e-4,
                      weight_decay=0.0, warmup_ratio=0.2, lr_scheduler_type="cosine",
                      logging_steps=1, output_dir=output_dir, report_to="none")

    trainer = Trainer(model=model, tokenizer=tokenizer, args=train_args,
                    train_dataset=train_dataset, data_collator=data_collator,
                    eval_dataset=None)

    if not random_initialize:
        print("resume training from: ", model_name_or_path)
        trainer.train(model_name_or_path)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=output_dir)
    return model, tokenizer


class EvalDataset(Dataset):

    def __init__(self,
                graph,
                tokenizer,
                split="ood_medium", # or "id", "ood_hard"
                num_options=10,
                use_rule_length=False,
                seed=42):

        super(EvalDataset, self).__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.eos_token = self.tokenizer.eos_token
        self.num_options = num_options
        self.num_test = graph.num_test
        set_seed(seed)
        self.path_length = []

        self.data = graph
        if split == 'id':
            self.triples = self.data.id_triples
            self.alt_ts = None  # ID test doesn't need alt_ts
        elif split == 'ood_medium':
            self.triples = self.data.ood_medium_triples
            self.alt_ts = self.data.ood_medium_alt_ts
        elif split == 'ood_hard':
            self.triples = self.data.ood_hard_triples
            self.alt_ts = self.data.ood_hard_alt_ts
        else:
            print("no such split: ", split)
            raise NotImplementedError

        if use_rule_length:
            print("using rule length")
            self.path_length = [min([len(rule) - 1 for rule in rules]) for rules in self.data.test_rules]

        self.get_data()
        if len(self.path_length) == 0:
            self.get_path_length()

    def get_path_length(self):
        for h, r, t in self.input_triples:
            try:
                l = nx.shortest_path_length(self.data.G, source=h, target=t)
            except nx.NetworkXNoPath:
                l = 0
            self.path_length.append(l)
        
        if len(self.path_length) > 0:
            avg_path = np.mean(self.path_length)
            min_path = np.min(self.path_length)
            max_path = np.max(self.path_length)
            print(f"  Path stats: avg={avg_path:.2f}, min={min_path}, max={max_path}")

    def get_data(self):
        self.input_text = []
        self.input_triples = []
        self.seen_ts = []
        self.options = []

        for idx, triple in enumerate(self.triples):
            h, r, t = triple

            if self.split == "id":
                # For ID test, use graph edges as seen targets
                seen_ts = []
                if h in self.data.G:
                    for e in self.data.G[h]:
                        if r in self.data.G[h][e]['id']:
                            seen_ts.append(e)
            else:
                # For OOD tests, use alt_ts from inference
                seen_ts = self.alt_ts[idx]
            self.seen_ts.append(seen_ts)

            question = h + ' ' + r + ' '
            ans = t

            options = [ans]
            for i in range(self.num_options-1):
                neg_e = random.choice(self.data.all_es)
                while neg_e == ans or neg_e in seen_ts:
                    neg_e = random.choice(self.data.all_es)
                options.append(neg_e)

            self.input_text.append(question)
            random.shuffle(options)
            self.options.append(options)

            self.input_triples.append(triple)

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, i):
        example = [self.input_text[i], self.input_triples[i], self.seen_ts[i]]
        example += [self.options[i]]
        example.append(self.path_length[i])
        return example
    
    
def eval_simple(eval_dataset, model, tokenizer, batch_size=16, max_length=64,
                num_test=100, device="cpu", verbose=False,
                normalize_by_len=True, compute_sparsity=True):

    model.eval().to(device)

    def collect_data(instances, device='cpu'):
        input_ids = instances["input_ids"]
        labels = instances["labels"]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attn_mask = input_ids.ne(tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
            attention_mask=attn_mask.to(device),
        )

    def compute_sparsity_metrics(last_h, valid_mask):
        B, T, H = last_h.shape
        vh = last_h[valid_mask]
        if vh.numel() == 0:
            return {}
        M = vh.size(0)
        
        h = vh.detach().cpu().float().numpy()
        
        metrics_list = {
            'l1_norm': [],
            'top5pct_energy': [],
            'top10pct_energy': [],
            'effective_rank': []
        }
        
        for i in range(h.shape[0]):
            sample_h = h[i]
            abs_h = np.abs(sample_h)
            
            # L1 Norm: sum of absolute values
            l1_norm = abs_h.sum()
            
            sorted_abs = np.sort(abs_h)[::-1]
            total_energy = (sorted_abs ** 2).sum()
            
            top5pct = max(1, int(0.05 * len(sorted_abs)))
            top10pct = max(1, int(0.10 * len(sorted_abs)))
            
            top5pct_energy = (sorted_abs[:top5pct] ** 2).sum() / total_energy if total_energy > 0 else 0
            top10pct_energy = (sorted_abs[:top10pct] ** 2).sum() / total_energy if total_energy > 0 else 0
            
            squared = sample_h ** 2
            normalized = squared / squared.sum() if squared.sum() > 0 else squared
            entropy = -(normalized * np.log(normalized + 1e-10)).sum()
            effective_rank = np.exp(entropy) / len(sample_h)
            
            metrics_list['l1_norm'].append(l1_norm)
            metrics_list['top5pct_energy'].append(top5pct_energy)
            metrics_list['top10pct_energy'].append(top10pct_energy)
            metrics_list['effective_rank'].append(effective_rank)
        
        avg_metrics = {key: (np.mean(values), M) for key, values in metrics_list.items()}
        return avg_metrics

    sparsity_accum = {
        'l1_norm_sum': 0.0,
        'top5pct_energy_sum': 0.0,
        'top10pct_energy_sum': 0.0,
        'effective_rank_sum': 0.0,
        'count': 0
    }

    num_choices = eval_dataset.num_options
    input_texts, output_texts, gts = [], [], []
    num_correct = num_all = 0
    gold_losses = []

    def flush_batch():
        nonlocal input_texts, output_texts, gts, num_correct, num_all, gold_losses
        nonlocal sparsity_accum

        if not input_texts:
            return
        
        data_dict = prepare_data(input_texts, output_texts, tokenizer, max_length)
        batch = collect_data(data_dict, device)

        outputs = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False, output_hidden_states=True, return_dict=True)

        logits = outputs.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        mask = labels.ne(IGNORE_INDEX)

        if compute_sparsity:
            last_h = outputs.hidden_states[-1][:, :-1, :]
            metrics_dict = compute_sparsity_metrics(last_h, mask)
            
            if metrics_dict:
                for key in ['l1_norm', 'top5pct_energy', 'top10pct_energy', 'effective_rank']:
                    value, M = metrics_dict[key]
                    sparsity_accum[f'{key}_sum'] += value * M
                sparsity_accum['count'] += M

        # ---- 标准 CE 评测 ----
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            torch.where(mask, labels, torch.zeros_like(labels)).reshape(-1),
            reduction='none'
        ).view(labels.size(0), labels.size(1))
        ce = ce * mask
        token_cnt = mask.sum(dim=1).clamp_min(1)
        seq_loss = ce.sum(dim=1) / (token_cnt if normalize_by_len else 1)

        B = seq_loss.size(0)
        assert B % num_choices == 0
        for i in range(B // num_choices):
            block = seq_loss[i*num_choices:(i+1)*num_choices]
            pred = int(torch.argmin(block))
            g_block = gts[i*num_choices:(i+1)*num_choices]
            gt = g_block.index(True)
            gold_losses.append(float(block[gt].item()))
            if pred == gt:
                num_correct += 1
            num_all += 1

        if verbose and num_all % 10 == 0:
            mg = (sum(gold_losses) / len(gold_losses)) if gold_losses else float("nan")
            msg = f"  Progress: {num_all}/{min(num_test, eval_dataset.num_test)}, "
            msg += f"Acc: {num_correct/num_all:.2%}, GoldLoss: {mg:.4f}"
            
            if compute_sparsity and sparsity_accum["count"] > 0:
                cnt = sparsity_accum["count"]
                msg += (
                    f"\n    [Sparsity] "
                    f"L1: {sparsity_accum['l1_norm_sum']/cnt:.2f}, "
                    f"Top5%: {sparsity_accum['top5pct_energy_sum']/cnt:.4f}, "
                    f"Top10%: {sparsity_accum['top10pct_energy_sum']/cnt:.4f}, "
                    f"EffRank: {sparsity_accum['effective_rank_sum']/cnt:.4f}"
                )
            print(msg)

        input_texts.clear()
        output_texts.clear()
        gts.clear()

    # ===== Main loop =====
    for idx, (q, triple, seen_t, opts, l) in enumerate(eval_dataset, start=1):
        if idx > min(num_test, getattr(eval_dataset, "num_test", idx)):
            break
        label = triple[-1]
        
        for op in opts:
            input_texts.append(q)
            output_texts.append(op)
            gts.append(op == label)
        if len(input_texts) >= batch_size:
            flush_batch()
    flush_batch()

    acc = num_correct / max(1, num_all)
    mean_gold_loss = (sum(gold_losses) / len(gold_losses)) if gold_losses else float("nan")

    sparsity_metrics = None
    if compute_sparsity and sparsity_accum["count"] > 0:
        cnt = sparsity_accum["count"]
        sparsity_metrics = {
            "l1_norm": sparsity_accum["l1_norm_sum"] / cnt,
            "top5pct_energy": sparsity_accum["top5pct_energy_sum"] / cnt,
            "top10pct_energy": sparsity_accum["top10pct_energy_sum"] / cnt,
            "effective_rank": sparsity_accum["effective_rank_sum"] / cnt,
        }
    
    return acc, mean_gold_loss, sparsity_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_size', type=str, default='llama-32-32', help='LLM size, e.g., llama-2-2, llama-32-32')
    parser.add_argument('--gpu_id', type=str, default="1", help='GPU ID to use, e.g., "0" or "0,1,2,3"')
    parser.add_argument('--steps', type=int, default=1, help='Training steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for KG generation')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    print(f"Using random seed: {args.seed} for KG generation")

    graph = LatentRuleGraph(
        n=4000,
        n_r=50,         
        n_triples=10000,
        n_rules=20,
        L_min=2,      
        L_max=5,
        power_law=True,
        deductible_ratio=0.5,  
        length_weighted=True, 
        m=6,
        num_test=1000,
        temperature=0.25,
        mcmc=1.0,
        seed=args.seed)
    
   
    
    train_dataset = TrainDataset(
        graph,
        tokenizer=None,
        seq_length=128,
        num_of_sequences=1024,
        chars_per_token=3.6,
        )
    
    output_dir = f'./{args.llm_size}-{args.steps}'
    print(f"Output directory: {output_dir}")
 
    model, tokenizer = train(train_dataset, model_name_or_path=args.llm_size, output_dir=output_dir, max_steps=args.steps)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Three difficulty levels (empirically determined)
    eval_dataset_easy = EvalDataset(graph, tokenizer, split="id", num_options=10)
    eval_dataset_medium = EvalDataset(graph, tokenizer, split="ood_medium", num_options=10)
    eval_dataset_hard = EvalDataset(graph, tokenizer, split="ood_hard", num_options=10)
    
    print("\n" + "="*80)
    print("Difficulty 1: Easy (ID - Training Set Memory)")
    print("="*80)
    acc_easy, mean_loss_easy, sp_easy = eval_simple(eval_dataset_easy, model, tokenizer, compute_sparsity=True, device="cuda")
    print(f"Accuracy: {acc_easy:.2%}")
    print(f"Mean Loss: {mean_loss_easy:.4f}")
    if sp_easy:
        print(f"Sparsity Metrics:")
        print(f"  - L1 Norm: {sp_easy['l1_norm']:.2f}")
        print(f"  - Top5% Energy: {sp_easy['top5pct_energy']:.4f}")
        print(f"  - Top10% Energy: {sp_easy['top10pct_energy']:.4f}")
        print(f"  - Effective Rank: {sp_easy['effective_rank']:.4f}")
    
    print("\n" + "="*80)
    print("Difficulty 2: Medium (OOD)")
    print("="*80)
    acc_medium, mean_loss_medium, sp_medium = eval_simple(eval_dataset_medium, model, tokenizer, compute_sparsity=True, device="cuda")
    print(f"Accuracy: {acc_medium:.2%}")
    print(f"Mean Loss: {mean_loss_medium:.4f}")
    if sp_medium:
        print(f"Sparsity Metrics:")
        print(f"  - L1 Norm: {sp_medium['l1_norm']:.2f}")
        print(f"  - Top5% Energy: {sp_medium['top5pct_energy']:.4f}")
        print(f"  - Top10% Energy: {sp_medium['top10pct_energy']:.4f}")
        print(f"  - Effective Rank: {sp_medium['effective_rank']:.4f}")
    
    print("\n" + "="*80)
    print("Difficulty 3: Hard (OOD)")
    print("="*80)
    acc_hard, mean_loss_hard, sp_hard = eval_simple(eval_dataset_hard, model, tokenizer, compute_sparsity=True, device="cuda")
    print(f"Accuracy: {acc_hard:.2%}")
    print(f"Mean Loss: {mean_loss_hard:.4f}")
    if sp_hard:
        print(f"Sparsity Metrics:")
        print(f"  - L1 Norm: {sp_hard['l1_norm']:.2f}")
        print(f"  - Top5% Energy: {sp_hard['top5pct_energy']:.4f}")
        print(f"  - Top10% Energy: {sp_hard['top10pct_energy']:.4f}")
        print(f"  - Effective Rank: {sp_hard['effective_rank']:.4f}")
    
    # Comparison analysis
    print("\n" + "="*80)
    print("Difficulty Comparison")
    print("="*80)
    print(f"Accuracy: Easy={acc_easy:.2%}, Medium={acc_medium:.2%}, Hard={acc_hard:.2%}")
    print(f"Accuracy Drop: Easy→Medium={acc_easy-acc_medium:+.2%}, Medium→Hard={acc_medium-acc_hard:+.2%}")
    if sp_easy and sp_medium and sp_hard:
        print(f"\nSparsity Comparison (L1 Norm):")
        print(f"  Easy={sp_easy['l1_norm']:.2f}, Medium={sp_medium['l1_norm']:.2f}, Hard={sp_hard['l1_norm']:.2f}")
        print(f"Sparsity Comparison (Top5% Energy):")
        print(f"  Easy={sp_easy['top5pct_energy']:.4f}, Medium={sp_medium['top5pct_energy']:.4f}, Hard={sp_hard['top5pct_energy']:.4f}")
        print(f"Sparsity Comparison (Top10% Energy):")
        print(f"  Easy={sp_easy['top10pct_energy']:.4f}, Medium={sp_medium['top10pct_energy']:.4f}, Hard={sp_hard['top10pct_energy']:.4f}")
        print(f"Sparsity Comparison (Effective Rank):")
        print(f"  Easy={sp_easy['effective_rank']:.4f}, Medium={sp_medium['effective_rank']:.4f}, Hard={sp_hard['effective_rank']:.4f}")

    # Delete output_dir
    os.system(f"rm -rf {output_dir}")
    print(f"Deleted output directory: {output_dir}")

    # Print model parameter count summary
    print("\n" + "="*80)
    print("Model Parameter Summary")
    print("="*80)
    model_name, l, h = args.llm_size.split('-')
    l, h = int(l), int(h)
    d = 64 * h
    v = len(tokenizer.vocab)
    
    theoretical_params = compute_llama_param(l, h, v)
    actual_params = count_params(model)
    
    print(f"Model Configuration: {args.llm_size}")
    print(f"  - Layers: {l}")
    print(f"  - Attention Heads: {h}")
    print(f"  - Hidden Size: {d}")
    print(f"  - Vocab Size: {v}")
    print(f"  - Intermediate Size: {2*d}")
    print(f"\nParameter Count:")
    print(f"  - Theoretical: {theoretical_params:,} ({theoretical_params/1e6:.2f}M / {theoretical_params/1e9:.2f}B)")
    print(f"  - Actual: {actual_params:,} ({actual_params/1e6:.2f}M / {actual_params/1e9:.2f}B)")
    print(f"  - Difference: {abs(theoretical_params - actual_params):,}")
    print("="*80)

    
    