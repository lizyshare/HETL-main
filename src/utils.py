import json
import collections
import numpy as np
import torch
import torch.utils.data.dataset as Dataset

"""
定义Vocalulary类：加载词汇表，转换词汇表和索引
"""
class Vocabulary(object):
    # 输入：词汇表文件路径，关系数量，实体数量
    def __init__(self, vocab_file, num_relations, num_entities):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.num_relations = num_relations
        self.num_entities = num_entities
        assert len(self.vocab) == self.num_relations + self.num_entities + 2

    # 输入：词汇表文件路径
    # 输出：词汇表字典（包含词汇token和索引index）
    def load_vocab(self, vocab_file):
        vocab = collections.OrderedDict()
        fin = open(vocab_file, encoding='utf-8')
        for num, line in enumerate(fin):
            items = line.strip().split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    # 输入字典vocab,转换成列表item输出
    def convert_by_vocab(self, vocab, items):
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    # 将词汇列表转换成索引列表
    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)

    # 将索引列表转换成词汇列表
    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)

    def __len__(self):
        return len(self.vocab)


"""
定义NaryExample类：表示一个N关系的实例
"""
class NaryExample(object):
    def __init__(self,
                 arity,
                 head,
                 relation,
                 tail,
                 auxiliary_info=None):
        self.arity = arity
        self.head = head
        self.relation = relation
        self.tail = tail
        self.auxiliary_info = auxiliary_info


"""
定义NaryFeature类：用于存储与训练相关的特征信息
"""
class NaryFeature(object):
    # 特征的唯一标识符、样本的唯一标识符、输入的词汇列表、输入的索引列表、输入的mask列表、mask的位置、mask的标签、mask的类型、特征的参数数量
    def __init__(self,feature_id, example_id,
                 input_tokens, input_ids,
                 input_mask, mask_position, mask_label,
                 mask_type, arity
                 ):
        self.feature_id = feature_id
        self.example_id = example_id
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.mask_position = mask_position
        self.mask_label = mask_label
        self.mask_type = mask_type
        self.arity = arity


"""
输入：文件的路径，最大实体数量
输出：examples 列表和 total_instance 计数器
功能：从指定的输入文件中读取 JSON 格式的数据，解析每一行数据，提取相关信息并创建 NaryExample 对象，同时，统计满足条件的实例总数
"""
def read_examples(input_file, max_arity):
    # examples（用于存储读取的示例）和 total_instance（用于统计实例总数）
    examples, total_instance = [], 0
    with open(input_file, "r", encoding='utf-8') as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip()) #  将每一行解析为 JSON 对象
            arity = obj["N"]
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()
                for attribute in sorted(obj.keys()):
                    if attribute in ("N", "relation", "subject", "object"):
                        continue
                    auxiliary_info[attribute] = sorted(obj[attribute])
            if arity <= max_arity:
                example = NaryExample(
                    arity=arity,
                    head=head,
                    relation=relation,
                    tail=tail,
                    auxiliary_info=auxiliary_info)
                examples.append(example)
                total_instance += (2 * (arity - 2) + 3)
    return examples, total_instance

"""
输入：NaryExample对象，词汇表，最大实体数量，最大序列长度
输出：特征列表
功能：针对N元关系实例，通过掩码的方式生成特征，并将其转换为NaryFeature对象，并存储在特征列表中
"""
def convert_examples_to_features(examples, vocabulary, max_arity, max_seq_length):
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, " \
        "and max_aux attribute-value pairs."

    features = [] # 用于存储特征的列表
    feature_id = 0
    for (example_id, example) in enumerate(examples):
        # get original input tokens and input mask
        hrt = [example.head, example.relation, example.tail]
        hrt_mask = [1, 1, 1]

        aux_q = []
        aux_q_mask = []
        aux_values = []
        aux_values_mask = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_q.append(attribute)
                    aux_q.append(value)
                    aux_q_mask.append(1)
                    aux_q_mask.append(1)

        # 如果 aux_q 长度小于 max_aux * 2，用 [PAD] 填充，并将其掩码设为0
        while len(aux_q) < max_aux * 2:
            aux_q.append("[PAD]")
            aux_q.append("[PAD]")
            aux_q_mask.append(0)
            aux_q_mask.append(0)
        assert len(aux_q) == max_aux * 2

        orig_input_tokens = hrt + aux_q
        orig_input_mask = hrt_mask + aux_q_mask
        assert len(orig_input_tokens) == max_seq_length and len(orig_input_mask) == max_seq_length

        # generate a feature by masking each of the tokens
        # 如果是[PAD]则跳过，否则将该位置的标记替换为[MASK]，并生成新的input_tokens、input_ids
        for mask_position in range(max_seq_length):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue
            mask_label = vocabulary.vocab[orig_input_tokens[mask_position]]
            mask_type = 1 if mask_position % 2 == 0 else -1

            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"
            input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
            assert len(input_tokens) == max_seq_length and len(input_ids) == max_seq_length

            feature = NaryFeature(
                feature_id=feature_id,
                example_id=example_id,
                input_tokens=input_tokens,
                input_ids=input_ids,
                input_mask=orig_input_mask,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                arity=example.arity)
            features.append(feature)
            feature_id += 1

    return features

"""
定义一个类，用于创建自定义的数据集
"""
class MultiDataset(Dataset.Dataset):
    def __init__(self, vocabulary: Vocabulary, examples, max_arity=2, max_seq_length=3):
        self.examples = examples
        self.vocabulary = vocabulary
        self.max_arity = max_arity
        self.max_seq_length = max_seq_length
        self.features = convert_examples_to_features(
            examples=self.examples,
            vocabulary=self.vocabulary,
            max_arity=self.max_arity,
            max_seq_length=self.max_seq_length)
        self.multidataset = []
        for feature in self.features:
            feature_out = [feature.input_ids] + [feature.input_mask] + \
                          [feature.mask_position] + [feature.mask_label] + [feature.mask_type]
            self.multidataset.append(feature_out)

    # 获取数据集长度，即数据集中样本的数量
    def __len__(self):
        return len(self.multidataset)

    # 获取第 index 个样本的特征
    def __getitem__(self, index):
        x = self.multidataset[index]
        batch_data = prepare_batch_data(x, self.vocabulary, self.max_arity, self.max_seq_length)
        return batch_data

"""
输入：输入数据的列表，词汇表，最大实体数量，最大序列长度
输出：处理后的输入数据，包括输入ID、输入掩码、掩码位置、掩码标签、掩码输出、边标签和查询类型
功能：对输入数据进行处理并返回
"""
def prepare_batch_data(inst, vocabulary: Vocabulary, max_arity, max_seq_length):
    # inst: [input_ids, input_mask, mask_position, mask_label, query_type]
    # 输入数据转换为int64类型的numpy数组
    input_ids = np.array(inst[0]).astype("int64")
    input_mask = np.array(inst[1]).astype("int64")
    mask_position = np.array(inst[2]).astype("int64")
    mask_label = np.array(inst[3]).astype("int64")
    query_type = np.array(inst[4]).astype("int64")

    # 生成输入掩码
    input_mask = np.outer(input_mask, input_mask).astype("bool")

    # 构建边标签
    # edge labels between input nodes (used for GRAN-hete)
    #     0: no edge
    #     1: relation-subject
    #     2: relation-object
    #     3: relation-attribute
    #     4: attribute-value
    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 2] + [3, 4] * max_aux)
    edge_labels.append([1, 0, 5] + [6, 7] * max_aux)
    edge_labels.append([2, 5, 0] + [8, 9] * max_aux)
    for idx in range(max_aux):
        edge_labels.append([3, 6, 8] + [11, 12] * idx + [0, 10] + [11, 12] * (max_aux - idx - 1))
        edge_labels.append([4, 7, 9] + [12, 13] * idx + [10, 0] + [12, 13] * (max_aux - idx - 1))
    edge_labels = np.asarray(edge_labels).astype("int64")
    mask_output = np.zeros(len(vocabulary.vocab)).astype("bool")
    if query_type == -1:
        mask_output[2:2 + vocabulary.num_relations] = True
    else:
        mask_output[2 + vocabulary.num_relations:] = True

    return input_ids, input_mask, mask_position, mask_label, mask_output, edge_labels, query_type

"""
输入：词汇表，训练集，超边丢弃率，设备号
输出：正向边、反向边和保留的超边数量
功能：构建一个图，包含超边和实体之间的连接关系
"""
def build_graph(vocabulary, examples, hyperedge_dropout, device):
    selected = int((1 - hyperedge_dropout) * len(examples))
    examples = examples[:selected]
    s, t = [], [] # 存储超边和实体的索引
    for hyperedge, example in enumerate(examples):
        L = [example.head, example.tail] # 将example的head和tail加入列表L
        if example.auxiliary_info: # 如果example有辅助信息，将其加入列表L
            for i in example.auxiliary_info.values():
                L += list(i)
        for entity in vocabulary.convert_tokens_to_ids(L):
            s.append(hyperedge)
            t.append(entity)
    forward_edge = torch.tensor([t, s], dtype=torch.long).to(device) # 构建正向边
    backward_edge = torch.tensor([s, t], dtype=torch.long).to(device) # 构建反向边
    return forward_edge, backward_edge, len(examples) # 返回正向边、反向边和超边数量