# ==========================================
# train_with_embedding.py
# 功能: 训练模型 + 保存 Loss + 保存 Node Embedding，用于可视化
# ==========================================

# ==========================================
# 1. 必要补丁 (必须放在最前面)
# ==========================================
import sys
import os
import networkx as nx
import pickle

# 修复 NetworkX 3.x 兼容性
if not hasattr(nx, 'from_numpy_matrix'):
    print("[Patch] 修复 NetworkX from_numpy_matrix ...")
    nx.from_numpy_matrix = nx.from_numpy_array

# 修复 RDKit six 兼容性
try:
    import six
    import rdkit
    if not hasattr(rdkit, 'six'):
        print("[Patch] 修复 RDKit.six ...")
        rdkit.six = six
        sys.modules['rdkit.six'] = six
except ImportError:
    pass

# ==========================================
# 2. 导入库
# ==========================================
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import GraphFlowModel
from dataloader import PretrainZinkDataset
import utils

# 设置随机种子
utils.set_seed(2019, torch.cuda.is_available())

# ==========================================
# 3. 配置类
# ==========================================
class Config:
    dataset = 'dataset3'
    path = 'data_preprocessed/dataset3'
    
    batch_size = 32
    edge_unroll = 12
    shuffle = True
    num_workers = 0  # Notebook 中必须为 0

    name = 'dataset3_model'
    deq_type = 'random'
    deq_coeff = 0.9
    num_flow_layer = 6
    gcn_layer = 3
    nhid = 128
    nout = 128
    st_type = 'sigmoid'
    sigmoid_shift = 2.0

    train = True
    save = True
    all_save_prefix = './'
    no_cuda = False
    learn_prior = False
    seed = 2019
    epochs = 5
    lr = 0.001
    weight_decay = 0.0
    dropout = 0.0
    is_bn = False
    is_bn_before = False
    scale_weight_norm = False
    divide_loss = False
    init_checkpoint = None
    show_loss_step = 100

    gen = True 
    gen_num = 1000
    gen_out_path = 'generated_dataset3.txt'
    temperature = 0.75
    min_atoms = 5

    max_atoms = 48  # 会在读取数据后自动更新

    @property
    def cuda(self):
        return not self.no_cuda and torch.cuda.is_available()
    
    @property
    def save_path(self):
        dir_path = os.path.join(self.all_save_prefix, 'save_pretrain', f'{self.st_type}_{self.dataset}_{self.name}')
        if self.save and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

args = Config()

# ==========================================
# 4. 读取数据与保存函数
# ==========================================
def save_model(model, optimizer, args, var_list, epoch=None):
    config_dict = {k: v for k, v in args.__class__.__dict__.items() if not k.startswith('__') and not isinstance(v, property)}
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(str(config_dict), f)

    epoch_str = str(epoch) if epoch is not None else ''
    path = os.path.join(args.save_path, f'checkpoint{epoch_str}')
    
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        path
    )
    print(f"模型已保存至: {path}")

def read_molecules(path):
    print(f'📥 Reading data from {path}...')
    if not os.path.exists(path + '_config.txt'):
        raise FileNotFoundError(f"找不到配置文件: {path}_config.txt")

    node_features = np.load(path + '_node_features.npy')
    adj_features = np.load(path + '_adj_features.npy')
    mol_sizes = np.load(path + '_mol_sizes.npy')

    with open(path + '_config.txt', 'r') as f:
        data_config = eval(f.read())

    all_smiles = []  # 可选，用于去重
    return node_features, adj_features, mol_sizes, data_config, all_smiles

# ==========================================
# 5. 训练器类
# ==========================================
class Trainer(object):
    def __init__(self, dataloader, data_config, args, all_train_smiles=None):
        self.dataloader = dataloader
        self.data_config = data_config
        self.args = args
        self.all_train_smiles = all_train_smiles

        self.max_size = self.data_config['max_size']
        self.node_dim = self.data_config['node_dim'] - 1
        self.bond_dim = self.data_config['bond_dim']

        print(f"初始化模型: Max Atoms={self.max_size}, Node Dim={self.node_dim}")
        self._model = GraphFlowModel(self.max_size, self.node_dim, self.bond_dim, self.args.edge_unroll, self.args)
        self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                                     lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.best_loss = 100.0
        self.start_epoch = 0
        self.loss_history = []

        if self.args.cuda:
            self._model = self._model.cuda()

    # --- 原 train_epoch, fit, save_embeddings 保持不变 ---

    # ==========================================
    # 生成分子
    # ==========================================
    def generate_molecules(self, num=None, out_path=None):
        if num is None:
            num = getattr(self.args, 'gen_num', 100)
        if out_path is None:
            out_path = getattr(self.args, 'gen_out_path', 'generated_dataset3.txt')

        self._model.eval()
        all_smiles = []
        cnt_mol = 0
        cnt_gen = 0
        print(f"Generating {num} molecules ...")

        while cnt_mol < num:
            smiles, _, num_atoms = self._model.generate(
                temperature=getattr(self.args, 'temperature', 0.75),
                mute=True,
                max_atoms=self.max_size,
                cnt=cnt_gen
            )
            cnt_gen += 1
            if num_atoms >= getattr(self.args, 'min_atoms', 5):
                all_smiles.append(smiles)
                cnt_mol += 1
                if cnt_mol % 20 == 0:
                    print(f' Generated: {cnt_mol}/{num}')

        # 保存生成分子
        with open(out_path, 'w') as f:
            for smi in all_smiles:
                f.write(smi + '\n')
        print(f"Molecules saved to {out_path}")

    def train_epoch(self, epoch_cnt):
        t_start = time.time()
        batch_losses = []
        self._model.train()

        for i_batch, batch_data in enumerate(self.dataloader):
            self._optimizer.zero_grad()
            
            inp_node_features = batch_data['node']
            inp_adj_features = batch_data['adj']

            if self.args.cuda:
                inp_node_features = inp_node_features.cuda()
                inp_adj_features = inp_adj_features.cuda()

            # Forward
            if self.args.deq_type == 'random':
                out_z, out_logdet, ln_var = self._model(inp_node_features, inp_adj_features)
                loss = self._model.log_prob(out_z, out_logdet)
            elif self.args.deq_type == 'variational':
                out_z, out_logdet, out_deq_logp, out_deq_logdet = self._model(inp_node_features, inp_adj_features)
                ll_node, ll_edge, ll_deq_node, ll_deq_edge = self._model.log_prob(out_z, out_logdet, out_deq_logp, out_deq_logdet)
                loss = -1. * ((ll_node-ll_deq_node) + (ll_edge-ll_deq_edge))
            else:
                raise ValueError('unsupported dequantization method')

            loss.backward()
            self._optimizer.step()
            batch_losses.append(loss.item())
            self.loss_history.append(loss.item())

            if (i_batch + 1) % self.args.show_loss_step == 0:
                print(f'   Step: {i_batch + 1} | Loss: {loss.item():.5f}')

        epoch_loss = sum(batch_losses) / len(batch_losses)
        print(f'Epoch: {epoch_cnt} | Avg Loss: {epoch_loss:.5f} | Time: {time.time()-t_start:.2f}s')
        return epoch_loss

    def save_embeddings(self, save_dir=None):
        """ 保存 Node Embedding 用于可视化 (支持不同节点数的分子) """
        if save_dir is None:
            save_dir = os.path.join(self.args.save_path, 'node_embeddings')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._model.eval()
        print("🔍 Extracting node embeddings batch by batch...")
        import pickle
        with torch.no_grad():
            for i_batch, batch_data in enumerate(self.dataloader):
                x = batch_data['node']
                adj = batch_data['adj']
                if self.args.cuda:
                    x, adj = x.cuda(), adj.cuda()

                ret = self._model(x, adj)
                out_z = ret[0] if isinstance(ret, tuple) else ret

                # 每个图节点数不同，保持列表结构
                batch_emb = [emb.cpu().numpy() for emb in out_z]

                save_path = os.path.join(save_dir, f'batch_{i_batch}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(batch_emb, f)

        print(f"All batch embeddings saved in {save_dir}")



    def fit(self):
        print('开始训练...')
        for epoch in range(self.args.epochs):
            cur_epoch = epoch + self.start_epoch
            epoch_loss = self.train_epoch(cur_epoch)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                if self.args.save:
                    var_list = {'cur_epoch': cur_epoch, 'best_loss': self.best_loss, 'loss_history': self.loss_history}
                    save_model(self._model, self._optimizer, self.args, var_list, epoch=cur_epoch)

        print("训练完成!")

# ==========================================
# 6. 主执行流程
# ==========================================
if __name__ == '__main__':
    # 1. 读取数据
    node_features, adj_features, mol_sizes, data_config, all_smiles = read_molecules(args.path)
    args.max_atoms = data_config['max_size']
    print(f"📊 Dataset Info: {node_features.shape}, Max Atoms: {args.max_atoms}")

    # 2. 构建 DataLoader
    train_dataset = PretrainZinkDataset(node_features, adj_features, mol_sizes)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=args.shuffle, num_workers=args.num_workers)

    # 3. 初始化 Trainer
    trainer = Trainer(train_dataloader, data_config, args, all_train_smiles=all_smiles)

    # 4. 训练
    if args.train:
        trainer.fit()
        # 保存 Node Embedding
        trainer.save_embeddings(save_dir=os.path.join(args.save_path, 'node_embeddings'))
        # 保存 loss
        np.save(os.path.join(args.save_path, 'loss_history.npy'), np.array(trainer.loss_history))
        print("Embeddings + Loss 已保存")

    # 5. 生成分子 (如果 args.gen=True)
    if getattr(args, 'gen', True):
        trainer.generate_molecules()
