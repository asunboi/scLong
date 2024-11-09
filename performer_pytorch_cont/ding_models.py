from performer_pytorch_cont.performer_pytorch import *
import torch.nn as nn
from torch_geometric.nn import SGConv
import pandas as pd
import os 
from tqdm import tqdm
import networkx as nx
import pickle

class Gene2VecPositionalEmbedding_NoFreeze(nn.Module):
    def __init__(self, dim, max_seq_len, gene2vec_file = 'data/gene2vec_16906.npy'):
        super().__init__()
        gene2vec_weight = np.load(gene2vec_file)
        gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze = False)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)
    
class TwoLayerMLPEmb(nn.Module):
    def __init__(self, base_dim):
        super(TwoLayerMLPEmb, self).__init__()
        self.fc1 = nn.Linear(1, 50)   # Input size: 1, Output size: 100
        self.fc2 = nn.Linear(50, base_dim) # Input size: 100, Output size: 200
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class Gene_Ontology_GNN(nn.Module):
    def __init__(self, dim, max_seq_len, G_go, G_go_weight,
                 num_layers = 1,
                 device = 'cuda',
                use_gene2vec = True, 
                gene2vec_file = 'data/gene2vec_16906.npy'):
        super().__init__()
        if use_gene2vec:
            gene2vec_weight = np.load(gene2vec_file)
            gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
            gene2vec_weight = torch.from_numpy(gene2vec_weight).float()
            self.emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze = False) 
        else:
            self.emb = nn.Embedding(max_seq_len, dim, max_norm=True)
        self.G_go = G_go.to(device)
        self.G_go_weight = G_go_weight.to(device)
        self.num_layers = num_layers
        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(dim, dim, 1, add_self_loops = False))  #The graph must include self loops itself.

    def forward(self, x): #x: B, N or B, N, D
        t = torch.arange(x.shape[1], device=x.device) #N, [0, 1, ..., N-1]
        t_emb = self.emb(t) #N, D
        #print("t_emb.dtype: ", t_emb.dtype)
        #print("self.G_go.dtype: ", self.G_go.dtype)
        #print("self.G_go_weight.dtype: ", self.G_go_weight.dtype)
        for idx, layer in enumerate(self.sim_layers):
            t_emb = layer(t_emb, self.G_go, self.G_go_weight)
            if idx < self.num_layers - 1:
                t_emb = t_emb.relu() 
        return t_emb

def get_go_auto(gene_list, data_path):
    go_path = data_path
    if go_path[-10:] != "_graph.csv":
        go_path = data_path + "_graph.csv"

    if os.path.exists(go_path):
        return pd.read_csv(go_path)
    else:
        ## download gene2go.pkl
        with open(data_path + '.pkl', 'rb') as f:
            gene2go = pickle.load(f)

        gene2go = {i: list(gene2go[i]) for i in gene_list if i in gene2go}
        edge_list = []
        for g1 in tqdm(gene2go.keys()):
            if len(gene2go[g1]) == 0:
                continue
            for g2 in gene2go.keys():
                if len(gene2go[g2]) == 0:
                    continue
                edge_list.append((g1, g2, len(np.intersect1d(gene2go[g1], gene2go[g2]))/len(np.union1d(gene2go[g1], gene2go[g2]))))

        edge_list_filter = [i for i in edge_list if i[2] > 0]
        further_filter = [i for i in edge_list if i[2] > 0.1]
        df_edge_list = pd.DataFrame(further_filter).rename(columns = {0: 'gene1', 1: 'gene2', 2: 'score'})

        df_edge_list = df_edge_list.rename(columns = {'gene1': 'source', 'gene2': 'target', 'score': 'importance'})
        df_edge_list.to_csv(go_path, index = False)        
        return df_edge_list

class GeneSimNetwork():
    def __init__(self, edge_list, gene_list):
        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        node_map = {}
        for i, n in enumerate(self.gene_list):
            node_map[n] = i
        
        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                      self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        #self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
        
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

def get_similarity_network(gene_list, data_path, num_similar_genes_go_graph):
    df_jaccard = get_go_auto(gene_list, data_path)
    df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(num_similar_genes_go_graph + 1,['importance'])).reset_index(drop = True)
    return df_out #edge_list



class PerformerLM_GO(nn.Module):
    def __init__(
        self,
        *,
        # num_tokens,                         # num of tokens
        max_seq_len,                        # max length of sequence
        dim,                                # dim of tokens
        depth,                              # layers
        heads,                              # num of heads
        dim_head = 64,                      # dim of heads
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,                  # False: output is num of tokens, True: output is dim of tokens  //multiply final embeddings with token weights for logits, like gpt decoder//
        g2v_position_emb = True,            # priority: gene2vec, no embedding
        auto_check_redraw = True,
        qkv_bias = False,
        G_go = None,
        G_go_weight = None,
        go_num_layers = 1,
        device = 'cuda',
        go_use_gene2vec = True, 
        gene2vec_file = 'data/gene2vec_16906.npy'
    ):    
        
        super().__init__()
        print('cast_tuple')
        local_attn_heads = cast_tuple(local_attn_heads)
        print('finish')
        self.max_seq_len = max_seq_len
        # self.token_emb = nn.Embedding(num_tokens, dim)
        self.token_emb = TwoLayerMLP()
        print('g2v')
        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(dim, max_seq_len, gene2vec_file=gene2vec_file)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = torch.zeros_like
            self.layer_pos_emb = Always(None)
        self.go_conv = Gene_Ontology_GNN(dim = dim, 
                                         max_seq_len = max_seq_len, 
                                         G_go = G_go,
                                         G_go_weight = G_go_weight,
                                         num_layers = go_num_layers,
                                         device = device, 
                                         use_gene2vec = go_use_gene2vec,
                                         gene2vec_file = gene2vec_file)
        print('dropout')
        self.dropout = nn.Dropout(emb_dropout)
        print('performer')
        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias)
        self.norm = nn.LayerNorm(dim)
        # self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None
        self.to_out = nn.Linear(dim, 1) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, output_attentions = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(x.unsqueeze(dim=-1))
        if output_attentions:
            x.requires_grad_()    # used for attn_map output
        x += self.pos_emb(x)
        x += self.go_conv(x)
        x = self.dropout(x)

        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)

        if output_attentions:
            x, attn_weights = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)
            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x, attn_weights

            if exists(self.to_out):
                return self.to_out(x), attn_weights

            return (x @ self.token_emb.weight.t()), attn_weights
        else:
            x = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)

            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x

            if exists(self.to_out):
                x = self.to_out(x)
                return x

            return x @ self.token_emb.weight.t()


class DualEncoderSCFM(nn.Module):
    def __init__(
        self,
        max_seq_len,                        # max length of sequence
        top_seq_len = 2048,
        base_dim = 200,                                # dim of tokens
        mini_enc_depth = 1,                              # layers in mini encoder
        mini_enc_heads = 8,                              # num of heads in mini encoder
        mini_enc_dim_head = 64,                      # dim of heads in mini encoder
        large_dim = 1280,                                # dim of tokens in large encoder
        large_enc_depth = 50,                              # layers in large encoder
        large_enc_heads = 10,                              # num of heads in large encoder
        large_enc_dim_head = 128,                      # dim of heads in large encoder
        dec_depth = 1,                              # layers in decoder
        dec_heads = 8,                              # num of heads decoder
        dec_dim_head = 64,                      # dim of heads decoder
        mask_token_thres = -1, #values smaller than -1 is a mask
        ############ All default below ###########
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        g2v_position_emb = True,            # priority: gene2vec, no embedding
        auto_check_redraw = True,
        qkv_bias = False,
        ################### For gnn-go encoder ##########
        G_go = None,
        G_go_weight = None,
        go_num_layers = 1,
        device = 'cuda',
        go_use_gene2vec = True, 
        gene2vec_file = 'data/gene2vec_16906.npy'
    ):
        
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)
        self.max_seq_len = max_seq_len
        self.mask_token_thres = mask_token_thres
        self.top_seq_len = top_seq_len
        self.base_dim = base_dim
        self.large_dim = large_dim

        self.token_emb = TwoLayerMLPEmb(base_dim)
        self.token_norm = nn.LayerNorm(base_dim)
        self.mask_emb = nn.Embedding(1, base_dim, max_norm=True, dtype = torch.float32)

        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding_NoFreeze(base_dim, max_seq_len, gene2vec_file=gene2vec_file)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = torch.zeros_like
            self.layer_pos_emb = Always(None)
        self.go_conv = Gene_Ontology_GNN(dim = base_dim, 
                                         max_seq_len = max_seq_len, 
                                         G_go = G_go,
                                         G_go_weight = G_go_weight,
                                         num_layers = go_num_layers,
                                         device = device, 
                                         use_gene2vec = go_use_gene2vec,
                                         gene2vec_file = gene2vec_file)
        self.dropout = nn.Dropout(emb_dropout)

        self.mini_encoder = Performer(base_dim, mini_enc_depth, mini_enc_heads, mini_enc_dim_head, 
                                      local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias)

        self.base_to_large = nn.Linear(base_dim, large_dim)
        self.large_in_norm = nn.LayerNorm(large_dim)
        self.large_encoder = Performer(large_dim, large_enc_depth, large_enc_heads, large_enc_dim_head, 
                                       local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias)
        self.large_to_base = nn.Linear(large_dim, base_dim)
        self.large_to_base_out_norm = nn.LayerNorm(base_dim)
        
        self.decoder = Performer(base_dim, dec_depth, dec_heads, dec_dim_head, 
                                       local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias)
        self.decode_norm = nn.LayerNorm(base_dim)
        
        self.exp_to_out = nn.Linear(base_dim, 1)


    def check_redraw_projections(self):
        self.mini_encoder.check_redraw_projections()
        self.large_encoder.check_redraw_projections()
        self.decoder.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.mini_encoder.fix_projection_matrices_()
        self.large_encoder.fix_projection_matrices_()
        self.decoder.fix_projection_matrices_()

    
    def forward(self, x, 
                return_encodings = False, 
                output_attentions = False,
                **kwargs):
        if (return_encodings or output_attentions):
            output = {"top_encodings": None, 
                      "left_encodings": None,
                      "merged_encodings": None,
                      "merged_decodings": None,
                      "large_enc_attentions": None,
                      "mini_enc_attentions": None,
                      "dec_attentions": None}
        else:
            output = None
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        d, L = self.base_dim, self.top_seq_len
        _, top_indices = torch.topk(x, L, dim=1) #(B, L, ), 

        x_mask = (x <= self.mask_token_thres).to(torch.float32).unsqueeze(dim=-1) #b, n, 1

        ### token and positional embedding
        x_emb = self.token_emb(x.unsqueeze(dim=-1)) #b, n, base_dim
        mask_vecs = self.mask_emb(torch.zeros((b, 1), dtype= torch.long, device = device)) #b, 1, base_dim
        x_emb = (1 - x_mask) * x_emb + x_mask * mask_vecs
        
        x_emb = self.token_norm(x_emb) #b, n, base_dim
        
        x_emb += self.pos_emb(x_emb)
        x_emb += self.go_conv(x_emb)  # caused CUDA error: device-side assert triggered

        x_emb = self.dropout(x_emb)
        layer_pos_emb = self.layer_pos_emb(x_emb)

        ### Get top-L and left n- L embeddings for each row.
        x_emb_top = torch.gather(x_emb, 1, top_indices.unsqueeze(-1).expand(-1, -1, d)) #b, L, d
        top_mask = torch.zeros(b, n, dtype=torch.bool, device=x_emb_top.device)
        top_mask.scatter_(1, top_indices, True)
        x_emb_left = x_emb.masked_select(~top_mask.unsqueeze(-1)).view(b, n - L, d) #b, n - L, d

        ### Use Dual Encoder to encode them

        ### Large Encoder
        x_emb_top = self.base_to_large(x_emb_top) #b, n, D
        x_emb_top = self.large_in_norm(x_emb_top)
        #x_enc_top = self.large_encoder(x_emb_top, pos_emb = layer_pos_emb, output_attentions = False, **kwargs)
        if output_attentions:
            x_enc_top, output["large_enc_attentions"] = self.large_encoder(x_emb_top, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)
        else:
            x_enc_top = self.large_encoder(x_emb_top, pos_emb = layer_pos_emb, output_attentions = False, **kwargs)
        x_enc_top = self.large_to_base(x_enc_top) #b, n, d
        x_enc_top = self.large_to_base_out_norm(x_enc_top)

        ### Mini Encoder
        x_enc_left = self.mini_encoder(x_emb_left, pos_emb = layer_pos_emb, output_attentions = False, **kwargs)
        #if output_attentions:
        #    x_enc_left, output["mini_enc_attentions"] = self.mini_encoder(x_emb_left, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)

        ### Merge top and left encodings
        x_enc_merge = torch.empty_like(x_emb)
        x_enc_merge.scatter_(1, top_indices.unsqueeze(-1).expand(-1, -1, d), x_enc_top)
        x_enc_merge.masked_scatter_(~top_mask.unsqueeze(-1), x_enc_left) #b, n, d
        
        ### Decode the merged encodings:
        x_enc_merge += self.pos_emb(x_enc_merge)
        x_enc_merge += self.go_conv(x_enc_merge)
        #if output_attentions:
        #    x_dec_merge, output["dec_attentions"] = self.decoder(x_enc_merge, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)
        x_dec_merge = self.decoder(x_enc_merge, pos_emb = layer_pos_emb, output_attentions = False, **kwargs)
        x_dec_merge = self.decode_norm(x_dec_merge)
        exp_out = self.exp_to_out(x_dec_merge) #Pretrain

        if not return_encodings:
            if output_attentions: 
                return exp_out, output
            else:
                return exp_out
        else:
            output["top_encodings"] = x_enc_top
            output["left_encodings"] = x_enc_left
            output["merged_encodings"] = x_enc_merge
            output["merged_decodings"] = x_dec_merge
            return output
