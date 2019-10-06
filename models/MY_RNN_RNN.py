from .BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.autograd import Variable


class RNN_RNN(BasicModule):
    def __init__(self, args, embed=None):
        super(RNN_RNN, self).__init__(args)
        self.model_name = 'RNN_RNN'
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num
        P_V = args.pos_num
        P_D = args.pos_dim

        self.abs_pos_embed = nn.Embedding(P_V, P_D)
        self.rel_pos_embed = nn.Embedding(S, P_D)
        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(input_size=D,
                               hidden_size=H,
                               batch_first=True,
                               bidirectional=True)
        self.sent_RNN = nn.GRU(
            input_size=2 * H,
            hidden_size=H,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(2 * H, 2 * H)

        # Parameters of Classification Layer
        self.content = nn.Linear(2 * H, 1, bias=False)
        self.ranks = nn.Linear(1, 1, bias=False)
        self.cosine_first_sent = nn.Linear(1, 1, bias=False)
        self.salience = nn.Bilinear(2 * H, 2 * H, 1, bias=False)
        self.novelty = nn.Bilinear(2 * H, 2 * H, 1, bias=False)
        self.abs_pos = nn.Linear(P_D, 1, bias=False)
        self.rel_pos = nn.Linear(P_D, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    def max_pool1d(self, x, seq_lens):
        # x:[N,L,O_in]
        out = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index], :]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t, t.size(2)))
        out = torch.cat(out).squeeze(2)
        return out

    def avg_pool1d(self, x, seq_lens):
        # x:[N,L,O_in]
        out = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index], :]
            t = torch.t(t).unsqueeze(0)
            out.append(F.avg_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def page_rank_rel(self, valid_hidden):
        """
               PageRank value of the sentence based on the sentence map

               :param thres: int
                   Every two sentences are regarded relevant if their similarity is above a threshold.
               :return: dict
                   Dictionary of index nodes with PageRank as value.
               """
        G = nx.Graph()
        cosine = nn.CosineSimilarity(dim=0)

        # Build a sentence map.
        # Every two sentences are regarded relevant if their similarity is above a threshold.
        # Every two relevant sentences are connected with a unidirectional link.
        for i in range(len(valid_hidden[:-2])):
            for j in range(len(valid_hidden[i + 1:])):
                cosine_similarity_sentence_doc = cosine(valid_hidden[i], valid_hidden[j])
                # if cosine_similarity_sentence_doc > thres:
                G.add_edge(i, j)

        pr = nx.pagerank(G)

        return pr

    def forward(self, x, sent_features, doc_lens):
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        cosine = nn.CosineSimilarity(dim=0)
        x = self.embed(x)  # (N,L,D)
        # word level GRU
        H = self.args.hidden_size
        # word
        x = self.word_RNN(x)[0]  # (N,2*H,L)
        # word_out = self.avg_pool1d(x,sent_lens)
        word_out = self.max_pool1d(x, sent_lens)
        # make sent features(pad with zeros)
        x = self.pad_doc(word_out, doc_lens)

        # sent level GRU
        sent_out = self.sent_RNN(x)[0]  # (B,max_doc_len,2*H)
        # docs = self.avg_pool1d(sent_out,doc_lens)                               # (B,2*H)
        docs = self.max_pool1d(sent_out, doc_lens)  # (B,2*H)out
        probs = []
        for index, doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index, :doc_len, :]  # (doc_len,2*H)
            doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
           #doc_index = docs[index]
            pr = self.page_rank_rel(valid_hidden)
            s = Variable(torch.zeros(1, 2 * H))
            if self.args.device is not None:
                s = s.cuda()
            first_sentent = valid_hidden[0]
            for position, h in enumerate(valid_hidden):

                cosine_similarity = cosine(h, first_sentent)
                #cosine_similarity_sentence_doc = cosine(h, doc_index)
                h = h.view(1, -1)  # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)

                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                # h la bieu dien cua cau
                # doc là biểu diễn của document
                ranks = [pr.get(position, 0)]
                tensor_ranks = torch.FloatTensor(ranks).cuda()
                content = self.content(h)
                ranks_sentent = self.ranks(tensor_ranks.view(1, -1))
                cosine_first = self.cosine_first_sent(cosine_similarity.view(1, -1))
                #cosine_with_doc = self.cosine_similarity_sentence_doc(cosine_similarity_sentence_doc.view(1, -1))
                salience = self.salience(h, doc)
                novelty = -1 * self.novelty(h, F.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = F.sigmoid(
                    ranks_sentent  + cosine_first + content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob, h)
                probs.append(prob)
        return torch.cat(probs).squeeze()
