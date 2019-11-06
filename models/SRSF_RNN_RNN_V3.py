from models.BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.autograd import Variable


# run with main.py
class SRSF_RNN_RNN_V3(BasicModule):
    def __init__(self, args, embed=None):
        super(SRSF_RNN_RNN_V3, self).__init__(args)
        self.model_name = 'SRSF_RNN_RNN_V3'
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
        self.sentent_features = nn.Linear(10, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    @staticmethod
    def page_rank_rel(valid_hidden, thres=0.1):
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
                if cosine_similarity_sentence_doc > thres:
                    G.add_edge(i, j)

        pr = nx.pagerank(G)

        return pr

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

    def forward(self, x, doc_lens, senten_lengths, numbericals, tf_idfs, stop_word_ratios, num_noun_adjs):
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
            pr = self.page_rank_rel(valid_hidden)
            #doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
            senten_lens_doc = senten_lengths[index]
            numberical = numbericals[index]
            tf_idf = tf_idfs[index]
            stop_word_ratio = stop_word_ratios[index]
            num_noun_adj = num_noun_adjs[index]
            centroidIndex = tf_idf.index(max(tf_idf))

            centroid_sentent = valid_hidden[centroidIndex]
            first_sentent = valid_hidden[0]
            length_sent = len(valid_hidden)
            s = Variable(torch.zeros(1, 2 * H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                cosine_similarity = cosine(h, first_sentent)
                centroid_similarity = cosine(h, centroid_sentent)
                h = h.view(1, -1)  # (1,2*H)
                # get position embeddings
                # abs_index = Variable(torch.LongTensor([[position]]))
                # if self.args.device is not None:
                #   abs_index = abs_index.cuda()
                # abs_features = self.abs_pos_embed(abs_index).squeeze(0)

                # rel_index = int(round((position + 1) * 9.0 / doc_len))
                # rel_index = Variable(torch.LongTensor([[rel_index]]))
                # if self.args.device is not None:
                #   rel_index = rel_index.cuda()
                # rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                # h la bieu dien cua cau
                # doc là biểu diễn của document
                # surface_features
                # Get length of sentence
                sent_len = senten_lens_doc[position]
                sent_numberical = numberical[position]
                sent_tf_idf = tf_idf[position]
                sent_stop = stop_word_ratio[position]
                sent_num_noun_adj = num_noun_adj[position]
                sent_position = (length_sent - position) / length_sent
                sent_score_page_rank = pr.get(position, 0)
                # Get doc_first
                doc_first = int(position == 0)
                if doc_first == 0:
                    doc_first = 0

                all_feature = [sent_position, sent_numberical, sent_len, doc_first, centroid_similarity, sent_tf_idf,
                               cosine_similarity, sent_score_page_rank, sent_stop, sent_num_noun_adj]
                feature = torch.FloatTensor(all_feature).cuda()

                prob = F.softmax(self.sentent_features(feature.view(1, -1)) + self.bias)
                s = s + torch.mm(prob, h)
                probs.append(prob)
        return torch.cat(probs).squeeze()
