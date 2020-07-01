# 專題
[CSA](https://arxiv.org/pdf/2002.07338.pdf)
[BIBLOSA]()
[attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

# passage Encoder
+ BiBLOSA

# CSA
+ CSA(args,dx,dq)

# query Encoder

+ CSAEncoder(word_dim,n_head,n_hid,dropout,nlayers):
    + <multihead-attention + positionwiseFeedforward>*2
    + S2T attention

# Decoder


# Program

+ train.py
+ test.py
+ model.py
    + BibloSA
        + mBloSA(args,mask='fw'/'bw')
        + customizedModule: 線性轉換+w b 初始化
        + s2tSA (args,hidden_size): 輸出為R(de)
    + Encoder
        + self-attention
        + s2tSA (args,hidden_size)
    + CSA(self,args,dx,dq)
        + output: batch,sequence_len,word_dim
        + CrossAttention(de,dq,mode)
            + de is xi_dim, dq is q_dim , mode = 'mul' or 'add'
            + output 為 S = [[p0],[p1],[p2]...]
        + positionwise feedforward network
            + 
+ data.py
    + 提供 HotpotQA資料
    + ITER內有三種屬性: Question / Context / Answer
    + wordEmbedding 為 glove6B300d
    + train.csv dev.csv
    + 注意在batch時，要進行transpose(0,1) 才能變成[batch_size * sequence_len]
 
 # args
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--gpu', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), type=int)
parser.add_argument('--csa-mode',default='add',type = str)
parser.add_argument('--word-dim',default=300,type = int)
parser.add_argument('--block-size', default=-1, type=int)
parser.add_argument('--mSA-scalar', default=5.0, type=float)