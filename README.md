# 專題
[CSA](https://arxiv.org/pdf/2002.07338.pdf)


# passage Encoder
+ BiBLOSA


# CSA


# query Encoder
+ self-attention + S2T attention


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
    + CSA
        + not yet
+ data.py
    + 提供 HotpotQA資料
    + ITER內有三種屬性: Question / Context / Answer
    + wordEmbedding 為 glove6B300d
    + train.csv dev.csv
 