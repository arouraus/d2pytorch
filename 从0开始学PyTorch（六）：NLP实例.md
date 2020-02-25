# 从0开始学PyTorch（六）：NLP实例

## 一、Word2Vector

### 1.为何不采用One-hot向量

one-hot编码原理：假设词典中不同词的数量（词典大小）为$N$，每个词可以和从0到$N-1$的连续整数一一对应。这些与词对应的整数叫做词的索引。假设一个词的索引为$i$，为了得到该词的`one-hot`向量表示，创建一个全0的长为$N$的向量，并将其第$i$位的设为1。这样一来，每个词就表示成了一个长度为$N$的向量，可以直接被神经网络使用。

但是一般不会使用`one-hot`来构建词向量，主要原因是，`one-hot`词向量无法准确表达不同词之间的相似度，例如常用的余弦相似度。

`Word2Vector`工具的提出正是为了解决这个问题。它将每个词表示成一个定长的向量，并使得这些向量能较好地表达不同词之间的相似和类比关系。`Word2Vector`工具包含了两个模型，即跳字模型(Skip-gram)和连续词袋模型(Continuous bag of words, CBOW)。

### 2.跳字模型（Skip-gram 模型）

**跳字模型假设基于某个词来生成它在文本序列周围的词**。

在`skip-gram`模型中，每个词被表示成两个$d$维向量，用来计算条件概率。假设这个词在词典中索引为$i$，当它为中心词时向量表示为$v_i \in \mathbb{R}^{d}$，而为背景词时表示为$u_i \in \mathbb{R}^{d}$。设中心词$w_c$在词典中索引为$c$，背景词$w_0$在词典中索引为$o$，给定中心词生成背景词的条件概率可以通过对向量内积做`softmax`运算而得到：
$$
P(w_0|w_c) = \frac{exp(u_0^Tv_c)}{\sum_{i \in v}exp(u_i^Tv_c)}
$$
其中，词典索引集$v = \{0,1,...,|v|-1\}$。假设给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$。假设给定中心词的情况下背景词的生成相互独立，当背景窗口大小为$m$时，`skip-gram`模型的似然函数即为给定任一中心词生成所有背景词的概率：
$$
\prod_{t=1}^{T}\prod_{-m \le j \le m,j \ne 0}P(w^{(t+j)}|w^{(t)})
$$
这里小于1和大于$T$的时间步可以忽略。

### 3.连续词袋模型（CBOW模型）

连续词袋模型与跳字模型相似，但与`skip-gram`模型最大的不同在于，**`CBOW`模型假设基于某中心词在文本序列前后的背景词来生成该中心词**。

因为连续词袋模型的背景词有多个，将这些背景词向量取平均，然后使用与`skip-gram`模型一样的方法来计算条件概率。设$v_i \in \mathbb{R}^d$和$u_i \in \mathbb{R}^d$分别表示词典中索引为$i$的词作为背景词和中心词的向量（注意符号的含义与skip-gram模型中的相反）。设中心词$w_c$在词典中索引为$c$，背景词$w_{o_1},...,w_{o_{2m}}$在词典中索引为$o_1,...,o_{2m}$，那么给定背景词生成中心词的条件概率：
$$
P(w_c|w_{o_1},...,w_{o_{2m}}) = \frac{exp(\frac{1}{2m}u_c^T(v_{o_1} + ... + v_{o_{2m}}))}{\sum_{i \in v}exp(\frac{1}{2m}u_i^T(v_{o_1} + ... + v_{o_{2m}}))}
$$
为了让符号更简单，记$W_0 = {w_{o_1},...,w_{o_{2m}}}$，且$\vec{v_0} = (v_{o_1} + ... + v_{o_{2m}})/(2m)$，那么上式可简写为：
$$
P(w_c|W_0) = \frac{exp(u_c^T\vec{v_0})}{\sum_{i \in v}exp(u_i^T\vec{v_0})}
$$
给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$，背景窗口大小为$m$。连续词袋模型的似然函数是由背景词生成任一中心词的概率：
$$
\prod_{t=1}^TP(w^{(t)}|w^{(t-m)},...,w^{(t-1)},w^{(t+1)},...,w^{(t+m)})
$$
不论是`skip-gram`模型还是`CBOW`模型，由于条件概率使用了softmax运算，每一步的梯度计算都包含词典大小数目的项的累加。对于含几十万或上百万词的较大词典，每次的梯度计算开销可能过大。为了降低计算复杂度，会采用近似训练方法，常用的有两种：负采样(negative sampling)或层序`softmax`(hierarchical softmax)。

- 负采样通过考虑同时含有正类样本和负类样本的相互独立事件来构造损失函数。其训练中每一步的梯度计算开销与采样的噪声词的个数线性相关。
- 层序`softmax`使用了二叉树，并根据根节点到叶节点的路径来构造损失函数。其训练中每一步的梯度计算开销与词典大小的对数相关。

## 二、文本分类

文本分类是自然语言处理的一个常见任务，它把一段不定长的文本序列变换为文本的类别。本次学习关注的是它的一个子问题：使用文本情感分类来分析文本作者的情绪，也叫作情感分析。

### 1.数据预处理

在数据预处理阶段，首先要读入文本数据集，在此过程中，一般真正的生产环境数据里是掺杂有脏数据的，需要使用正则表达式等手段剔除这些“噪声”；其次就是分词，英文数据及分词比较容易，一般根据空格进行分词就行了，但中文分词就不一样了，还需要考虑语义，最常用的工具有jieba分词，PKUSeg(北大开源的分词工具)等；

### 2.基于RNN的文本分类模型

在这个模型中，每个词先通过嵌入层得到特征向量。然后，使用双向循环神经网络对特征序列进一步编码得到序列信息。最后，将编码的序列信息通过全连接层变换为输出。具体来说，可以将双向LSTM在最初时间步和最终时间步的隐藏状态连接，作为特征序列的表征传递给输出层分类。

在下面实现的`BiRNN`中，`Embedding`实例即嵌入层，`LSTM`实例即为序列编码的隐藏层，`Linear`实例即生成分类结果的输出层。

```python
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Emdedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                              hidden_size=num_hiddens,
                              num_layers=num_layers,
                              bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层的输入
        self.decoder = nn.Linear(4*num_hiddens, 2)
        
    def forward(self, inputs):
        # inputs的形状是（批量大小，词数），因为LSTM需要将序列长度（seq_len）作为第一维，所以将输入转置后，再提取特征，输出形状为（词数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.permute(1,0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态
        # outputs形状是（词数，批量大小，2*隐藏神经元个数）
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连接初始时间步和最终时间步的隐藏状态作为全连接层的输入。它的形状为（批量大小，4*隐藏单元个数）
        encodings = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs
```

#### 2.1 加载预训练的词向量

可以直接使用公开的已训练好的词向量作为每个词的特征向量。

```python
glove_vocab = Vocab.Glove(name='6B', dim=100, cache=os.path.join(DATA_ROOT, "glove"))

def load_pretrained_embedding(words, pretrained_vocab):
    embed = torch.zeros(len(words),
    pretrained_vocab.vectors[0].shape[0]) 
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
        	idx = pretrained_vocab.stoi[word]
        	embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
        	oov_count += 0
    if oov_count > 0:
    	print("There are %d oov words.")
    return embed
    net.embedding.weight.data.copy_(
    load_pretrained_embedding(vocab.itos, glove_vocab))
    net.embedding.weight.requires_grad = False 
```

#### 2.2 训练并评价模型

```python
lr, num_epochs = 0.01, 5
# 要过滤不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, optimizer, device,
num_epochs)
```

最后，定义预测函数：

```python
def predict_sentiment(net, vocab, sentence):
    """sentence是词语的列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence],
    device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'
```







