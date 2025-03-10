from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import KeyedVectors
import jieba
import torch
from transformers import AutoTokenizer, AutoModel

# nltk.download("stopwords")
cachedStopWords = stopwords.words("chinese")

def transliterate(line):
    cedilla2latin = [[u'Á', u'A'], [u'á', u'a'], [u'Č', u'C'], [u'č', u'c'], [u'Š', u'S'], [u'š', u's']]
    tr = dict([(a[0], a[1]) for (a) in cedilla2latin])
    new_line = ""
    for letter in line:
        if letter in tr:
            new_line += tr[letter]
        else:
            new_line += letter
    return new_line


def text_cleaner(text,
                 deep_clean=False,
                 stem=True,
                 stop_words=True,
                 translite_rate=True):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning

    ]

    if deep_clean:
        text = text.replace(".", "")
        text = text.replace("[", " ")
        text = text.replace(",", " ")
        text = text.replace("]", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("\"", "")
        text = text.replace("-", " ")
        text = text.replace("=", " ")
        text = text.replace("?", " ")
        text = text.replace("!", " ")

        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
        text = text.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
        text = re.sub("(^|\W)\d+($|\W)", " ", text)
        if translite_rate:
            text = transliterate(text)
        if stem:
            text = PorterStemmer().stem(text)
        text = WordNetLemmatizer().lemmatize(text)
        if stop_words:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            text = [w for w in word_tokens if not w in stop_words]
            text = ' '.join(str(e) for e in text)
    else:
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
    return text.lower()

def loadData_Tokenizer(X_train, X_test, Word2Vec_PATH, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test), axis=0)
    text = np.array(text)

    # 确保所有文本都是UTF-8编码的
    text = [str(sentence).encode('utf-8', 'ignore').decode('utf-8') for sentence in text]

    text = [' '.join(jieba.cut(sentence)) for sentence in text]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    text = text[indices]
    print(text.shape)
    X_train = text[0:len(X_train), ]
    X_test = text[len(X_train):, ]

    # Load pre-trained Word2Vec model
    embeddings_index = {}
    with open(Word2Vec_PATH, 'r', encoding='latin1') as f:  # 修改编码为latin1
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings_index[word] = vector
    print('Total %s word vectors.' % len(embeddings_index))
    return (X_train, X_test, word_index, embeddings_index)


def loadData(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with", str(np.array(X_train).shape[1]), "features")
    return (X_train, X_test)
def loadData_BGE(train_sentences, test_sentences, max_sequence_length):
    # 初始化BERT tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained("/public/home/xxxy4/PWHSB/BGE")
    model = AutoModel.from_pretrained("/public/home/xxxy4/PWHSB/BGE")
   # pretrained_weights = torch.load('/public/home/xxxy4/PWHSB/pytorch_model.bin',weights_only=True)
    # 将预训练的权重加载到模型中
  #  model.load_state_dict(pretrained_weights, strict=False)
    # 将模型移动到 GPU 上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    instruction = "为这个句子生成表示以用于情绪匹配："
    train_sentences=[instruction + q for q in train_sentences]
    test_sentences=[instruction + q for q in test_sentences]
    # 对训练集和测试集的文本进行编码
    train_encodings = tokenizer(train_sentences, truncation=True, padding='max_length', return_tensors='pt', max_length=max_sequence_length)
    test_encodings = tokenizer(test_sentences, truncation=True, padding='max_length', return_tensors='pt', max_length=max_sequence_length)

    
    # 将编码的输入移动到 GPU 上
    train_encodings = {key: value.to(device) for key, value in train_encodings.items()}
    test_encodings = {key: value.to(device) for key, value in test_encodings.items()}
    
    with torch.no_grad():
        train_features = model(**train_encodings)[0]
        test_features = model(**test_encodings)[0]
    
    # 将特征移动回 CPU 并返回特征表示和词汇表
    train_features = train_features.cpu()
    test_features = test_features.cpu()
    print(train_features.shape)  # Should be [batch_size, max_sequence_length, hidden_size]
    print(test_features.shape)
    return train_features, test_features, tokenizer.get_vocab()
def loadData_BERT(train_sentences, test_sentences, max_sequence_length):
    # 初始化BERT tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained("/public/home/xxxy4/PWHSB/bert")
    model = AutoModel.from_pretrained("/public/home/xxxy4/PWHSB/bert")

    # 对训练集和测试集的文本进行编码
    train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=max_sequence_length)
    test_encodings = tokenizer(test_sentences, truncation=True, padding=True, max_length=max_sequence_length)

    # 将编码转换为PyTorch张量
    train_input_ids = torch.tensor(train_encodings['input_ids'], dtype=torch.long)
    train_attention_mask = torch.tensor(train_encodings['attention_mask'], dtype=torch.long)
    test_input_ids = torch.tensor(test_encodings['input_ids'], dtype=torch.long)
    test_attention_mask = torch.tensor(test_encodings['attention_mask'], dtype=torch.long)

    # 获取文本的BERT特征表示
    with torch.no_grad():
        train_features = model(train_input_ids, attention_mask=train_attention_mask)[0]
        test_features = model(test_input_ids, attention_mask=test_attention_mask)[0]

    # 将特征表示转换为NumPy数组
    train_features = train_features.numpy()
    test_features = test_features.numpy()

    # 返回特征表示和词汇表
    return train_features, test_features, tokenizer.get_vocab()

# 使用BERT提取词语的语义信息
def loadCSV_sentiment():
    # 读取情感词典
    file_path = '/public/home/xxxy4/PWHSB/情感词汇本体.xlsx'
    df = pd.read_excel(file_path)

    # 提取必要的信息
    words = df['词语'].tolist()
    emotion_categories = df['情感分类'].tolist()
    intensities = df['强度'].tolist()
    pos_tags = df['词性种类'].tolist()
    polarities = df['极性'].tolist()
    main_emotions = df['辅助情感分类'].tolist()

    # 独热编码器
    emotion_encoder = OneHotEncoder(sparse_output=False)
    pos_encoder = OneHotEncoder(sparse_output=False)
    polarity_encoder = OneHotEncoder(sparse_output=False)
    main_emotion_encoder = OneHotEncoder(sparse_output=False)

    # 独热编码情感类别、词性、极性和情感大类
    emotion_onehot = emotion_encoder.fit_transform(np.array(emotion_categories).reshape(-1, 1))
    pos_onehot = pos_encoder.fit_transform(np.array(pos_tags).reshape(-1, 1))
    polarity_onehot = polarity_encoder.fit_transform(np.array(polarities).reshape(-1, 1))
    main_emotion_onehot = main_emotion_encoder.fit_transform(np.array(main_emotions).reshape(-1, 1))

    # 将情感强度归一化，并适当增大其权重
    scaler = MinMaxScaler()
    intensities_normalized = scaler.fit_transform(np.array(intensities).reshape(-1, 1)) * 2

    # 使用BERT提取词语的语义信息
    tokenizer = AutoTokenizer.from_pretrained("/public/home/xxxy4/PWHSB/BGE")
    bert_model = AutoModel.from_pretrained("/public/home/xxxy4/PWHSB/BGE")

    # 将模型移动到 GPU 上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    bert_model.eval()

    cleaned_words = [str(word) for word in words if pd.notna(word)]

    # 使用BERT提取词语的语义信息
    word_embeddings = []
    for word in cleaned_words:
        inputs = tokenizer(word, return_tensors='pt').to(device)  # 将输入移动到GPU
        with torch.no_grad():  # 不计算梯度
            outputs = bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        word_embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()  # 转回CPU以进行后续处理
        word_embeddings.append(word_embedding)

    word_embeddings = np.array(word_embeddings)

    # 合并所有特征
    def generate_emotion_vector(emotion_onehot, intensity, pos_onehot, polarity_onehot, main_emotion_onehot,
                                word_embedding):
        return np.concatenate(
            (emotion_onehot, intensity, pos_onehot, polarity_onehot, main_emotion_onehot, word_embedding), axis=None)

    # 创建情感向量字典
    emotion_vectors = {}
    for word, emotion_oh, intensity, pos_oh, polarity_oh, main_emotion_oh, word_emb in zip(words, emotion_onehot,
                                                                                           intensities_normalized,
                                                                                           pos_onehot, polarity_onehot,
                                                                                           main_emotion_onehot,
                                                                                           word_embeddings):
        emotion_vector = generate_emotion_vector(emotion_oh, intensity, pos_oh, polarity_oh, main_emotion_oh, word_emb)
        emotion_vectors[word] = emotion_vector

    # 转换情感向量到PCA
    emotion_vectors_array = np.array(list(emotion_vectors.values()))

    # 进行特征选择
    selected_features = emotion_vectors_array[:, :300]  # 选择前300个特征（根据数据调整）
    pca = PCA(n_components=20)
    emotion_vectors_reduced = pca.fit_transform(selected_features)

    # 更新情感向量字典
    emotion_vectors_reduced_dict = {word: vec for word, vec in zip(emotion_vectors.keys(), emotion_vectors_reduced)}

    return emotion_vectors_reduced_dict
def loadData_BGEwei(train_sentences, test_sentences,train_labels,test_labels, max_sequence_length):
    # 初始化BERT tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained("/public/home/xxxy4/PWHSB/BGE")
    model = AutoModel.from_pretrained("/public/home/xxxy4/PWHSB/BGE")

    # 对训练集和测试集的文本进行编码
    train_encodings = tokenizer(train_sentences, truncation=True, padding=True, return_tensors='pt', max_length=max_sequence_length)
    test_encodings = tokenizer(test_sentences, truncation=True, padding=True, return_tensors='pt', max_length=max_sequence_length)

    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    # 创建TensorDataset
    train_dataset = TensorDataset(
        train_encodings['input_ids'], 
        train_encodings['attention_mask'], 
        train_labels  # 直接传入train_labels张量
    )
    test_dataset = TensorDataset(
        test_encodings['input_ids'], 
        test_encodings['attention_mask'], 
        test_labels  # 直接传入test_labels张量
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练模型
    model.train()
    for epoch in range(10):  # 选择合适的epoch数量
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} finished")

    # 保存微调后的模型
    model.save_pretrained("/public/home/xxxy4/PWHSB/fine_tuned_model")
    model.eval()
    total_eval_accuracy = 0
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        total_eval_accuracy += (predictions == labels).cpu().numpy().mean()

    print(f"Test Accuracy: {total_eval_accuracy / len(test_loader):.4f}")

    with torch.no_grad():
        train_features = model(**train_encodings)[0]
        test_features = model(**test_encodings)[0]

    # 将特征移动回 CPU 并返回特征表示和词汇表
    train_features = train_features.cpu()
    test_features = test_features.cpu()

    return train_features, test_features, tokenizer.get_vocab()