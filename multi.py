import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import text_feature_extraction as txt
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DilatedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(out_channels)  # Adding batch normalization
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):  # Adding dropout to LSTM
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, input_dim)  # 输入和输出维度都为 input_dim
        self.v = nn.Parameter(torch.rand(input_dim))  # 将权重向量的维度也设置为 input_dim

    def forward(self, x):
        scores = torch.tanh(self.attn(x))  # x: (batch_size, seq_len, input_dim)
        attn_weights = torch.matmul(scores, self.v)  # (batch_size, seq_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=1)  # 对每个序列元素进行softmax
        weighted_sum = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)
        return weighted_sum

def position(seq_len, is_cuda=True):
    if is_cuda:
        loc = torch.linspace(-1.0, 1.0, seq_len).cuda()
    else:
        loc = torch.linspace(-1.0, 1.0, seq_len)
    loc = loc.unsqueeze(0)  # Shape: (1, seq_len)
    return loc

class MultiChannelDilatedCNN_BiLSTM_Attention(nn.Module):
    def __init__(self,embedding_dim, cnn_out_channels, cnn_kernel_size, lstm_hidden_dim, lstm_num_layers, num_classes, dropout=0.5):
        super(MultiChannelDilatedCNN_BiLSTM_Attention, self).__init__()
        self.in_planes = embedding_dim
        self.out_planes = cnn_out_channels
        self.head = 8
        self.kernel_att = 7
        self.kernel_conv = cnn_kernel_size
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head
        self.stride = 1
        self.dilation = 1 

        self.dilated_cnn1 = DilatedCNN(embedding_dim, cnn_out_channels, cnn_kernel_size, dilation=1)
        self.dilated_cnn2 = DilatedCNN(embedding_dim, cnn_out_channels, cnn_kernel_size, dilation=2)
        self.dilated_cnn3 = DilatedCNN(embedding_dim, cnn_out_channels, cnn_kernel_size, dilation=3)
        self.conv_p = nn.Conv1d(1, self.head_dim, kernel_size=1)
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.ConstantPad1d(self.padding_att, 0)  # Changed to ConstantPad1d for 1D
        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Conv1d(3*self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv1d(self.kernel_conv * self.kernel_conv * self.head_dim, self.out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=self.stride)
        #440
        self.att_to_conv = nn.Linear(880, 128)

        self.reset_parameters()


        self.lstm_cnn = BiLSTM(cnn_out_channels, lstm_hidden_dim, lstm_num_layers)
        self.lstm_direct = BiLSTM(embedding_dim, lstm_hidden_dim, lstm_num_layers)
        self.local_attention = Attention(lstm_hidden_dim * 2)
        self.global_attention = Attention(lstm_hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_dim * 4, lstm_hidden_dim*2 )
        self.fc2 = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim)
        self.fc3 = nn.Linear(lstm_hidden_dim, num_classes)
    def reset_parameters(self):
        self.rate1.data.fill_(0.5)
        self.rate2.data.fill_(0.5)

        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i % self.kernel_conv] = 1.
        kernel = kernel.unsqueeze(0).repeat(self.out_planes, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias.data.fill_(0.)
    def forward(self, x):
        x = x.transpose(1, 2)
        #4,1024,110
        q = self.dilated_cnn1(x)
        k = self.dilated_cnn2(x)
        v = self.dilated_cnn3(x)
        #[4, 128, 110]
        scaling = float(self.head_dim) ** -0.5
        b, c, seq_len = q.shape
        seq_len_out = seq_len // self.stride

        pe = self.conv_p(position(seq_len, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, seq_len) * scaling
        k_att = k.view(b * self.head, self.head_dim, seq_len)
        v_att = v.view(b * self.head, self.head_dim, seq_len)

        q_pe = pe
        unfold_k = k_att.unsqueeze(1)  # Adjusted for 1D
        unfold_pe = pe.unsqueeze(0).unsqueeze(2)  # Adjusted for 1D

        att = (q_att.unsqueeze(2) * (unfold_k + unfold_pe)).sum(1)  # (b*head, head_dim, seq_len)
        att = self.softmax(att)
        # Ensure matrix multiplication dimensions are correct
        att = att.permute(0, 2, 1)  # (b*head, seq_len, head_dim)
        out_att = torch.bmm(att, v_att)  # (b*head, seq_len, head_dim) @ (b*head, head_dim, seq_len) -> (b*head, head_dim, seq_len)
        out_att = out_att.view(b, -1, seq_len)  # Reshape to (b, head_dim, seq_len)
        out_att = out_att.transpose(1, 2)
        out_att = self.att_to_conv(out_att)
        out_att = out_att.transpose(1, 2)
        # Perform matrix multiplication between attention and value
        # out_att = torch.bmm(att, v_att.transpose(1, 2)).transpose(1, 2)  # Shape: (b*head, seq_len, head_dim)
        # Convolution
        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim*seq_len), 
                                   k.view(b, self.head, self.head_dim*seq_len), 
                                   v.view(b, self.head, self.head_dim*seq_len)], 1))
    
        num_channels = self.kernel_conv * self.kernel_conv * self.head_dim
        f_conv = f_all.permute(0, 2, 1).reshape(b, num_channels, seq_len)  # Ensure shape is (b, num_channels, seq_len)

        out_conv = self.dep_conv(f_conv)
        x_cnn = self.rate1 * out_att + self.rate2 * out_conv
      #  x_cnn = self.rate1 * out_att + self.rate2 * out_conv
       # x_cnn = out_att

        x_cnn = x_cnn.transpose(1, 2)
        lstm_cnn_out = self.lstm_cnn(x_cnn)
        local_cnn_attn_out = self.local_attention(lstm_cnn_out)
        lstm_direct_out = self.lstm_direct(x.transpose(1, 2))
        local_direct_attn_out = self.local_attention(lstm_direct_out)

        combined_local_attn_out = torch.cat((local_cnn_attn_out, local_direct_attn_out), dim=1)
        global_attn_out = self.global_attention(combined_local_attn_out.unsqueeze(1))

        output = self.dropout(global_attn_out)
        output = torch.relu(self.fc1(output))
        output = torch.relu(self.fc2(output))
        output = self.fc3(output)
        return output

# Load and preprocess data
sentences = []
labels = []
label_list = ['难过', '愉快', '喜欢', '愤怒', '害怕', '惊讶', '厌恶']
#The_Truman_ShowQ.csv
#Three_Kingdoms45Q.csv
#TrumanQ.csv
#ThreeQ.csv
file_path = 'ThreeQ.csv'
with open('ThreeQ.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    flag = 1  # 用于跳过第一行（假设第一行为表头）
    line_number = 0  # 用于记录当前行号

    for row in reader:
        line_number += 1  # 行号递增

        # 跳过表头
        if flag == 1:
            flag = 0
            continue

        try:
            # 解析当前行
            new_list = [item.split(',') for item in row]
            sentences.append(new_list[0][0])  # 添加句子
            labels.append(label_list.index(new_list[1][0]))  # 添加标签索引

        except Exception as e:
            # 捕获异常并打印行号和错误信息
            print(f"Error on line {line_number}: {row}")
            print(f"Error message: {e}")
# 划分训练集和测试集
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.25)
print("abc")
train_data, test_data, word_index = txt.loadData_BGE(train_sentences, test_sentences, 110)
#train_data, test_data, word_index = txt.loadData_BGEwei(train_sentences, test_sentences,train_labels, test_labels, 110)
print("cde")
embedding_dim = 1024
cnn_out_channels = 128
cnn_kernel_size = 3
lstm_hidden_dim = 256
lstm_num_layers = 2
num_classes = 7
dropout = 0.5
learning_rate = 0.00004  # Adjust learning rate
batch_size = 4
num_epochs = 50

train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
train_tensor = train_data_tensor.to(device)
train_labels_tensor = train_labels_tensor.to(device)
test_tensor = test_data_tensor.to(device)
test_labels_tensor = test_labels_tensor.to(device)
# 创建数据集和数据加载器
train_dataset = CustomDataset(train_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = MultiChannelDilatedCNN_BiLSTM_Attention(embedding_dim, cnn_out_channels, cnn_kernel_size, lstm_hidden_dim, lstm_num_layers, num_classes, dropout)
# model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
scaler = GradScaler()

best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        with autocast():
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    scheduler.step()
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_outputs = model(test_tensor)
        val_loss = criterion(val_outputs, test_labels_tensor).item()
    
    with open("train_logmultiT.txt", 'a') as f:
        f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}\n')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
# Load the best model for testing
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    # 模型预测
    test_outputs = model(test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    correct = (predicted == test_labels_tensor).sum().item()
    accuracy = correct / len(test_labels_tensor)
    
    # 转换为 NumPy 数组
    predicted = predicted.cpu().numpy()
    test_labels = test_labels_tensor.cpu().numpy()
    
    # 计算 F1 Score
    f1 = f1_score(test_labels, predicted, average='weighted')
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(test_labels, predicted)
    
    # 保存结果到文件
    with open("test_logmultiT.txt", 'w') as f:
        f.write(f'Test Accuracy: {accuracy:.4f}\n')
        f.write(f'Test F1 Score: {f1:.4f}\n')
        f.write('Confusion Matrix:\n')
        for row in conf_matrix:
            f.write(' '.join(map(str, row)) + '\n')
