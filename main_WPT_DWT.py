import numpy as np
import os
import pywt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf

# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置随机种子，保证可重复性
tf.random.set_seed(2)
np.random.seed(2)
random.seed(2)

# 加载pkl文件的函数
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def wpt_dwt_feature_extraction(signal, wpt_wavelet='db12', dwt_wavelet='db12',
                               wpt_level=2, dwt_level=3, feature_dim=16):
    """
    WPT-DWT混合特征提取 (先WPT后DWT)
    参数：
        signal: 输入信号
        wpt_wavelet: WPT使用的小波基
        dwt_wavelet: DWT使用的小波基
        wpt_level: WPT分解层数
        dwt_level: DWT分解层数
        feature_dim: 最终特征维度
    返回：
        特征向量 (feature_dim,)
    """
    features = []

    # 1. 先进行WPT全子带分解
    wp = pywt.WaveletPacket(data=signal, wavelet=wpt_wavelet, mode='symmetric', maxlevel=wpt_level)
    wpt_nodes = wp.get_level(wpt_level, 'natural')

    # 2. 对每个WPT子带进行DWT分解
    for node in wpt_nodes:
        node_data = wp[node.path].data

        # 对WPT子带进行DWT分解
        coeffs = pywt.wavedec(node_data, dwt_wavelet, level=dwt_level)

        # 提取DWT特征
        # 处理近似系数
        approx = coeffs[0]
        features.append(np.mean(approx))  # 均值
        features.append(np.std(approx))  # 标准差

        # 处理细节系数
        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            features.append(np.sum(detail ** 2))  # 能量
            features.append(-np.sum(detail ** 2 * np.log(detail ** 2 + 1e-8)))  # 熵

    # 3. 特征选择与降维
    # 取前feature_dim个特征（可根据需要调整）
    selected_features = features[:feature_dim]
    # 如果特征不足则补零
    if len(selected_features) < feature_dim:
        selected_features += [0] * (feature_dim - len(selected_features))
    return np.array(selected_features)

def dwt_wpt_feature_extraction(signal, dwt_wavelet='db12', wpt_wavelet='db12',
                               dwt_level=3, wpt_level=2, feature_dim=16):
    """
    DWT-WPT混合特征提取
    参数：
        signal: 输入信号
        dwt_wavelet: DWT使用的小波基
        wpt_wavelet: WPT使用的小波基
        dwt_level: DWT分解层数
        wpt_level: WPT分解层数
        feature_dim: 最终特征维度
    返回：
        特征向量 (feature_dim,)
    """
    # 1. DWT粗分解
    coeffs = pywt.wavedec(signal, dwt_wavelet, level=dwt_level)

    features = []

    # 2. 处理近似系数（低频部分）
    approx = coeffs[0]
    features.append(np.mean(approx))  # 均值
    features.append(np.std(approx))  # 标准差
    features.append(np.sum(approx ** 2))  # 能量

    # 3. 处理细节系数（高频部分）
    for i in range(1, len(coeffs)):
        detail = coeffs[i]

        # 进行WPT细分解
        wp = pywt.WaveletPacket(data=detail, wavelet=wpt_wavelet, mode='symmetric', maxlevel=wpt_level)

        # 提取WPT子带特征
        nodes = wp.get_level(wpt_level, 'natural')
        for node in nodes:
            node_data = wp[node.path].data
            # 计算子带能量和熵
            energy = np.sum(node_data ** 2)
            entropy = -np.sum(node_data ** 2 * np.log(node_data ** 2 + 1e-8))
            features.extend([energy, entropy])

    # 4. 特征选择与降维
    # 取前feature_dim个特征（可根据需要调整）
    selected_features = features[:feature_dim]

    # 如果特征不足则补零
    if len(selected_features) < feature_dim:
        selected_features += [0] * (feature_dim - len(selected_features))

    return np.array(selected_features)


# 绘制并保存混淆矩阵的函数
def save_confusion_matrix(cm, class_names, file_path, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 20})  # 设置混淆矩阵中数字的字体大小
    plt.title(title, fontsize=20)
    plt.xlabel('Predicted labels', fontsize=20)
    plt.ylabel('True labels', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # 设置 colorbar 的字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(file_path)  # 保存图片
    plt.show()  # 展示当前图形

# 设置保存小波包图像的目录
save_dir = './fig/wavelet_packet_plots'  # 你可以修改为你想要的保存路径

# 加载数据
file_path = f'./data1/data.pkl'  # 修改为你的pkl文件路径
data = load_pkl(file_path)

# 3. 标准化/归一化处理
scaler = StandardScaler()  # 选择标准化方法
data = scaler.fit_transform(data)  # 归一化或者标准化
print(data.max(), data.min(), data.mean(), data.std())



# 特征提取流程
features = []
for i in range(data.shape[0]):
    # 对每个信号提取特征
    feature_vector = wpt_dwt_feature_extraction(
        data[i,:],
        dwt_wavelet='db4',
        wpt_wavelet='db12',
        dwt_level=3,
        wpt_level=4,
        feature_dim=32
    )
    features.append(feature_vector)

features = np.array(features).reshape(data.shape[0], 32)

plt.figure(figsize=(10,6))
plt.plot(features[0], 'b-', label='Sample 1')
plt.plot(features[100], 'r--', label='Sample 2')
plt.title("DWT-WPT Feature Visualization")
plt.xlabel("Feature Dimension")
plt.ylabel("Feature Value")
plt.legend()
plt.show()

# for idx in range(data.shape[0]):
#     signal = data[idx, :]
#     # 对每个信号进行小波包分解并提取特征
#     feature_vector = wavelet_packet_decomposition(signal, wavelets=[wavelet], max_level=max_level)
#     features.append(feature_vector)
#     # 绘制并保存小波包变换图
#     plot_and_save_wavelet_packet(signal, wavelet, max_level, idx, save_dir)
#     print(f"Processed signal {idx+1}/{data.shape[0]} and saved plot.")

# features = np.array(features).reshape(data.shape[0], -1)

# 划分类别
C0 = features[0:500, :]
C1 = features[500:1000, :]
C2 = features[1000:1500, :]
C3 = features[1500:2000, :]
C4 = features[2000:2500, :]
C5 = features[2500:3000, :]
C6 = features[3000:3500, :]
C7 = features[3500:4000, :]
C8 = features[4000:4500, :]

#标签
y=[]
for i in range(9):
    y.append(np.array([i]*500))
y=np.array(y).reshape(4500,1)

X0 = C0
X1 = C1
X2 = C2
X3 = C3
X4 = C4
X5 = C5
X6 = C6
X7 = C7
X8 = C8

# Shuffle
mix0 = [i for i in range (len(X0))]
np.random.shuffle(mix0)
x0 = X0[mix0]

mix1 = [i for i in range (len(X1))]
np.random.shuffle(mix1)
x1 = X1[mix1]

mix2 = [i for i in range (len(X2))]
np.random.shuffle(mix2)
x2 = X2[mix2]

mix3 = [i for i in range (len(X3))]
np.random.shuffle(mix3)
x3 = X3[mix3]

mix4 = [i for i in range (len(X4))]
np.random.shuffle(mix4)
x4 = X4[mix4]

mix5 = [i for i in range (len(X5))]
np.random.shuffle(mix5)
x5 = X5[mix5]

mix6 = [i for i in range (len(X6))]
np.random.shuffle(mix6)
x6 = X6[mix6]

mix7 = [i for i in range (len(X7))]
np.random.shuffle(mix7)
x7 = X7[mix7]

mix8 = [i for i in range (len(X8))]
np.random.shuffle(mix8)
x8 = X8[mix8]

# 划分测试集与数据集
x0_train,x0_test,y0_train,y0_test=train_test_split(x0,y[0:500,:],test_size=0.2)
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y[500:1000,:],test_size=0.2)
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y[1000:1500,:],test_size=0.2)
x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y[1500:2000,:],test_size=0.2)
x4_train,x4_test,y4_train,y4_test=train_test_split(x4,y[2000:2500,:],test_size=0.2)
# x5_train,x5_test,y5_train,y5_test=train_test_split(x5,y[2500:3000,:],test_size=0.2)
# x6_train,x6_test,y6_train,y6_test=train_test_split(x6,y[3000:3500,:],test_size=0.2)
# x7_train,x7_test,y7_train,y7_test=train_test_split(x7,y[3500:4000,:],test_size=0.2)
# x8_train,x8_test,y8_train,y8_test=train_test_split(x8,y[4000:4500,:],test_size=0.2)

train_x=np.vstack([x0_train, x1_train[0:50,:], x2_train[0:30,:], x3_train[0:20,:], x4_train[0:10,:]])
train_y=np.vstack([y0_train, y1_train[0:50,:], y2_train[0:30,:], y3_train[0:20,:], y4_train[0:10,:]])
test_x=np.vstack([x0_test, x1_test, x2_test, x3_test, x4_test])
test_y=np.vstack([y0_test, y1_test, y2_test, y3_test, y4_test])

print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)

mix = [i for i in range(len(train_x))]
np.random.shuffle(mix)
X_train = train_x[mix]
y_train = train_y[mix]

# 使用随机森林进行分类
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2, class_weight="balanced")
rf_classifier.fit(X_train, y_train)

result = rf_classifier.score(test_x, test_y)
labels_pred = rf_classifier.predict(test_x)
accuracy = accuracy_score(test_y,labels_pred)
print(f'Random Forest Classifier Accuracy: {accuracy:.4f}')

import pandas as pd
# 获取分类报告并保留4位小数
report = classification_report(test_y, labels_pred, zero_division=0, output_dict=True)
formatted_report = pd.DataFrame(report).transpose()
formatted_report = formatted_report.applymap(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

# 打印格式化后的分类报告
print("Classification Report (4 decimal places):")
print(formatted_report)

# 计算混淆矩阵
cm = confusion_matrix(test_y, labels_pred)
print("Confusion Matrix:")
print(cm)
classes = ['C0', 'C1', 'C2', 'C3', 'C4']
# 保存混淆矩阵
save_confusion_matrix(cm, classes, './fig/WPT_DWT.png')




from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, perplexity=30)
# tsne_result = tsne.fit_transform(test_x)
# 进行 PCA 降维
pca = PCA(n_components=2)  # 将数据降至2维，以便可视化
pca_result = pca.fit_transform(test_x)  # 使用测试集数据进行PCA降维

# 使用 KMeans 聚类进行分析
kmeans = KMeans(n_clusters=5, random_state=2)  # 假设7个类别，适应于我们的数据集
kmeans_labels = kmeans.fit_predict(pca_result)

# 可视化 PCA 结果与聚类结果
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=kmeans_labels, palette='tab10', s=150, alpha=0.7)

# 绘制类别中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

# plt.title('PCA + KMeans Clustering', fontsize=24)
plt.xlabel('Principal Component 1', fontsize=16)
plt.ylabel('Principal Component 2', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=16)
# 保存图像
plt.savefig('./fig/WPT_DWT_cluster.png')
plt.show()

# 打印聚类结果
print("KMeans Clustering Labels:")
print(kmeans_labels)

# 输出聚类中心
print("Cluster Centers:")
print(centers)

# 如果想要评估聚类结果，可以使用以下方法
from sklearn.metrics import silhouette_score, calinski_harabasz_score
# 评估聚类效果
def evaluate_clustering(data, clusters):
    sil_score = silhouette_score(data, clusters)
    calinski_score = calinski_harabasz_score(data, clusters)
    print(f'Silhouette Score: {sil_score:.4f}')
    print(f'Calinski-Harabasz Score: {calinski_score:.4f}')

# 评估聚类效果
evaluate_clustering(pca_result, kmeans_labels)
