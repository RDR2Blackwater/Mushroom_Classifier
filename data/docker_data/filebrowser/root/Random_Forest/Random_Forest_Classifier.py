# 该文件为随机森林的实现类
import random
from time import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


class RandomForest:
    """
        随机森林对象初始化变量如下
        n_estimators: 决定使用生成树的数量
        max_depth: 每棵树的最大深度不大于max_depth
        max_features: 每棵树最多选用选用数据集中max_features个特征，默认值为sqrt，即特征总数的平方根
        features: 每一列特征的具体名称，用于后续打印各特征的对分类结果的贡献值
        estimators: 一个列表对象，用于装载随机森林中的CART树
        feature_importance_: 随机森林的特征重要性指标，为所有CART树上特征重要性的平均值，
                             该变量将被初始化为一个矩阵
    """

    # RandomForest的构造函数，对象初始化赋值
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.features = features
        self.estimators = []
        self.feature_importance_ = None

    # 训练，每棵树使用随机的数据集和随机的特征
    # X为传入训练集的样本，y为传入训练集的标签
    # -> None 表示该方法无返回值
    def fit(self, X, y) -> None:
        # 获取训练集样本数为n_samples
        n_samples = X.shape[0]
        # 初始化self.feature_importance_为一个单行，列数为特征数的0矩阵
        self.feature_importance_ = np.zeros(X.shape[1])
        # 开始生成CART树
        for i in range(self.n_estimators):
            # random.seed()保证每次运行程序生成随机森林为真随机
            random.seed(time()+i)
            # bootstrap法抽样，每次从n_samples个样本中抽取sqrt(样本数)个子样本，replace=True表示有放回取样
            # indice为list对象，内含CART树所需子样本的索引
            indices = np.random.choice(n_samples, int(np.sqrt(n_samples)), replace=True)
            # 将对应的样本和标签提取到X_bootstrap和y_bootstrap上
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            # 定义CART树，最大深度为self.max_depth，使用特征上限为self.max_features(默认为sqrt(n_features_in_))
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
            # CART树进行训练，这里调用的是CART树的实现类DecisionTreeClassifier的fit方法，而非RandomForest的fit方法
            tree.fit(X_bootstrap, y_bootstrap)
            # 训练完成后，将该树加入到随机森林中
            self.estimators.append(tree)
            # 获取当前CART树的特征重要性指标，并加和至self.feature_importance_
            self.feature_importance_ += tree.feature_importances_
        # self.feature_importance_/self.n_estimators，获得随机森林的特征重要性指标
        self.feature_importance_ /= self.n_estimators

    # 随机森林预测输出，该方法被封装在score方法，用于观察随机森林模型测试效果
    # X为传入测试集的样本
    def predict(self, X) -> np.array:
        # 初始化predictions为一个0矩阵，该矩阵行数与测试集样本数相同，列数为随机森林的CART树数量
        predictions = np.zeros((X.shape[0], len(self.estimators)))
        # enumerate方法接收一个可迭代对象，并返回一个枚举列表[(index1, value1), (index2, value2), ...]
        # 这里的i被赋值为index，即对应CART树在随机森林的索引号，而tree被赋值为随机森林的CART树对象
        for i, tree in enumerate(self.estimators):
            # 遍历每棵树，将每棵树对测试集的预测结果赋值到predictions的对应列上
            predictions[:, i] = tree.predict(X)

        # predictions[i, :].astype(int) 将预测结果转化为整数，便于后续处理

        # np.bincount 统计传入的ndarray对象中各数字的出现次数，并赋值到对应的索引位置上
        # 如np.bincount([1 0 0 1 1]) ==> [2 3]
        # np.argmax 返回第一个出现的最大数字的索引，并返回该索引
        # 如np.argmax([2 3]) ==> 1   np.bincount和np.argmax的结合使用将提取出预测结果中的"众数"

        # [np.arg... for i in range(predictions.shape[0])]表示一个list推导式
        # 该推导式以predictions的行向量(每一棵树对当前样本的判决结果)为参数，将参数传入到前项的np相关方法中进行运算
        # np.array将list对象转换为array对象并返回
        return np.array([np.argmax(np.bincount(predictions[i, :].astype(int))) for i in range(predictions.shape[0])])

    # 返回随机森林对于传入样本集(X, y)的得分
    def score(self, X, y):
        # 调用RandomForest的perdict方法，得到y_pred预测值
        y_pred = self.predict(X)
        # 得分=(y_pred==y（预测正确数）)/合计预测数
        return np.mean(y_pred == y)

    # 打印随机森林各特征的影响力
    def plot_feature_importance(self, max_features=20):
        # 如果随机森林构造时没有传入特征名，则将feature_names初始化为Feature_1，Feature_2，...
        # 否则使用传入特征名为feature_names
        if self.features is None:
            feature_indices = range(len(self.feature_importance_))
            feature_names = ['Feature ' + str(i + 1) for i in feature_indices]
        else:
            feature_names = self.features

        # 计算每个属性对分类结果的影响力得分，feature_importance为字典对象
        feature_importance = {}
        # 以enumerate(feature_names)返回的特征名feature为key，i为特征影响力的索引
        # 将对应位置特征的影响力self.feature_importance_[i]赋值到对应的feature_importance位置上
        for i, feature in enumerate(feature_names):
            feature_importance[feature] = self.feature_importance_[i]

        # 按影响力得分从高到低排序
        # {k: v for k, v in ...}为字典推导式
        # feature_importance.items() 表示feature_importance字典的各个 key:value 对
        # sorted()为排序函数，接收三个参数:
        #   可迭代对象feature_importance.items()，为需要排序的对象
        #   排序比较元素key=lambda item: item[1] lambda表示匿名函数，
        #                              表示以feature_importance.items()的第二个元素(score)进行排序
        #   reverse=True 表示降序排序
        sorted_feature_importance = {k: v for k, v in
                                     sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}

        # 以文字方式打印特征重要性排序
        print(f"Top {max_features} features importance ranking:")
        rank = 1
        for feature, score in sorted_feature_importance.items():
            # 排序大于max_features时，停止打印
            if rank > max_features:
                break
            print(f"{rank}. {feature} ({score:.4f})")
            rank += 1

        # 绘制特征重要性条形图
        plt.figure()
        plt.title("Feature Importance")
        # 将sorted_feature_importance.items()转换为list，截取前max_features个 key:value 对
        # 最后再转换为字典对象
        top_influence_features = dict(list(sorted_feature_importance.items())[:max_features])
        # 以柱状图打印top_influence_features的 key:value 对
        plt.bar(top_influence_features.keys(), top_influence_features.values())
        # 控制标签的旋转角度
        plt.xticks(rotation=30)
        plt.ylabel("Importance Score")
        plt.show()

    # RandomForest的析构函数，回收内存空间
    def __del__(self):
        pass


# 随机森林分类准确性与使用CART树数量的关系
def plot_accuracy(max_estimators, X_train, y_train, X_test, y_test, Columns):
    # 得分list，值为对应随机森林的得分，索引号 + 1 = 对应随机森林的CART树数量
    accuracy_scores = []
    for i in range(1, max_estimators + 1):
        # 使用CART树数量为 1 ~ max_estimators，树深度和特征不变
        clf_test = RandomForest(n_estimators=i, max_depth=9, features=Columns)
        clf_test.fit(X_train, y_train)
        # 将当前随机森林的得分接入accuracy_scores
        accuracy_scores.append(clf_test.score(X_test, y_test))
        # 立即删除对象，释放内存空间
        del clf_test
    # 构造图表，x轴为 1 ~ max_estimators，y轴为accuracy_scores
    plt.plot(range(1, max_estimators + 1), accuracy_scores)
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    # 打印图表
    plt.show()