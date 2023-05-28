# 该文件为具体的毒蘑菇分类器实现
import pandas
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from Random_Forest_Classifier import RandomForest, plot_accuracy
import matplotlib.pyplot as plt


# 数据集预处理函数
def set_normalize(input_frame: pandas.DataFrame):
    # input_frame.info()反馈结果可知，在本样本集中是没有缺失值的（均为7499 non-null）
    input_frame.info()
    # input_frame.dropna()删除数据集的空行
    input_frame = input_frame.dropna(axis=0, how='all')
    # df['class'].value_counts(normalize=True)用于查看数据集中有毒和可使用样本的比例，
    print('\n', input_frame['class'].value_counts(normalize=True))

    # 将类别列单独抽取出来作为标签列，并使用LabelEncoder()对象将其标准化（0、1值）
    frame_label = input_frame['class']
    encoder = LabelEncoder()
    frame_label = encoder.fit_transform(frame_label)
    # 利用pandas.drop()方法，将数据集的数据与标签分离
    input_frame = input_frame.drop(['class'], axis=1)
    # 使用pandas的get_dummies方法，对样本进行one hot encode，实现样本属性的离散化
    # 如cap-shape特征值有b、c、x... ，在one hot encode后将变为独立的列cap-shape_b=True/False、cap-shape_c=True/False，实现属性离散化
    # input_frame占用内存将减少，因为bool对象（True、False）大小明显小于str对象（'p'、'e'）
    # 该做法将不会处理stalk-root的missing值，而是将missing本身也作为一个特征分支去处理，且每行数据的stalk-root值有且只有一个为1
    input_frame = pd.get_dummies(input_frame)
    return input_frame, frame_label


# ROC绘制函数
def roc_printer(x_scaled, y) -> None:
    # 获取测试集的分类结果
    y_predict = clf.predict(x_scaled)
    # 根据测试集标签和分类结果，获得ROC曲线的fpr和tpr
    fpr, tpr, _ = roc_curve(y, y_predict)
    # 通过auc方法获取AUC
    roc_auc = auc(fpr, tpr)
    # 绘图
    plt.plot(fpr, tpr, 'k', label='AUC = {0:.4f}'.format(roc_auc))
    plt.legend()
    plt.show()


# 运行开始点
if __name__ == '__main__':
    # 1 加载、检查数据集，对数据集进行预处理
    df = pd.read_csv('mushrooms.csv')
    # 调用数据预处理函数
    X, Y = set_normalize(df)
    # 分离训练集和测试集，此处测试集占总数据集的30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # StandardScaler()对象将会对样本的各特征进行归一化处理
    scale = StandardScaler()
    # fit_transform(X_train)沿着数据列方向，减去该列的均值，除以标准差以实现标准化
    X_train_scaled = scale.fit_transform(X_train)
    # 根据对之前X_train进行fit的整体指标，对剩余的数据（X_test）使用同样的均值和标准差
    # 等指标进行转换transform(X_test)，从而保证X_train、X_test处理方式相同
    # （保证随机森林对X_train、X_test的分类是无偏的）
    X_test_scaled = scale.transform(X_test)

    # 2 建立随机森林模型，使用训练集进行训练
    # n_estimators=100将在随机森林建立100课CART树
    # max_depth=9规定CART树最大深度为9
    # features=X.columns将数据集的特征名称传入，用于后续的plot_feature_importance()方法中
    clf = RandomForest(n_estimators=100, max_depth=9, features=X.columns)
    # 传入训练集，进行训练
    clf.fit(X_train_scaled, Y_train)

    # 3 利用测试集，验证分类器效果
    # 打印当前随机森林的分类正确度（得分）
    print('\nPrecision(Using 100 CART classifier trees):', clf.score(X_test_scaled, Y_test))

    # 4 建立模型后对结果的分析
    # 查看随机森林随着使用树增长时分类性能的变化程度
    # plot_accuracy(1000, X_train_scaled, Y_train, X_test_scaled, Y_test, X.columns)

    # 查看模型中各特征的影响力（降序排列，打印前10高影响力的属性）
    clf.plot_feature_importance(max_features=10)

    # 调用roc_printer方法绘制ROC曲线
    roc_printer(X_test_scaled, Y_test)

    # 5 课堂测试集输入
    while True:
        # 判断是否需要额外测试集，输入y则进行输入
        choice = input('Require additional test set?(y/N)')
        if choice == 'y':
            # 获取额外测试集并进行处理
            new_test = pd.read_csv(input('Input your test set name:'))
            new_test_x, new_test_y = set_normalize(new_test)
            test_x_scaled = scale.transform(new_test_x)
            # 使用已有的训练树进行分类，打印分类结果
            print('\nPrecision(Using 100 CART classifier trees):', clf.score(test_x_scaled, new_test_y))

            # 调用roc_printer方法绘制ROC曲线
            roc_printer(test_x_scaled, new_test_y)
        else:
            break
