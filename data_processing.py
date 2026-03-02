import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class GlycopeptideDataProcessor:
    """
    糖肽数据处理类，用于加载、预处理和准备用于机器学习的数据
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """
        从CSV文件加载数据
        
        参数:
        file_path: str, 数据文件路径
        
        返回:
        pandas.DataFrame, 加载的数据
        """
        try:
            data = pd.read_csv(file_path)
            print(f"成功加载数据: {file_path}")
            return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
    
    def preprocess_data(self, data, target_column='label', health_control_label=0, igan_patients_label=1, non_igan_patients_control_label=2):
        """
        预处理数据，包括标签转换和特征标准化
        
        参数:
        data: pandas.DataFrame, 原始数据
        target_column: str, 标签列名称
        health_control_label: int, 健康对照的标签值
        igan_patients_label: int, IgAN病人的标签值
        non_igan_patients_control_label: int, 非IgAN病人对照的标签值
        
        返回:
        X: numpy.ndarray, 标准化后的特征数据
        y: numpy.ndarray, 标签数据
        feature_names: list, 特征名称列表
        """
        # 分离特征和标签
        feature_columns = [col for col in data.columns if col != target_column]
        X = data[feature_columns].values
        y = data[target_column].values
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_columns
    
    def prepare_glycopeptide_data(self, glycopeptide_structures, binding_data, labels):
        """
        准备糖肽数据，将糖肽结构、结合力和标签整合为机器学习可用的格式
        
        参数:
        glycopeptide_structures: list, 糖肽结构列表
        binding_data: numpy.ndarray, 结合力数据，形状为 (样本数, 糖肽数)
        labels: numpy.ndarray, 标签数据，形状为 (样本数,)
        
        返回:
        pandas.DataFrame, 整合后的数据
        """
        # 创建数据框
        data = pd.DataFrame(binding_data, columns=glycopeptide_structures)
        data['label'] = labels
        
        return data
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        将数据分为训练集和测试集
        
        参数:
        X: numpy.ndarray, 特征数据
        y: numpy.ndarray, 标签数据
        test_size: float, 测试集比例
        random_state: int, 随机种子
        
        返回:
        X_train, X_test, y_train, y_test: numpy.ndarray, 训练和测试数据
        """
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def cross_validation_split(self, X, y, n_splits=5, random_state=42):
        """
        生成交叉验证的训练和测试索引
        
        参数:
        X: numpy.ndarray, 特征数据
        y: numpy.ndarray, 标签数据
        n_splits: int, 交叉验证的折数
        random_state: int, 随机种子
        
        返回:
        sklearn.model_selection.KFold, 交叉验证生成器
        """
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
