import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class GlycopeptideFeatureSelector:
    """
    糖肽特征选择类，用于找出区分病人和健康人能力最好的多肽或多肽组合
    """
    
    def __init__(self):
        self.selected_features = None
        
    def select_best_single_features(self, X, y, feature_names, method='f_classif', top_k=10):
        """
        选择最佳的单个特征（单个多肽）
        
        参数:
        X: numpy.ndarray, 特征数据
        y: numpy.ndarray, 标签数据
        feature_names: list, 特征名称列表
        method: str, 特征选择方法，可选 'f_classif', 'mutual_info_classif'
        top_k: int, 返回前k个最佳特征
        
        返回:
        pandas.DataFrame, 包含特征名称和得分的排序结果
        """
        # 选择特征选择方法
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=top_k)
        elif method == 'mutual_info_classif':
            selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
        else:
            raise ValueError("不支持的特征选择方法，可选方法为 'f_classif' 或 'mutual_info_classif'")
        
        # 拟合选择器
        selector.fit(X, y)
        
        # 获取得分和特征名称
        scores = selector.scores_
        
        # 创建结果数据框
        results = pd.DataFrame({
            'feature': feature_names,
            'score': scores
        })
        
        # 按得分排序
        results = results.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        return results.head(top_k)
    
    def recursive_feature_elimination(self, X, y, feature_names, model_type='rf', n_features_to_select=5):
        """
        使用递归特征消除选择最佳特征组合
        
        参数:
        X: numpy.ndarray, 特征数据
        y: numpy.ndarray, 标签数据
        feature_names: list, 特征名称列表
        model_type: str, 基础模型类型，可选 'rf' (随机森林), 'lr' (逻辑回归), 'svm' (支持向量机)
        n_features_to_select: int, 要选择的特征数量
        
        返回:
        list, 选择的特征名称列表
        numpy.ndarray, 选择的特征数据
        """
        # 选择基础模型
        if model_type == 'rf':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            estimator = LogisticRegression(random_state=42)
        elif model_type == 'svm':
            estimator = SVC(kernel='linear', random_state=42)
        else:
            raise ValueError("不支持的模型类型，可选模型为 'rf', 'lr' 或 'svm'")
        
        # 创建RFE选择器
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        
        # 拟合选择器
        rfe.fit(X, y)
        
        # 获取选择的特征
        selected_feature_indices = np.where(rfe.support_)[0]
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        
        # 选择特征数据
        X_selected = X[:, selected_feature_indices]
        
        self.selected_features = selected_feature_names
        
        return selected_feature_names, X_selected
    
    def select_best_combination(self, X, y, feature_names, model_type='rf', max_features=5):
        """
        选择最佳的特征组合
        
        参数:
        X: numpy.ndarray, 特征数据
        y: numpy.ndarray, 标签数据
        feature_names: list, 特征名称列表
        model_type: str, 用于评估组合的模型类型
        max_features: int, 考虑的最大特征数量
        
        返回:
        dict, 包含不同特征数量下的最佳组合和性能
        """
        from sklearn.model_selection import cross_val_score
        
        # 选择基础模型
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42)
        elif model_type == 'svm':
            model = SVC(kernel='linear', random_state=42)
        else:
            raise ValueError("不支持的模型类型，可选模型为 'rf', 'lr' 或 'svm'")
        
        best_combinations = {}
        
        # 从单个特征到max_features个特征
        for k in range(1, max_features + 1):
            # 使用RFE选择k个特征
            _, X_selected = self.recursive_feature_elimination(X, y, feature_names, model_type, k)
            
            # 使用交叉验证评估性能
            scores = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc_ovr', n_jobs=1)
            mean_score = scores.mean()
            
            best_combinations[k] = {
                'features': self.selected_features,
                'mean_roc_auc': mean_score,
                'cv_scores': scores
            }
        
        return best_combinations
    
    def get_feature_importance(self, X, y, feature_names, model_type='rf'):
        """
        获取特征重要性
        
        参数:
        X: numpy.ndarray, 特征数据
        y: numpy.ndarray, 标签数据
        feature_names: list, 特征名称列表
        model_type: str, 模型类型，可选 'rf' (随机森林) 或 'lr' (逻辑回归)
        
        返回:
        pandas.DataFrame, 包含特征名称和重要性的排序结果
        """
        # 选择模型
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42)
        else:
            raise ValueError("不支持的模型类型，可选模型为 'rf' 或 'lr'")
        
        # 拟合模型
        model.fit(X, y)
        
        # 获取特征重要性
        if model_type == 'rf':
            importances = model.feature_importances_
        elif model_type == 'lr':
            importances = np.abs(model.coef_[0])
        
        # 创建结果数据框
        results = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # 按重要性排序
        results = results.sort_values(by='importance', ascending=False).reset_index(drop=True)
        
        return results
