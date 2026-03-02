import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class GlycopeptideModelTrainer:
    """
    糖肽模型训练类，用于训练和评估机器学习模型
    """
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train_model(self, X_train, y_train, model_type='rf', hyperparams=None):
        """
        训练机器学习模型
        
        参数:
        X_train: numpy.ndarray, 训练特征数据
        y_train: numpy.ndarray, 训练标签数据
        model_type: str, 模型类型，可选 'rf' (随机森林), 'lr' (逻辑回归), 'svm' (支持向量机)
        hyperparams: dict, 超参数设置
        
        返回:
        训练好的模型
        """
        # 选择模型
        if model_type == 'rf':
            self.model = RandomForestClassifier(random_state=42, **(hyperparams or {}))
        elif model_type == 'lr':
            self.model = LogisticRegression(random_state=42, **(hyperparams or {}))
        elif model_type == 'svm':
            self.model = SVC(random_state=42, probability=True, **(hyperparams or {}))
        else:
            raise ValueError("不支持的模型类型，可选模型为 'rf', 'lr' 或 'svm'")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='rf', param_grid=None, cv=5):
        """
        超参数调优
        
        参数:
        X_train: numpy.ndarray, 训练特征数据
        y_train: numpy.ndarray, 训练标签数据
        model_type: str, 模型类型
        param_grid: dict, 超参数搜索空间
        cv: int, 交叉验证折数
        
        返回:
        最佳模型
        """
        # 选择模型
        if model_type == 'rf':
            base_model = RandomForestClassifier(random_state=42)
            default_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'lr':
            base_model = LogisticRegression(random_state=42)
            default_param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif model_type == 'svm':
            base_model = SVC(random_state=42, probability=True)
            default_param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        else:
            raise ValueError("不支持的模型类型，可选模型为 'rf', 'lr' 或 'svm'")
        
        # 使用默认参数网格或用户提供的参数网格
        param_grid = param_grid or default_param_grid
        
        # 创建网格搜索
        # 对于多分类问题，使用roc_auc_ovr作为评分指标
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                                   cv=cv, scoring='roc_auc_ovr', n_jobs=1)
        
        # 拟合网格搜索
        grid_search.fit(X_train, y_train)
        
        # 获取最佳模型和参数
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"最佳参数: {self.best_params}")
        print(f"最佳交叉验证ROC-AUC得分: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能
        
        参数:
        X_test: numpy.ndarray, 测试特征数据
        y_test: numpy.ndarray, 测试标签数据
        
        返回:
        dict, 包含各种性能指标
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train_model 或 hyperparameter_tuning 方法")
        
        # 预测概率和类别
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        
        # 计算性能指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        # 创建结果字典
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
        
        return results
    
    def train_single_feature_model(self, X, y, model_type='rf', feature_name=None):
        """
        训练单个特征的模型
        
        参数:
        X: numpy.ndarray, 特征数据 (单个特征)
        y: numpy.ndarray, 标签数据
        model_type: str, 模型类型
        feature_name: str, 特征名称
        
        返回:
        tuple, 包含训练好的模型和评估结果
        """
        # 选择模型
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42)
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=42)
        else:
            raise ValueError("不支持的模型类型，可选模型为 'rf', 'lr' 或 'svm'")
        
        # 训练模型
        model.fit(X, y)
        
        # 交叉验证评估
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc_ovr', n_jobs=1)
        mean_roc_auc = scores.mean()
        
        # 保存模型
        self.model = model
        
        return model, {
            'mean_roc_auc': mean_roc_auc,
            'cv_scores': scores,
            'feature_name': feature_name
        }
    
    def cross_validate_model(self, X, y, model_type='rf', cv=5, hyperparams=None):
        """
        使用交叉验证评估模型性能
        
        参数:
        X: numpy.ndarray, 特征数据
        y: numpy.ndarray, 标签数据
        model_type: str, 模型类型
        cv: int, 交叉验证折数
        hyperparams: dict, 超参数设置
        
        返回:
        dict, 包含交叉验证的性能指标
        """
        # 选择模型
        if model_type == 'rf':
            model = RandomForestClassifier(random_state=42, **(hyperparams or {}))
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42, **(hyperparams or {}))
        elif model_type == 'svm':
            model = SVC(random_state=42, probability=True, **(hyperparams or {}))
        else:
            raise ValueError("不支持的模型类型，可选模型为 'rf', 'lr' 或 'svm'")
        
        # 交叉验证
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted')
        recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc_ovr')
        
        # 创建结果字典
        results = {
            'accuracy': {
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std(),
                'scores': accuracy_scores
            },
            'precision': {
                'mean': precision_scores.mean(),
                'std': precision_scores.std(),
                'scores': precision_scores
            },
            'recall': {
                'mean': recall_scores.mean(),
                'std': recall_scores.std(),
                'scores': recall_scores
            },
            'f1_score': {
                'mean': f1_scores.mean(),
                'std': f1_scores.std(),
                'scores': f1_scores
            },
            'roc_auc': {
                'mean': roc_auc_scores.mean(),
                'std': roc_auc_scores.std(),
                'scores': roc_auc_scores
            }
        }
        
        return results
