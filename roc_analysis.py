import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

# 禁用matplotlib字体警告
warnings.filterwarnings('ignore', message='Glyph.*missing from font')

# 配置中文字体，避免中文显示问题
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 
                                         'WenQuanYi Micro Hei', 'Heiti TC', 'STHeiti',
                                         'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

class ROCAnalyzer:
    """
    ROC曲线分析类，用于绘制和分析ROC曲线
    """
    
    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.roc_auc = None
        
    def compute_roc_curve(self, y_true, y_pred_proba):
        """
        计算ROC曲线
        
        参数:
        y_true: numpy.ndarray, 真实标签
        y_pred_proba: numpy.ndarray, 预测概率
        
        返回:
        fpr: numpy.ndarray, 假阳性率
        tpr: numpy.ndarray, 真阳性率
        thresholds: numpy.ndarray, 阈值
        roc_auc: float, ROC曲线下面积
        """
        # 处理多分类情况
        if y_pred_proba.ndim > 1:
            # 对于多分类，使用One-vs-Rest方法计算ROC曲线
            from sklearn.preprocessing import label_binarize
            n_classes = y_pred_proba.shape[1]
            y_bin = label_binarize(y_true, classes=range(n_classes))
            
            # 计算每个类别的ROC曲线
            fpr = dict()
            tpr = dict()
            thresholds = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], thresholds[i] = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 计算微平均ROC曲线
            fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            self.fpr = fpr
            self.tpr = tpr
            self.thresholds = thresholds
            self.roc_auc = roc_auc
            
            return fpr, tpr, thresholds, roc_auc
        else:
            # 二分类情况
            self.fpr, self.tpr, self.thresholds = roc_curve(y_true, y_pred_proba)
            self.roc_auc = auc(self.fpr, self.tpr)
            
            return self.fpr, self.tpr, self.thresholds, self.roc_auc
    
    def plot_roc_curve(self, title='ROC Curve', save_path=None):
        """
        Plot ROC curve
        
        Parameters:
        title: str, Plot title
        save_path: str, Path to save the plot
        """
        if self.fpr is None or self.tpr is None:
            raise ValueError("Please call compute_roc_curve method first")
        
        plt.figure(figsize=(10, 8))
        
        # 处理多分类情况
        if isinstance(self.fpr, dict):
            # 为每个类别绘制ROC曲线
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            class_names = ['Health Control', 'IgAN Patients', 'Non-IgAN Patients Control']
            
            for i, color in zip(self.fpr.keys(), colors):
                if isinstance(i, int):
                    plt.plot(self.fpr[i], self.tpr[i], color=color, lw=2, 
                             label=f'{class_names[i]} (AUC = {self.roc_auc[i]:.2f})')
            
            # 绘制微平均ROC曲线
            plt.plot(self.fpr["micro"], self.tpr["micro"], 
                     color='deeppink', linestyle=':', lw=4, 
                     label=f'Micro-average (AUC = {self.roc_auc["micro"]:.2f})')
        else:
            # 二分类情况
            plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, 
                     label=f'ROC Curve (AUC = {self.roc_auc:.2f})')
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        plt.close()
    
    def plot_multiple_roc_curves(self, roc_data_list, titles=None, save_path=None):
        """
        Plot multiple ROC curves for comparison
        
        Parameters:
        roc_data_list: list, List containing multiple ROC data elements, each (fpr, tpr, roc_auc, label)
        titles: list, Labels for each ROC curve
        save_path: str, Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Plot each ROC curve
        for i, (fpr, tpr, roc_auc, label) in enumerate(roc_data_list):
            # 处理多分类情况
            if isinstance(fpr, dict):
                # 使用微平均ROC曲线
                plt.plot(fpr['micro'], tpr['micro'], lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
            else:
                # 二分类情况
                plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Comparison of Multiple ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        
        plt.close()
    
    def cross_validation_roc(self, X, y, model, cv=5, title='Cross-Validation ROC', save_path=None):
        """
        Plot cross-validation ROC curve
        
        Parameters:
        X: numpy.ndarray, Feature data
        y: numpy.ndarray, Label data
        model: Machine learning model
        cv: int, Number of cross-validation folds
        title: str, Plot title
        save_path: str, Path to save the plot
        
        Returns:
        mean_fpr: numpy.ndarray, Mean false positive rate
        mean_tpr: numpy.ndarray, Mean true positive rate
        mean_auc: float, Mean AUC score
        """
        from sklearn.preprocessing import label_binarize
        
        cv = StratifiedKFold(n_splits=cv)
        
        # 检查是否为多分类问题
        n_classes = len(np.unique(y))
        is_multiclass = n_classes > 2
        
        if is_multiclass:
            y_bin = label_binarize(y, classes=range(n_classes))
            plt.figure(figsize=(10, 8))
            
            # 为每个类别存储TPR
            mean_tpr = dict()
            mean_fpr = np.linspace(0, 1, 100)
            class_names = ['Health Control', 'IgAN Patients', 'Non-IgAN Patients Control']
            colors = ['blue', 'green', 'red']
            
            # 初始化每个类别的mean_tpr
            for i in range(n_classes):
                mean_tpr[i] = 0.0
            
            # 计算微平均
            mean_tpr_micro = 0.0
            
            for i, (train, test) in enumerate(cv.split(X, y)):
                # Train model
                model.fit(X[train], y[train])
                
                # Predict probabilities
                y_pred_proba = model.predict_proba(X[test])
                
                # 为每个类别计算ROC曲线
                for j in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_bin[test, j], y_pred_proba[:, j])
                    mean_tpr[j] += np.interp(mean_fpr, fpr, tpr)
                    mean_tpr[j][0] = 0.0
                
                # 计算微平均
                fpr_micro, tpr_micro, _ = roc_curve(y_bin[test].ravel(), y_pred_proba.ravel())
                mean_tpr_micro += np.interp(mean_fpr, fpr_micro, tpr_micro)
                mean_tpr_micro[0] = 0.0
            
            # 计算每个类别的平均ROC曲线
            for i in range(n_classes):
                mean_tpr[i] /= cv.get_n_splits(X, y)
                mean_tpr[i][-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr[i])
                plt.plot(mean_fpr, mean_tpr[i], color=colors[i], lw=2, 
                         label=f'{class_names[i]} (AUC = {mean_auc:.2f})')
            
            # 计算微平均ROC曲线
            mean_tpr_micro /= cv.get_n_splits(X, y)
            mean_tpr_micro[-1] = 1.0
            mean_auc_micro = auc(mean_fpr, mean_tpr_micro)
            plt.plot(mean_fpr, mean_tpr_micro, 
                     color='deeppink', linestyle=':', lw=4, 
                     label=f'Micro-average (AUC = {mean_auc_micro:.2f})')
        else:
            # 二分类情况
            plt.figure(figsize=(8, 6))
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            
            for i, (train, test) in enumerate(cv.split(X, y)):
                # Train model
                model.fit(X[train], y[train])
                
                # Predict probabilities
                y_pred_proba = model.predict_proba(X[test])[:, 1]
                
                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y[test], y_pred_proba)
                
                # Interpolate to same fpr points
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                
                # Plot single fold ROC curve
                plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {auc(fpr, tpr):.2f})')
            
            # Compute mean ROC curve
            mean_tpr /= cv.get_n_splits(X, y)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            
            # Plot mean ROC curve
            plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, label=f'Mean ROC (AUC = {mean_auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cross-validation ROC curve saved to: {save_path}")
        
        plt.close()
        
        return mean_fpr, mean_tpr, mean_auc
    
    def find_optimal_threshold(self, y_true, y_pred_proba, method='youden'):
        """
        寻找最佳阈值
        
        参数:
        y_true: numpy.ndarray, 真实标签
        y_pred_proba: numpy.ndarray, 预测概率
        method: str, 最佳阈值选择方法，可选 'youden' (约登指数) 或 'closest_to_top_left' (最接近左上角)
        
        返回:
        optimal_threshold: float, 最佳阈值
        optimal_index: int, 最佳阈值在thresholds中的索引
        """
        # 处理多分类情况
        if y_pred_proba.ndim > 1:
            # 对于多分类，返回每个类别的最佳阈值
            optimal_thresholds = dict()
            optimal_indices = dict()
            
            # 计算ROC曲线
            self.compute_roc_curve(y_true, y_pred_proba)
            
            # 只遍历实际的类别键，排除'micro'等非类别键
            for i in self.fpr.keys():
                if isinstance(i, int):  # 只处理整数类别的键
                    if method == 'youden':
                        # 约登指数法：TPR - FPR 最大化
                        youden_index = self.tpr[i] - self.fpr[i]
                        optimal_index = np.argmax(youden_index)
                    elif method == 'closest_to_top_left':
                        # 最接近左上角法
                        distances = np.sqrt(self.fpr[i]**2 + (1 - self.tpr[i])**2)
                        optimal_index = np.argmin(distances)
                    else:
                        raise ValueError("不支持的阈值选择方法，可选方法为 'youden' 或 'closest_to_top_left'")
                    
                    optimal_thresholds[i] = self.thresholds[i][optimal_index]
                    optimal_indices[i] = optimal_index
            
            return optimal_thresholds, optimal_indices
        else:
            # 二分类情况
            if method == 'youden':
                # 约登指数法：TPR - FPR 最大化
                self.compute_roc_curve(y_true, y_pred_proba)
                youden_index = self.tpr - self.fpr
                optimal_index = np.argmax(youden_index)
            elif method == 'closest_to_top_left':
                # 最接近左上角法
                self.compute_roc_curve(y_true, y_pred_proba)
                distances = np.sqrt(self.fpr**2 + (1 - self.tpr)**2)
                optimal_index = np.argmin(distances)
            else:
                raise ValueError("不支持的阈值选择方法，可选方法为 'youden' 或 'closest_to_top_left'")
            
            optimal_threshold = self.thresholds[optimal_index]
            
            return optimal_threshold, optimal_index
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, title='Precision-Recall Curve', save_path=None):
        """
        Plot precision-recall curve
        
        Parameters:
        y_true: numpy.ndarray, True labels
        y_pred_proba: numpy.ndarray, Predicted probabilities
        title: str, Plot title
        save_path: str, Path to save the plot
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import average_precision_score
        
        # 处理多分类情况
        if y_pred_proba.ndim > 1:
            n_classes = y_pred_proba.shape[1]
            y_bin = label_binarize(y_true, classes=range(n_classes))
            class_names = ['Health Control', 'IgAN Patients', 'Non-IgAN Patients Control']
            colors = ['blue', 'green', 'red']
            
            plt.figure(figsize=(10, 8))
            
            # 为每个类别绘制精确率-召回率曲线
            for i, color in zip(range(n_classes), colors):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
                avg_precision = average_precision_score(y_bin[:, i], y_pred_proba[:, i])
                plt.plot(recall, precision, color=color, lw=2, 
                         label=f'{class_names[i]} (AP = {avg_precision:.2f})')
            
            # 计算微平均精确率-召回率曲线
            precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), y_pred_proba.ravel())
            avg_precision_micro = average_precision_score(y_bin, y_pred_proba, average='micro')
            plt.plot(recall_micro, precision_micro, 
                     color='deeppink', linestyle=':', lw=4, 
                     label=f'Micro-average (AP = {avg_precision_micro:.2f})')
        else:
            # 二分类情况
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to: {save_path}")
        
        plt.close()
