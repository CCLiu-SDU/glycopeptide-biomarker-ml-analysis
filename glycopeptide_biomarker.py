#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖肽诊断方法机器学习分析主程序

该程序用于分析糖肽与健康人和病人血清的结合力数据，找出最佳的诊断糖肽或糖肽组合。
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
import matplotlib
import warnings

# 禁用matplotlib字体警告
warnings.filterwarnings('ignore', message='Glyph.*missing from font')

# 设置matplotlib后端为非交互式，避免在打包的exe中出现问题
matplotlib.use('Agg')

# 配置中文字体，避免中文显示问题
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 
                                         'WenQuanYi Micro Hei', 'Heiti TC', 'STHeiti',
                                         'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入matplotlib.pyplot
import matplotlib.pyplot as plt
from data_processing import GlycopeptideDataProcessor
from feature_selection import GlycopeptideFeatureSelector
from model_training import GlycopeptideModelTrainer
from roc_analysis import ROCAnalyzer

def generate_example_data():
    """
    生成示例数据用于演示
    
    返回:
    glycopeptide_structures: list, 糖肽结构列表
    binding_data: numpy.ndarray, 结合力数据
    labels: numpy.ndarray, 标签数据
    """
    # 生成6个潜在糖基化位点的糖肽结构（示例）
    # G0：未糖基化，G1：GalNAc糖基化修饰，G2：Siaalpha2,6GalNAc糖基化修饰
    glycopeptide_structures = [
        "GP_1(G1)GP_2(G0)GP_3(G0)GP_4(G2)GP_5(G0)GP_6(G0)",
        "GP_1(G0)GP_2(G1)GP_3(G0)GP_4(G0)GP_5(G2)GP_6(G0)",
        "GP_1(G0)GP_2(G0)GP_3(G1)GP_4(G0)GP_5(G0)GP_6(G2)",
        "GP_1(G1)GP_2(G1)GP_3(G0)GP_4(G0)GP_5(G0)GP_6(G2)",
        "GP_1(G0)GP_2(G0)GP_3(G1)GP_4(G2)GP_5(G0)GP_6(G0)",
        "GP_1(G0)GP_2(G0)GP_3(G0)GP_4(G0)GP_5(G1)GP_6(G2)",
        "GP_1(G1)GP_2(G0)GP_3(G2)GP_4(G0)GP_5(G1)GP_6(G0)",
        "GP_1(G0)GP_2(G2)GP_3(G0)GP_4(G1)GP_5(G0)GP_6(G1)",
        "GP_1(G1)GP_2(G1)GP_3(G2)GP_4(G0)GP_5(G0)GP_6(G0)",
        "GP_1(G0)GP_2(G0)GP_3(G0)GP_4(G2)GP_5(G1)GP_6(G1)"
    ]
    
    # 生成结合力数据
    np.random.seed(42)
    n_samples = 150  # 50个健康对照，50个IgAN病人，50个非IgAN病人对照
    n_glycopeptides = len(glycopeptide_structures)
    
    # IgAN病人的结合力数据（均值较高）
    igan_patients_binding = np.random.normal(5.0, 1.0, size=(50, n_glycopeptides))
    
    # 健康对照的结合力数据（均值较低）
    health_control_binding = np.random.normal(3.0, 1.0, size=(50, n_glycopeptides))
    
    # 非IgAN病人对照的结合力数据（均值介于健康对照和IgAN病人之间）
    non_igan_patients_control_binding = np.random.normal(4.0, 1.0, size=(50, n_glycopeptides))
    
    # 合并数据
    binding_data = np.vstack([health_control_binding, igan_patients_binding, non_igan_patients_control_binding])
    
    # 生成标签（0=健康对照，1=IgAN病人，2=非IgAN病人对照）
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
    
    return glycopeptide_structures, binding_data, labels

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='糖肽诊断方法机器学习分析')
    parser.add_argument('--data-file', type=str, help='数据文件路径（CSV格式）')
    parser.add_argument('--example', action='store_true', help='使用示例数据')
    parser.add_argument('--model-type', type=str, default='rf', choices=['rf', 'lr', 'svm'], 
                        help='机器学习模型类型')
    parser.add_argument('--n-features', type=int, default=5, help='选择的特征数量')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        # 如果是相对路径，使用程序所在目录作为基础路径
        # 在PyInstaller打包的exe中，使用sys.executable获取正确的路径
        if getattr(sys, 'frozen', False):
            # 打包后的exe环境
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            # 正常Python环境
            base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output_dir = output_dir
    
    # 初始化各个模块
    data_processor = GlycopeptideDataProcessor()
    feature_selector = GlycopeptideFeatureSelector()
    model_trainer = GlycopeptideModelTrainer()
    roc_analyzer = ROCAnalyzer()
    
    # 加载或生成数据
    if args.example or not args.data_file:
        print("使用示例数据...")
        glycopeptide_structures, binding_data, labels = generate_example_data()
        data = data_processor.prepare_glycopeptide_data(glycopeptide_structures, binding_data, labels)
    elif args.data_file:
        print(f"加载数据文件: {args.data_file}...")
        data = data_processor.load_data(args.data_file)
    
    # 预处理数据
    print("预处理数据...")
    X, y, feature_names = data_processor.preprocess_data(data)
    
    # 选择最佳单个特征
    print("选择最佳单个特征...")
    best_single_features = feature_selector.select_best_single_features(X, y, feature_names, top_k=10)
    print("\n最佳单个特征:")
    print(best_single_features)
    
    # 保存最佳单个特征结果
    best_single_features.to_csv(os.path.join(args.output_dir, 'best_single_features.csv'), index=False)
    
    # 为每个最佳单个特征生成ROC曲线
    print("\n为每个最佳单个特征生成ROC曲线...")
    single_feature_roc_data = []
    for i, feature_name in enumerate(best_single_features['feature']):
        print(f"\n处理特征 {i+1}/{len(best_single_features)}: {feature_name}")
        
        # 获取单个特征数据
        feature_index = feature_names.index(feature_name)
        X_single = X[:, [feature_index]]
        
        # 划分训练集和测试集
        X_train_single, X_test_single, y_train_single, y_test_single = data_processor.split_data(X_single, y)
        
        # 训练模型
        single_model, single_results = model_trainer.train_single_feature_model(
            X_train_single, y_train_single, args.model_type, feature_name
        )
        
        # 评估模型
        single_eval_results = model_trainer.evaluate_model(X_test_single, y_test_single)
        
        # 计算ROC曲线
        fpr, tpr, thresholds, roc_auc = roc_analyzer.compute_roc_curve(
            y_test_single, single_eval_results['y_pred_proba']
        )
        
        # 处理多分类情况
        if isinstance(roc_auc, dict):
            # 对于多分类，使用微平均ROC-AUC
            roc_auc_value = roc_auc['micro']
            # 保存微平均ROC曲线数据
            roc_df = pd.DataFrame({
                'fpr': fpr['micro'],
                'tpr': tpr['micro'],
                'thresholds': thresholds['micro']
            })
        else:
            # 二分类情况
            roc_auc_value = roc_auc
            roc_df = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            })
        
        # 保存ROC曲线数据到CSV
        roc_df.to_csv(os.path.join(args.output_dir, f'single_feature_roc_{i+1}_{feature_name[:20]}...csv'), index=False)
        
        # 绘制并保存ROC曲线
        roc_analyzer.plot_roc_curve(
            title=f'{feature_name} 的ROC曲线 (AUC = {roc_auc_value:.2f})',
            save_path=os.path.join(args.output_dir, f'single_feature_roc_{i+1}_{feature_name[:20]}...png')
        )
        
        # 保存ROC数据用于后续分析
        single_feature_roc_data.append((fpr, tpr, roc_auc_value, feature_name))
    
    # 绘制所有单个特征的ROC曲线比较
    roc_analyzer.plot_multiple_roc_curves(
        single_feature_roc_data,
        titles=[f'特征 {i+1}: {feature_name}' for i, feature_name in enumerate(best_single_features['feature'])],
        save_path=os.path.join(args.output_dir, 'all_single_features_roc.png')
    )
    
    # 保存所有单个特征的ROC数据到一个CSV文件
    all_roc_data = []
    # 重新计算每个特征的ROC数据，确保能访问到thresholds
    for i, feature_name in enumerate(best_single_features['feature']):
        # 获取单个特征数据
        feature_index = feature_names.index(feature_name)
        X_single = X[:, [feature_index]]
        
        # 划分训练集和测试集
        X_train_single, X_test_single, y_train_single, y_test_single = data_processor.split_data(X_single, y)
        
        # 训练模型
        single_model, single_results = model_trainer.train_single_feature_model(
            X_train_single, y_train_single, args.model_type, feature_name
        )
        
        # 评估模型
        single_eval_results = model_trainer.evaluate_model(X_test_single, y_test_single)
        
        # 计算ROC曲线
        fpr, tpr, thresholds, roc_auc = roc_analyzer.compute_roc_curve(
            y_test_single, single_eval_results['y_pred_proba']
        )
        
        # 保存到all_roc_data
        if isinstance(fpr, dict):
            # 多分类情况，使用微平均
            for j in range(len(fpr['micro'])):
                all_roc_data.append({
                    'feature_index': i+1,
                    'feature_name': feature_name,
                    'fpr': fpr['micro'][j],
                    'tpr': tpr['micro'][j],
                    'threshold': thresholds['micro'][j],
                    'auc': roc_auc['micro']
                })
        else:
            # 二分类情况
            for j in range(len(fpr)):
                all_roc_data.append({
                    'feature_index': i+1,
                    'feature_name': feature_name,
                    'fpr': fpr[j],
                    'tpr': tpr[j],
                    'threshold': thresholds[j],
                    'auc': roc_auc
                })
    all_roc_df = pd.DataFrame(all_roc_data)
    all_roc_df.to_csv(os.path.join(args.output_dir, 'all_single_features_roc.csv'), index=False)
    
    # 选择最佳特征组合
    print(f"\n选择最佳特征组合（最多{args.n_features}个特征）...")
    best_combinations = feature_selector.select_best_combination(X, y, feature_names, 
                                                                 args.model_type, args.n_features)
    
    # 输出最佳组合结果
    for k, result in best_combinations.items():
        print(f"\n{k}个特征的最佳组合:")
        print(f"特征: {', '.join(result['features'])}")
        print(f"平均ROC-AUC得分: {result['mean_roc_auc']:.4f}")
    
    # 保存最佳组合结果
    with open(os.path.join(args.output_dir, 'best_combinations.txt'), 'w') as f:
        for k, result in best_combinations.items():
            f.write(f"\n{k}个特征的最佳组合:\n")
            f.write(f"特征: {', '.join(result['features'])}\n")
            f.write(f"平均ROC-AUC得分: {result['mean_roc_auc']:.4f}\n")
            f.write(f"交叉验证得分: {', '.join([f'{s:.4f}' for s in result['cv_scores']])}\n")
    
    # 为最佳组合的前五个生成ROC曲线
    print("\n为最佳组合的前五个生成ROC曲线...")
    # 按ROC-AUC得分排序最佳组合
    sorted_combinations = sorted(best_combinations.items(), 
                               key=lambda x: x[1]['mean_roc_auc'], 
                               reverse=True)[:5]
    
    combination_roc_data = []
    for i, (k, result) in enumerate(sorted_combinations):
        print(f"\n处理最佳组合 {i+1}/{len(sorted_combinations)}: {k}个特征")
        
        # 获取组合特征数据
        feature_indices = [feature_names.index(f) for f in result['features']]
        X_comb = X[:, feature_indices]
        
        # 划分训练集和测试集
        X_train_comb, X_test_comb, y_train_comb, y_test_comb = data_processor.split_data(X_comb, y)
        
        # 超参数调优
        print(f"  超参数调优...")
        best_model = model_trainer.hyperparameter_tuning(X_train_comb, y_train_comb, args.model_type)
        
        # 评估模型
        comb_eval_results = model_trainer.evaluate_model(X_test_comb, y_test_comb)
        
        # 计算ROC曲线
        fpr, tpr, thresholds, roc_auc = roc_analyzer.compute_roc_curve(
            y_test_comb, comb_eval_results['y_pred_proba']
        )
        
        # 处理多分类情况
        if isinstance(roc_auc, dict):
            # 对于多分类，使用微平均ROC-AUC
            roc_auc_value = roc_auc['micro']
            # 保存微平均ROC曲线数据
            roc_df = pd.DataFrame({
                'fpr': fpr['micro'],
                'tpr': tpr['micro'],
                'thresholds': thresholds['micro']
            })
        else:
            # 二分类情况
            roc_auc_value = roc_auc
            roc_df = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            })
        
        # 保存ROC曲线数据到CSV
        comb_name = f"comb_{k}features_{'_'.join([f[:5] for f in result['features']])}"
        roc_df.to_csv(os.path.join(args.output_dir, f'combination_roc_{i+1}_{comb_name}.csv'), index=False)
        
        # 绘制并保存ROC曲线
        roc_analyzer.plot_roc_curve(
            title=f'{k}个特征的最佳组合的ROC曲线 (AUC = {roc_auc_value:.2f})\n特征: {', '.join(result['features'])}',
            save_path=os.path.join(args.output_dir, f'combination_roc_{i+1}_{comb_name}.png')
        )
        
        # 保存ROC数据用于后续分析
        combination_roc_data.append((fpr, tpr, roc_auc_value, f'{k}个特征: {', '.join(result['features'])}'))
    
    # 绘制所有最佳组合的ROC曲线比较
    roc_analyzer.plot_multiple_roc_curves(
        combination_roc_data,
        titles=[f'组合 {i+1}' for i in range(len(combination_roc_data))],
        save_path=os.path.join(args.output_dir, 'all_combinations_roc.png')
    )
    
    # 保存所有最佳组合的ROC数据到一个CSV文件
    all_comb_roc_data = []
    # 修改combination_roc_data，包含thresholds
    for i, (k, result) in enumerate(sorted_combinations):
        # 获取组合特征数据
        feature_indices = [feature_names.index(f) for f in result['features']]
        X_comb = X[:, feature_indices]
        
        # 划分训练集和测试集
        X_train_comb, X_test_comb, y_train_comb, y_test_comb = data_processor.split_data(X_comb, y)
        
        # 超参数调优
        best_model = model_trainer.hyperparameter_tuning(X_train_comb, y_train_comb, args.model_type)
        
        # 评估模型
        comb_eval_results = model_trainer.evaluate_model(X_test_comb, y_test_comb)
        
        # 计算ROC曲线
        fpr, tpr, thresholds, roc_auc = roc_analyzer.compute_roc_curve(
            y_test_comb, comb_eval_results['y_pred_proba']
        )
        
        # 保存到all_comb_roc_data
        comb_name = f'{k}个特征: {', '.join(result['features'])}'
        if isinstance(fpr, dict):
            # 多分类情况，使用微平均
            for j in range(len(fpr['micro'])):
                all_comb_roc_data.append({
                    'comb_index': i+1,
                    'comb_name': comb_name,
                    'fpr': fpr['micro'][j],
                    'tpr': tpr['micro'][j],
                    'threshold': thresholds['micro'][j],
                    'auc': roc_auc['micro']
                })
        else:
            # 二分类情况
            for j in range(len(fpr)):
                all_comb_roc_data.append({
                    'comb_index': i+1,
                    'comb_name': comb_name,
                    'fpr': fpr[j],
                    'tpr': tpr[j],
                    'threshold': thresholds[j],
                    'auc': roc_auc
                })
    all_comb_roc_df = pd.DataFrame(all_comb_roc_data)
    all_comb_roc_df.to_csv(os.path.join(args.output_dir, 'all_combinations_roc.csv'), index=False)
    
    # 训练最终模型
    print(f"\n训练最终模型（{args.model_type}）...")
    best_k = max(best_combinations, key=lambda k: best_combinations[k]['mean_roc_auc'])
    best_features = best_combinations[best_k]['features']
    print(f"使用性能最佳的{k}个特征组合: {', '.join(best_features)}")
    
    # 获取最佳特征的索引
    best_feature_indices = [feature_names.index(f) for f in best_features]
    X_best = X[:, best_feature_indices]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = data_processor.split_data(X_best, y)
    
    # 超参数调优
    print("超参数调优...")
    best_model = model_trainer.hyperparameter_tuning(X_train, y_train, args.model_type)
    
    # 评估模型
    print("\n评估模型性能...")
    results = model_trainer.evaluate_model(X_test, y_test)
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1得分: {results['f1_score']:.4f}")
    print(f"ROC-AUC得分: {results['roc_auc']:.4f}")
    
    # 保存评估结果
    with open(os.path.join(args.output_dir, 'model_evaluation.txt'), 'w') as f:
        f.write("模型评估结果:\n")
        f.write(f"准确率: {results['accuracy']:.4f}\n")
        f.write(f"精确率: {results['precision']:.4f}\n")
        f.write(f"召回率: {results['recall']:.4f}\n")
        f.write(f"F1得分: {results['f1_score']:.4f}\n")
        f.write(f"ROC-AUC得分: {results['roc_auc']:.4f}\n")
    
    # ROC曲线分析
    print("\nROC曲线分析...")
    fpr, tpr, thresholds, roc_auc = roc_analyzer.compute_roc_curve(y_test, results['y_pred_proba'])
    
    # 处理多分类情况
    if isinstance(roc_auc, dict):
        # 对于多分类，使用微平均ROC-AUC
        roc_auc_value = roc_auc['micro']
    else:
        # 二分类情况
        roc_auc_value = roc_auc
    
    roc_analyzer.plot_roc_curve(
        title=f'最佳模型的ROC曲线 (AUC = {roc_auc_value:.2f})',
        save_path=os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # 交叉验证ROC曲线
    print("\n交叉验证ROC曲线分析...")
    roc_analyzer.cross_validation_roc(
        X_best, y, best_model,
        title=f'交叉验证ROC曲线 (AUC = {best_combinations[best_k]["mean_roc_auc"]:.2f})',
        save_path=os.path.join(args.output_dir, 'cv_roc_curve.png')
    )
    
    # 寻找最佳阈值
    print("\n寻找最佳阈值...")
    optimal_threshold, optimal_index = roc_analyzer.find_optimal_threshold(y_test, results['y_pred_proba'])
    
    # 处理多分类情况
    if isinstance(optimal_threshold, dict):
        # 对于多分类，打印每个类别的最佳阈值
        for i, threshold in optimal_threshold.items():
            print(f"类别 {i} 的最佳阈值: {threshold:.4f}")
    else:
        # 二分类情况
        print(f"最佳阈值: {optimal_threshold:.4f}")
        print(f"对应的真阳性率: {tpr[optimal_index]:.4f}")
        print(f"对应的假阳性率: {fpr[optimal_index]:.4f}")
    
    # 绘制精确率-召回率曲线
    print("\n绘制精确率-召回率曲线...")
    roc_analyzer.plot_precision_recall_curve(
        y_test, results['y_pred_proba'],
        title='最佳模型的精确率-召回率曲线',
        save_path=os.path.join(args.output_dir, 'precision_recall_curve.png')
    )
    
    print(f"\n分析完成，结果保存在 {args.output_dir} 目录中")

if __name__ == "__main__":
    main()
