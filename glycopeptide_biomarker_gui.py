#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖肽诊断方法机器学习分析GUI程序

该程序用于分析糖肽与健康人和病人血清的结合力数据，找出最佳的诊断糖肽或糖肽组合。
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
import matplotlib
# 设置matplotlib后端为非交互式，避免在打包的exe中出现问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_processing import GlycopeptideDataProcessor
from feature_selection import GlycopeptideFeatureSelector
from model_training import GlycopeptideModelTrainer
from roc_analysis import ROCAnalyzer
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue

class GlycopeptideBiomarkerGUI:
    """
    糖肽诊断GUI类
    """
    
    def __init__(self, root):
        """
        初始化GUI
        """
        self.root = root
        self.root.title("IgAN糖肽生物标志物机器学习分析 / IgAN Glycopeptide Biomarker Machine Learning Analysis")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 创建日志队列
        self.log_queue = queue.Queue()
        
        # 初始化各个模块
        self.data_processor = GlycopeptideDataProcessor()
        self.feature_selector = GlycopeptideFeatureSelector()
        self.model_trainer = GlycopeptideModelTrainer()
        self.roc_analyzer = ROCAnalyzer()
        
        # 创建GUI组件
        self.create_widgets()
        
        # 开始日志处理线程
        self.log_thread = threading.Thread(target=self.process_logs, daemon=True)
        self.log_thread.start()
    
    def create_widgets(self):
        """
        创建GUI组件
        """
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 设置列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        # 数据文件选择
        ttk.Label(main_frame, text="数据文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_file_var = tk.StringVar()
        self.data_file_entry = ttk.Entry(main_frame, textvariable=self.data_file_var, width=50)
        self.data_file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_data_file).grid(row=0, column=2, sticky=tk.W, pady=5, padx=5)
        
        # 使用示例数据复选框
        self.use_example_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="使用示例数据", variable=self.use_example_var, command=self.toggle_data_file_entry).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 模型类型选择
        ttk.Label(main_frame, text="模型类型:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.model_type_var = tk.StringVar(value="rf")
        model_types = ttk.Combobox(main_frame, textvariable=self.model_type_var, values=["rf", "lr", "svm"], state="readonly")
        model_types.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 特征数量输入
        ttk.Label(main_frame, text="选择的特征数量:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.n_features_var = tk.IntVar(value=5)
        ttk.Spinbox(main_frame, from_=1, to=20, textvariable=self.n_features_var).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # 输出目录选择
        ttk.Label(main_frame, text="输出目录:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value="output")
        ttk.Entry(main_frame, textvariable=self.output_dir_var, width=50).grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_output_dir).grid(row=4, column=2, sticky=tk.W, pady=5, padx=5)
        
        # 运行按钮
        self.run_button = ttk.Button(main_frame, text="运行分析", command=self.run_analysis)
        self.run_button.grid(row=5, column=0, columnspan=3, pady=10)
        
        # 日志输出区域
        ttk.Label(main_frame, text="运行日志:").grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5)
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, width=80, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 日志滚动条
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # 结果输出区域
        ttk.Label(main_frame, text="分析结果:").grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=5)
        result_frame = ttk.Frame(main_frame)
        result_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 结果文本框
        self.result_text = tk.Text(result_frame, width=80, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果滚动条
        result_scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        result_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
    
    def browse_data_file(self):
        """
        浏览数据文件
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="选择数据文件"
        )
        if file_path:
            self.data_file_var.set(file_path)
            self.use_example_var.set(False)
            self.toggle_data_file_entry()
    
    def browse_output_dir(self):
        """
        浏览输出目录
        """
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def toggle_data_file_entry(self):
        """
        切换数据文件输入框的状态
        """
        if self.use_example_var.get():
            self.data_file_var.set("")
            self.data_file_entry.config(state=tk.DISABLED)
        else:
            self.data_file_entry.config(state=tk.NORMAL)
    
    def run_analysis(self):
        """
        运行分析
        """
        # 禁用运行按钮
        self.run_button.config(state=tk.DISABLED)
        
        # 清空日志和结果
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        
        # 获取参数
        data_file = self.data_file_var.get()
        use_example = self.use_example_var.get()
        model_type = self.model_type_var.get()
        n_features = self.n_features_var.get()
        output_dir = self.output_dir_var.get()
        
        # 验证参数
        if not use_example and not data_file:
            messagebox.showerror("错误", "必须指定数据文件或使用示例数据")
            self.run_button.config(state=tk.NORMAL)
            return
        
        # 创建分析线程
        analysis_thread = threading.Thread(
            target=self.analysis_thread_func,
            args=(data_file, use_example, model_type, n_features, output_dir),
            daemon=True
        )
        analysis_thread.start()
    
    def analysis_thread_func(self, data_file, use_example, model_type, n_features, output_dir):
        """
        分析线程函数
        """
        try:
            # 创建输出目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.log(f"创建输出目录: {output_dir}")
            
            # 加载或生成数据
            if use_example:
                self.log("使用示例数据...")
                from glycopeptide_biomarker import generate_example_data
                glycopeptide_structures, binding_data, labels = generate_example_data()
                data = self.data_processor.prepare_glycopeptide_data(glycopeptide_structures, binding_data, labels)
            elif data_file:
                self.log(f"加载数据文件: {data_file}...")
                data = self.data_processor.load_data(data_file)
            
            # 预处理数据
            self.log("预处理数据...")
            X, y, feature_names = self.data_processor.preprocess_data(data)
            
            # 选择最佳单个特征
            self.log("选择最佳单个特征...")
            best_single_features = self.feature_selector.select_best_single_features(X, y, feature_names, top_k=10)
            self.log("\n最佳单个特征:")
            self.log(str(best_single_features))
            
            # 保存最佳单个特征结果
            best_single_features_path = os.path.join(output_dir, 'best_single_features.csv')
            best_single_features.to_csv(best_single_features_path, index=False)
            self.log(f"最佳单个特征结果已保存到: {best_single_features_path}")
            
            # 选择最佳特征组合
            self.log(f"\n选择最佳特征组合（最多{n_features}个特征）...")
            best_combinations = self.feature_selector.select_best_combination(X, y, feature_names, model_type, n_features)
            
            # 输出最佳组合结果
            for k, result in best_combinations.items():
                self.log(f"\n{k}个特征的最佳组合:")
                self.log(f"特征: {', '.join(result['features'])}")
                self.log(f"平均ROC-AUC得分: {result['mean_roc_auc']:.4f}")
            
            # 保存最佳组合结果
            best_combinations_path = os.path.join(output_dir, 'best_combinations.txt')
            with open(best_combinations_path, 'w') as f:
                for k, result in best_combinations.items():
                    f.write(f"\n{k}个特征的最佳组合:\n")
                    f.write(f"特征: {', '.join(result['features'])}\n")
                    f.write(f"平均ROC-AUC得分: {result['mean_roc_auc']:.4f}\n")
                    f.write(f"交叉验证得分: {', '.join([f'{s:.4f}' for s in result['cv_scores']])}\n")
            self.log(f"最佳组合结果已保存到: {best_combinations_path}")
            
            # 训练最终模型
            self.log(f"\n训练最终模型（{model_type}）...")
            best_k = max(best_combinations, key=lambda k: best_combinations[k]['mean_roc_auc'])
            best_features = best_combinations[best_k]['features']
            self.log(f"使用性能最佳的{best_k}个特征组合: {', '.join(best_features)}")
            
            # 获取最佳特征的索引
            best_feature_indices = [feature_names.index(f) for f in best_features]
            X_best = X[:, best_feature_indices]
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = self.data_processor.split_data(X_best, y)
            
            # 超参数调优
            self.log("超参数调优...")
            best_model = self.model_trainer.hyperparameter_tuning(X_train, y_train, model_type)
            
            # 评估模型
            self.log("\n评估模型性能...")
            results = self.model_trainer.evaluate_model(X_test, y_test)
            self.log(f"准确率: {results['accuracy']:.4f}")
            self.log(f"精确率: {results['precision']:.4f}")
            self.log(f"召回率: {results['recall']:.4f}")
            self.log(f"F1得分: {results['f1_score']:.4f}")
            self.log(f"ROC-AUC得分: {results['roc_auc']:.4f}")
            
            # 保存评估结果
            model_evaluation_path = os.path.join(output_dir, 'model_evaluation.txt')
            with open(model_evaluation_path, 'w') as f:
                f.write("模型评估结果:\n")
                f.write(f"准确率: {results['accuracy']:.4f}\n")
                f.write(f"精确率: {results['precision']:.4f}\n")
                f.write(f"召回率: {results['recall']:.4f}\n")
                f.write(f"F1得分: {results['f1_score']:.4f}\n")
                f.write(f"ROC-AUC得分: {results['roc_auc']:.4f}\n")
            self.log(f"模型评估结果已保存到: {model_evaluation_path}")
            
            # ROC曲线分析
            self.log("\nROC曲线分析...")
            fpr, tpr, thresholds, roc_auc = self.roc_analyzer.compute_roc_curve(y_test, results['y_pred_proba'])
            roc_curve_path = os.path.join(output_dir, 'roc_curve.png')
            
            # 处理多分类情况
            if isinstance(roc_auc, dict):
                roc_auc_value = roc_auc['micro']
            else:
                roc_auc_value = roc_auc
            
            self.roc_analyzer.plot_roc_curve(
                title=f'最佳模型的ROC曲线 (AUC = {roc_auc_value:.2f})',
                save_path=roc_curve_path
            )
            self.log(f"ROC曲线已保存到: {roc_curve_path}")
            
            # 交叉验证ROC曲线
            self.log("\n交叉验证ROC曲线分析...")
            cv_roc_curve_path = os.path.join(output_dir, 'cv_roc_curve.png')
            self.roc_analyzer.cross_validation_roc(
                X_best, y, best_model,
                title=f'交叉验证ROC曲线 (AUC = {best_combinations[best_k]["mean_roc_auc"]:.2f})',
                save_path=cv_roc_curve_path
            )
            self.log(f"交叉验证ROC曲线已保存到: {cv_roc_curve_path}")
            
            # 寻找最佳阈值
            self.log("\n寻找最佳阈值...")
            optimal_threshold, optimal_index = self.roc_analyzer.find_optimal_threshold(y_test, results['y_pred_proba'])
            
            # 处理多分类情况
            if isinstance(optimal_threshold, dict):
                # 多分类情况
                class_names = ['Health Control', 'IgAN Patients', 'Non-IgAN Patients Control']
                for i in optimal_threshold.keys():
                    self.log(f"类别 {class_names[i]} 的最佳阈值: {optimal_threshold[i]:.4f}")
            else:
                # 二分类情况
                self.log(f"最佳阈值: {optimal_threshold:.4f}")
                self.log(f"对应的真阳性率: {tpr[optimal_index]:.4f}")
                self.log(f"对应的假阳性率: {fpr[optimal_index]:.4f}")
            
            # 绘制精确率-召回率曲线
            self.log("\n绘制精确率-召回率曲线...")
            precision_recall_curve_path = os.path.join(output_dir, 'precision_recall_curve.png')
            self.roc_analyzer.plot_precision_recall_curve(
                y_test, results['y_pred_proba'],
                title='最佳模型的精确率-召回率曲线',
                save_path=precision_recall_curve_path
            )
            self.log(f"精确率-召回率曲线已保存到: {precision_recall_curve_path}")
            
            self.log(f"\n分析完成，结果保存在 {output_dir} 目录中")
            
            # 显示完成消息
            self.root.after(0, lambda: messagebox.showinfo("完成", f"分析完成，结果保存在 {output_dir} 目录中"))
        except Exception as error:
            error_message = str(error)
            self.log(f"\n错误: {error_message}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"分析过程中发生错误: {error_message}"))
        finally:
            # 启用运行按钮
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))
    
    def log(self, message):
        """
        记录日志
        """
        self.log_queue.put(message)
    
    def process_logs(self):
        """
        处理日志
        """
        while True:
            try:
                message = self.log_queue.get(timeout=0.1)
                self.root.after(0, self.update_log, message)
            except queue.Empty:
                pass
    
    def update_log(self, message):
        """
        更新日志显示
        """
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 同时更新结果显示
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)



# 主函数
if __name__ == "__main__":
    root = tk.Tk()
    app = GlycopeptideBiomarkerGUI(root)
    root.mainloop()