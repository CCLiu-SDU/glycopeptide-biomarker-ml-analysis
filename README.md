# 糖肽生物标志物机器学习分析系统 / Glycopeptide Biomarker Machine Learning Analysis System

本项目开发了一套机器学习算法，用于分析糖肽与健康人和病人血清的相互作用，找出区分能力最好的糖肽或糖肽组合，为潜在的诊断方法提供支持。
This project has developed a set of machine learning algorithms to analyze the interaction between glycopeptides and serum from healthy individuals and patients, identifying the glycopeptides or glycopeptide combinations with the best discriminatory ability to provide support for potential diagnostic methods.

## 项目功能 / Project Features

1. **数据处理与特征工程 / Data Processing and Feature Engineering**
   - 加载和预处理糖肽结合力数据
     Load and preprocess glycopeptide binding affinity data
   - 特征标准化和标签转换
     Feature standardization and label transformation
   - 数据分割和交叉验证准备
     Data splitting and cross-validation preparation

2. **特征选择算法 / Feature Selection Algorithms**
   - 单个特征（单个糖肽）的统计检验（F检验、互信息）
     Statistical tests for individual features (individual glycopeptides) (F-test, mutual information)
   - 递归特征消除（RFE）选择最佳特征组合
     Recursive Feature Elimination (RFE) to select optimal feature combinations
   - 不同特征数量下的性能评估
     Performance evaluation under different feature numbers
   - 特征重要性分析
     Feature importance analysis

3. **机器学习模型训练与评估 / Machine Learning Model Training and Evaluation**
   - 支持随机森林（RF）、逻辑回归（LR）和支持向量机（SVM）
     Support Random Forest (RF), Logistic Regression (LR), and Support Vector Machine (SVM)
   - 超参数调优
     Hyperparameter tuning
   - 交叉验证评估
     Cross-validation evaluation
   - 多性能指标计算（准确率、精确率、召回率、F1得分、ROC-AUC）
     Multi-performance metric calculation (accuracy, precision, recall, F1 score, ROC-AUC)

4. **ROC曲线分析 / ROC Curve Analysis**
   - ROC曲线绘制和AUC计算
     ROC curve plotting and AUC calculation
   - 交叉验证ROC曲线
     Cross-validation ROC curves
   - 最佳阈值选择（约登指数法、最接近左上角法）
     Optimal threshold selection (Youden's index method, closest to top-left corner method)
   - 精确率-召回率曲线
     Precision-recall curves

5. **图形用户界面（GUI）/ Graphical User Interface (GUI)**
   - 友好的用户界面，无需编程知识
     User-friendly interface, no programming knowledge required
   - 支持数据文件选择和参数配置
     Support data file selection and parameter configuration
   - 实时显示分析进度和结果
     Real-time display of analysis progress and results
   - 可视化结果展示
     Visualized result display

## 糖基化类型说明 / Glycosylation Types Description

本系统支持三种糖基化类型：
This system supports three types of glycosylation:

- **G0**：表示该糖基化位点没有糖链（未糖基化）
  Indicates that this glycosylation site has no sugar chain (unglycosylated)

- **G1**：表示该糖基化位点有GalNAc糖基化修饰
  Indicates that this glycosylation site has GalNAc glycosylation modification

- **G2**：表示该糖基化位点有Siaalpha2,6GalNAc糖基化修饰
  Indicates that this glycosylation site has Siaalpha2,6GalNAc glycosylation modification

## 标签分类 / Label Classification

本系统支持三类标签：
This system supports three classes of labels:

- **0**：健康对照（health control）
- **1**：IgAN病人（IgAN patients）
- **2**：非IgAN病人对照（non-IgAN patients control）

## 项目结构 / Project Structure

```
igang/
├── data_processing.py           # 数据处理和特征工程模块 / Data processing and feature engineering module
├── feature_selection.py         # 特征选择算法模块 / Feature selection algorithms module
├── model_training.py            # 模型训练和评估模块 / Model training and evaluation module
├── roc_analysis.py              # ROC曲线分析模块 / ROC curve analysis module
├── glycopeptide_biomarker.py     # 主程序脚本（命令行）/ Main program script (command line)
├── glycopeptide_biomarker_gui.py # GUI程序脚本 / GUI program script
├── glycopeptide_biomarker_gui.spec # PyInstaller配置文件 / PyInstaller configuration file
├── example.csv                   # 示例数据文件 / Example data file
├── requirements.txt              # 项目依赖 / Project dependencies
├── data_format_example.txt       # 数据格式说明 / Data format description
└── README.md                     # 项目文档 / Project documentation
```

## 安装方法 / Installation

1. 克隆或下载项目到本地
   Clone or download the project to local

2. 安装所需的Python库：
   Install required Python libraries:

```bash
pip install -r requirements.txt
```

## 使用说明 / Usage Instructions

### 方法1：使用GUI程序（推荐）/ Method 1: Use GUI Program (Recommended)

#### 运行GUI程序 / Run GUI Program

**选项A：直接运行可执行文件 / Option A: Directly Run Executable File**
```
双击运行 glycopeptide_biomarker_gui.exe
Double-click to run glycopeptide_biomarker_gui.exe
```

**选项B：使用Python运行 / Option B: Run with Python**
```bash
python glycopeptide_biomarker_gui.py
```

#### GUI操作步骤 / GUI Operation Steps

1. **选择数据文件 / Select Data File**
   - 点击"选择数据文件"按钮
     Click the "Select Data File" button
   - 选择包含糖肽数据的CSV文件
     Select the CSV file containing glycopeptide data
   - 或勾选"使用示例数据"使用内置示例
     Or check "Use Example Data" to use built-in examples

2. **配置参数 / Configure Parameters**
   - 选择机器学习模型类型（随机森林/逻辑回归/SVM）
     Select machine learning model type (Random Forest/Logistic Regression/SVM)
   - 设置最大特征数量
     Set maximum number of features
   - 设置交叉验证折数
     Set number of cross-validation folds
   - 设置训练集比例
     Set training set ratio

3. **开始分析 / Start Analysis**
   - 点击"开始分析"按钮
     Click the "Start Analysis" button
   - 等待分析完成
     Wait for analysis to complete
   - 查看分析结果和可视化图表
     View analysis results and visualization charts

4. **查看结果 / View Results**
   - 结果将保存在output目录中
     Results will be saved in the output directory
   - 包括CSV结果文件和PNG图表
     Including CSV result files and PNG charts

### 方法2：使用命令行程序 / Method 2: Use Command Line Program

#### 1. 使用示例数据 / 1. Use Example Data

运行以下命令使用内置的示例数据进行分析：
Run the following command to analyze using built-in example data:

```bash
python glycopeptide_biomarker.py --example
```

#### 2. 使用自定义数据 / 2. Use Custom Data

准备CSV格式的自定义数据，数据格式如下：
Prepare custom data in CSV format, data format as follows:

- 每行代表一个样本（健康对照、IgAN病人或非IgAN病人对照）
  Each row represents a sample (health control, IgAN patient, or non-IgAN patient control)
- 每列代表一个糖肽的结合力
  Each column represents the binding affinity of a glycopeptide
- 最后一列是标签（0=健康对照，1=IgAN病人，2=非IgAN病人对照）
  The last column is the label (0=health control, 1=IgAN patient, 2=non-IgAN patient control)

示例数据格式：
Example data format:

```
GP_1(G1)GP_2(G1)GP_3(G1)GP_4(G1)GP_5(G1)GP_6(G1),GP_1(G1)GP_2(G1)GP_3(G1)GP_4(G1)GP_5(G0)GP_6(G1),...,label
1.876,1.923,1.854,...,0
6.876,6.923,6.854,...,1
2.876,2.923,2.854,...,2
```

运行命令使用自定义数据：
Run the command to use custom data:

```bash
python glycopeptide_biomarker.py --data-file your_data.csv
```

#### 3. 参数说明 / 3. Parameter Description

- `--data-file`: 指定数据文件路径（CSV格式）
  Specify data file path (CSV format)
- `--example`: 使用内置示例数据
  Use built-in example data
- `--model-type`: 选择机器学习模型类型，可选值为 'rf'（随机森林）、'lr'（逻辑回归）、'svm'（支持向量机），默认值为 'rf'
  Select machine learning model type, optional values are 'rf' (Random Forest), 'lr' (Logistic Regression), 'svm' (Support Vector Machine), default value is 'rf'
- `--n-features`: 指定选择的最大特征数量，默认值为 5
  Specify the maximum number of features to select, default value is 5
- `--output-dir`: 指定输出结果目录，默认值为 'output'
  Specify output results directory, default value is 'output'
- `--cv-folds`: 交叉验证折数，默认值为 5
  Number of cross-validation folds, default value is 5
- `--test-size`: 测试集比例，默认值为 0.2
  Test set ratio, default value is 0.2

## 输出结果 / Output Results

分析完成后，结果将保存在指定的输出目录中，包括：
After analysis is completed, results will be saved in the specified output directory, including:

1. **best_single_features.csv**：最佳单个特征的排序结果
   Sorted results of best individual features

2. **best_combinations.txt**：不同特征数量下的最佳组合和性能
   Best combinations and performance under different feature numbers

3. **model_evaluation.txt**：最终模型的性能评估结果
   Performance evaluation results of the final model

4. **roc_curve.png**：ROC曲线图像
   ROC curve image

5. **cv_roc_curve.png**：交叉验证ROC曲线图像
   Cross-validation ROC curve image

6. **precision_recall_curve.png**：精确率-召回率曲线图像
   Precision-recall curve image

## 示例分析流程 / Example Analysis Workflow

1. **数据加载与预处理**：从文件或示例生成数据，进行标准化和标签转换
   **Data Loading and Preprocessing**: Generate data from files or examples, perform standardization and label transformation

2. **最佳单个特征选择**：使用统计检验找出区分能力最强的单个糖肽
   **Best Individual Feature Selection**: Use statistical tests to find the single glycopeptide with the strongest discriminatory ability

3. **最佳特征组合选择**：使用递归特征消除选择不同数量的特征组合
   **Best Feature Combination Selection**: Use recursive feature elimination to select feature combinations of different numbers

4. **模型训练与调优**：使用选择的特征组合训练模型并进行超参数调优
   **Model Training and Tuning**: Train the model using selected feature combinations and perform hyperparameter tuning

5. **模型评估**：计算性能指标并绘制ROC曲线
   **Model Evaluation**: Calculate performance metrics and plot ROC curves

6. **最佳阈值选择**：确定最佳的诊断阈值
   **Optimal Threshold Selection**: Determine the optimal diagnostic threshold

## 技术细节 / Technical Details

### 特征选择方法 / Feature Selection Methods

- **F检验**：基于方差分析的统计检验，评估特征与标签之间的线性关系
  **F-test**: Statistical test based on analysis of variance, evaluating the linear relationship between features and labels

- **互信息**：评估特征与标签之间的非线性关系
  **Mutual Information**: Evaluates the non-linear relationship between features and labels

- **递归特征消除（RFE）**：通过反复训练模型并消除最不重要的特征来选择最佳组合
  **Recursive Feature Elimination (RFE)**: Selects the best combination by repeatedly training the model and eliminating the least important features

### 机器学习模型 / Machine Learning Models

- **随机森林**：集成学习算法，通过构建多个决策树并平均预测结果提高准确性
  **Random Forest**: Ensemble learning algorithm that improves accuracy by constructing multiple decision trees and averaging prediction results

- **逻辑回归**：线性分类器，适合多分类问题，提供概率输出
  **Logistic Regression**: Linear classifier suitable for multi-class problems, providing probability outputs

- **支持向量机**：通过寻找最大间隔超平面进行分类，支持核函数变换
  **Support Vector Machine**: Classifies by finding the maximum margin hyperplane, supports kernel function transformation

### 评估指标 / Evaluation Metrics

- **准确率**：正确预测的样本比例
  **Accuracy**: Proportion of correctly predicted samples

- **精确率**：预测为某类的样本中实际为该类的比例
  **Precision**: Proportion of samples predicted as a class that are actually that class

- **召回率**：实际为某类的样本中被正确预测的比例
  **Recall**: Proportion of samples actually belonging to a class that are correctly predicted

- **F1得分**：精确率和召回率的调和平均
  **F1 Score**: Harmonic mean of precision and recall

- **ROC-AUC**：ROC曲线下面积，评估模型区分能力（使用One-vs-Rest策略处理多分类）
  **ROC-AUC**: Area under the ROC curve, evaluating model discrimination ability (using One-vs-Rest strategy for multi-class)

## 注意事项 / Important Notes

1. 建议使用标准化或归一化后的数据以提高模型性能
   It is recommended to use standardized or normalized data to improve model performance

2. 对于不平衡数据，建议使用过采样或欠采样技术
   For imbalanced data, it is recommended to use oversampling or undersampling techniques

3. 交叉验证有助于评估模型的泛化能力
   Cross-validation helps evaluate the generalization ability of the model

4. 最佳特征数量应根据实际数据和应用需求选择
   The optimal number of features should be selected based on actual data and application requirements

5. IgAN病人（label=1）的糖肽结合力通常比健康对照（label=0）和非IgAN病人对照（label=2）更强
   The glycopeptide binding affinity of IgAN patients (label=1) is usually stronger than that of healthy controls (label=0) and non-IgAN patients controls (label=2)

6. 确保数据文件包含所有三类标签的样本
   Ensure the data file contains samples from all three label classes

## 扩展建议 / Extension Suggestions

1. 添加更多的特征选择算法（如LASSO、弹性网络）
   Add more feature selection algorithms (such as LASSO, Elastic Net)

2. 支持深度学习模型（如神经网络）
   Support deep learning models (such as neural networks)

3. 实现特征可视化功能
   Implement feature visualization functionality

4. 添加模型解释性分析（如SHAP值）
   Add model interpretability analysis (such as SHAP values)

5. 支持更多的数据格式和导入方式
   Support more data formats and import methods

6. 添加批量分析功能
   Add batch analysis functionality

## 常见问题 / Frequently Asked Questions

### Q: 如何重新编译GUI可执行文件？/ Q: How to recompile the GUI executable file?

A: 使用PyInstaller重新编译：
A: Recompile using PyInstaller:

```bash
pyinstaller glycopeptide_biomarker_gui.spec
```

### Q: 示例数据包含多少样本？/ Q: How many samples does the example data contain?

A: 示例数据包含60个样本，每类标签（0,1,2）各20个样本，共14个糖肽结构。
A: The example data contains 60 samples, with 20 samples for each label class (0, 1, 2), totaling 14 glycopeptide structures.

### Q: 支持哪些糖基化类型？/ Q: What glycosylation types are supported?

A: 支持三种糖基化类型：G0（未糖基化）、G1（GalNAc糖基化）、G2（Siaalpha2,6GalNAc糖基化）。
A: Three glycosylation types are supported: G0 (unglycosylated), G1 (GalNAc glycosylation), G2 (Siaalpha2,6GalNAc glycosylation).

### Q: 如何处理缺失值？/ Q: How to handle missing values?

A: 确保CSV文件中没有缺失值。如果有缺失值，可以在加载数据前进行填充或删除。
A: Ensure there are no missing values in the CSV file. If there are missing values, you can fill or delete them before loading the data.

## 联系方式 / Contact

如有问题或建议，请联系项目负责人。
If you have any questions or suggestions, please contact the project manager.

---

© 2025 糖肽生物标志物机器学习分析系统 / © 2025 Glycopeptide Biomarker Machine Learning Analysis System
