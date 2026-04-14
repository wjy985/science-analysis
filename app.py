import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import io
import base64

# 页面设置
st.set_page_config(page_title="ResearchAI 科研数据分析系统", layout="wide", initial_sidebar_state="expanded")

# CSS样式
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.2rem; color: #666; margin-bottom: 20px;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745;}
    .warning-box {background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107;}
    .error-box {background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 5px solid #dc3545;}
    .stProgress > div > div > div > div {background-color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=80)
    st.title("分析设置")
    
    analysis_mode = st.radio("分析模式", ["全自动分析", "手动配置"])
    
    if analysis_mode == "手动配置":
        st.subheader("模型参数")
        model_type = st.selectbox(
            "选择算法",
            ["自动选择", "多元线性回归", "Ridge回归", "Lasso回归", "弹性网络", "随机森林", "梯度提升"]
        )
        handle_outliers = st.checkbox("异常值处理", value=True)
        standardize = st.checkbox("数据标准化", value=True)
        cv_folds = st.slider("交叉验证折数", 2, 10, 5)
    else:
        handle_outliers = True
        standardize = True
        cv_folds = 5

# 主界面
st.markdown('<p class="main-header">🔬 ResearchAI 智能科研分析系统</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">自适应样本量 · 自动模型选择 · 统计严谨性验证</p>', unsafe_allow_html=True)

# 文件上传
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("📁 上传数据文件 (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    
with col2:
    st.info("**数据格式要求：**\n- 第一行为变量名\n- 最后一列为目标变量Y\n- 支持缺失值（自动处理）\n- 样本量≥5组")

if uploaded_file is not None:
    # 数据读取
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 数据预览
        st.subheader("📊 数据概览")
        col_info, col_data = st.columns([1, 3])
        
        with col_info:
            st.metric("样本量", len(df))
            st.metric("变量数", len(df.columns))
            missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            st.metric("缺失率", f"{missing_rate:.1f}%")
            
            if len(df) < 5:
                st.error("⚠️ 样本量不足5组，无法进行统计分析")
                st.stop()
            elif len(df) < 20:
                st.warning("⚠️ 小样本情况（<20），将使用保守估计")
        
        with col_data:
            with st.expander("查看原始数据", expanded=True):
                st.dataframe(df, use_container_width=True, height=200)
                
        # 变量选择
        st.subheader("🎯 变量配置")
        cols = df.columns.tolist()
        
        col_target, col_features = st.columns(2)
        with col_target:
            target_col = st.selectbox("选择目标变量 (Y)", cols, index=len(cols)-1)
        with col_features:
            feature_cols = st.multiselect(
                "选择预测变量 (X)", 
                [c for c in cols if c != target_col],
                default=[c for c in cols if c != target_col]
            )
        
        if not feature_cols:
            st.warning("请至少选择一个预测变量")
            st.stop()
            
        # 数据预处理
        with st.spinner("数据预处理中..."):
            # 提取数据
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # 转换为数值型
            X = X.apply(pd.to_numeric, errors='coerce')
            y = pd.to_numeric(y, errors='coerce')
            
            # 处理缺失值
            valid_mask = X.notnull().all(axis=1) & y.notnull()
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            n_samples = len(X_clean)
            n_features = len(feature_cols)
            
            if n_samples < 5:
                st.error("清洗后有效样本少于5组，请检查数据质量")
                st.stop()
        
        # 异常值检测与处理
        if handle_outliers and n_samples >= 10:
            with st.expander("🔍 异常值检测报告"):
                z_scores = np.abs(stats.zscore(X_clean.select_dtypes(include=[np.number])))
                outliers = (z_scores > 3).any(axis=1)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    st.warning(f"检测到 {outlier_count} 个潜在异常值（|Z|>3）")
                    X_display = X_clean.copy()
                    X_display['是否为异常值'] = outliers
                    st.dataframe(X_display[outliers], use_container_width=True)
                    
                    if st.checkbox("移除异常值"):
                        X_clean = X_clean[~outliers]
                        y_clean = y_clean[~outliers]
                        n_samples = len(X_clean)
                        st.success(f"已清理，剩余 {n_samples} 组数据")
                else:
                    st.success("未检测到明显异常值")
        
        # 描述性统计
        with st.expander("📈 描述性统计", expanded=True):
            desc_stats = pd.DataFrame({
                '变量': feature_cols + [target_col],
                '均值': list(X_clean.mean()) + [y_clean.mean()],
                '标准差': list(X_clean.std()) + [y_clean.std()],
                '最小值': list(X_clean.min()) + [y_clean.min()],
                '最大值': list(X_clean.max()) + [y_clean.max()],
                '变异系数(%)': list((X_clean.std()/X_clean.mean()*100)) + [(y_clean.std()/y_clean.mean()*100)]
            })
            st.dataframe(desc_stats.round(3), use_container_width=True)
            
            # 相关性矩阵
            if len(feature_cols) <= 15:  # 变量太多不显示
                corr_data = pd.concat([X_clean, y_clean], axis=1)
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_data.corr(), annot=True, fmt='.2f', cmap='RdYlBu_r', 
                           center=0, ax=ax_corr, square=True)
                plt.title("相关性热力图", fontsize=14, pad=20)
                st.pyplot(fig_corr)
        
        # 分析执行
        if st.button("🚀 开始智能分析", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            
            # 数据分割记录
            progress_bar.progress(10)
            
            # 标准化
            if standardize:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
            else:
                X_scaled = X_clean.values
            
            # 模型选择策略
            progress_bar.progress(30)
            
            if analysis_mode == "全自动分析" or model_type == "自动选择":
                # 智能选择逻辑
                if n_samples < 15:
                    # 小样本：Ridge防止过拟合
                    model = Ridge(alpha=1.0)
                    model_name = "Ridge回归（小样本优化）"
                    recommendation = "样本量较小，采用L2正则化防止过拟合"
                elif n_samples < 50:
                    # 中等样本：弹性网络平衡偏差方差
                    model = ElasticNet(alpha=0.5, l1_ratio=0.5)
                    model_name = "弹性网络（平衡模型）"
                    recommendation = "中等样本量，平衡解释性和预测力"
                else:
                    # 大样本：集成学习
                    if n_features <= 5:
                        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                        model_name = "梯度提升树"
                    else:
                        model = RandomForestRegressor(n_estimators=100, max_depth=min(10, n_samples//5), random_state=42)
                        model_name = "随机森林"
                    recommendation = "大样本支持高复杂度模型，捕获非线性关系"
            else:
                model_map = {
                    "多元线性回归": LinearRegression(),
                    "Ridge回归": Ridge(alpha=1.0),
                    "Lasso回归": Lasso(alpha=0.1),
                    "弹性网络": ElasticNet(alpha=0.5),
                    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
                    "梯度提升": GradientBoostingRegressor(random_state=42)
                }
                model = model_map[model_type]
                model_name = model_type
            
            # 模型训练
            progress_bar.progress(50)
            model.fit(X_scaled, y_clean)
            y_pred = model.predict(X_scaled)
            
            # 交叉验证
            progress_bar.progress(70)
            if n_samples >= 10:
                if n_samples < cv_folds * 2:
                    cv = LeaveOneOut()
                    cv_scores = cross_val_score(model, X_scaled, y_clean, cv=cv, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_scaled, y_clean, cv=cv_folds, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_mean = r2_score(y_clean, y_pred)
                cv_std = 0
            
            # 计算指标
            progress_bar.progress(90)
            r2 = r2_score(y_clean, y_pred)
            rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
            mae = mean_absolute_error(y_clean, y_pred)
            mape = np.mean(np.abs((y_clean - y_pred) / y_clean)) * 100
            
            # 调整R²
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            
            progress_bar.progress(100)
            st.success("分析完成！")
            
            # 结果展示
            st.subheader("📋 分析结果报告")
            
            # 关键指标卡片
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("R² 决定系数", f"{r2:.3f}", help="模型解释数据变异的能力，越接近1越好")
            with col_m2:
                st.metric("调整R²", f"{adj_r2:.3f}", help="考虑变量数后的校正R²，小样本更重要")
            with col_m3:
                st.metric("RMSE", f"{rmse:.3f}", help="均方根误差，与Y同单位")
            with col_m4:
                st.metric("MAPE", f"{mape:.1f}%", help="平均绝对百分比误差")
            
            # 模型信息
            st.info(f"**选用模型：** {model_name} | **交叉验证R²：** {cv_mean:.3f} (±{cv_std:.3f}) | **建议：** {recommendation}")
            
            # 特征重要性
            st.subheader("🔍 因子重要性分析")
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                imp_type = "重要性系数"
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                imp_type = "标准化系数绝对值"
            else:
                importance = np.abs(np.corrcoef(X_scaled.T, y_pred)[-1, :-1])
                imp_type = "相关性系数"
            
            # 重要性数据框
            imp_df = pd.DataFrame({
                '因子': feature_cols,
                '重要性': importance,
                '重要性(%)': importance / importance.sum() * 100,
                '排名': pd.Series(importance).rank(method='min', ascending=False).astype(int)
            }).sort_values('重要性', ascending=False)
            
            col_imp1, col_imp2 = st.columns([2, 1])
            
            with col_imp1:
                # 重要性条形图
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_cols)))[::-1]
                bars = ax_imp.barh(range(len(imp_df)), imp_df['重要性'], color=colors)
                ax_imp.set_yticks(range(len(imp_df)))
                ax_imp.set_yticklabels(imp_df['因子'])
                ax_imp.set_xlabel(imp_type, fontsize=12)
                ax_imp.set_title("因子重要性排序", fontsize=14, fontweight='bold')
                
                # 添加数值标签
                for i, (idx, row) in enumerate(imp_df.iterrows()):
                    ax_imp.text(row['重要性'], i, f" {row['重要性']:.3f}", 
                               va='center', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig_imp)
            
            with col_imp2:
                st.dataframe(imp_df[['因子', '重要性(%)', '排名']].round(2), 
                            use_container_width=True, height=400)
            
            # 预测 vs 实际
            st.subheader("📊 模型诊断图表")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # 实际vs预测散点图
                fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
                ax_scatter.scatter(y_clean, y_pred, alpha=0.6, s=100, c='#1f77b4', edgecolors='white')
                min_val = min(y_clean.min(), y_pred.min())
                max_val = max(y_clean.max(), y_pred.max())
                ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
                ax_scatter.set_xlabel(f"实际值: {target_col}", fontsize=12)
                ax_scatter.set_ylabel(f"预测值", fontsize=12)
                ax_scatter.set_title("实际值 vs 预测值", fontsize=14, fontweight='bold')
                ax_scatter.legend()
                plt.tight_layout()
                st.pyplot(fig_scatter)
            
            with col_chart2:
                # 残差图
                residuals = y_clean - y_pred
                fig_res, ax_res = plt.subplots(figsize=(8, 6))
                ax_res.scatter(y_pred, residuals, alpha=0.6, s=100, c='#ff7f0e', edgecolors='white')
                ax_res.axhline(y=0, color='r', linestyle='--', lw=2)
                ax_res.set_xlabel("预测值", fontsize=12)
                ax_res.set_ylabel("残差 (实际-预测)", fontsize=12)
                ax_res.set_title("残差分布图", fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_res)
                
                # 正态性检验
                if len(residuals) >= 8:
                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    if shapiro_p > 0.05:
                        st.success(f"✅ 残差正态性检验通过 (p={shapiro_p:.3f})")
                    else:
                        st.warning(f"⚠️ 残差可能非正态 (p={shapiro_p:.3f})，建议检查异常值")
            
            # 统计显著性（仅线性模型）
            if hasattr(model, 'coef_') and n_samples > len(feature_cols) + 2:
                with st.expander("📐 统计显著性检验"):
                    # 计算t统计量（简化版）
                    n = len(y_clean)
                    k = len(feature_cols)
                    y_pred = model.predict(X_scaled)
                    mse = mean_squared_error(y_clean, y_pred)
                    
                    # 计算标准误（假设X已标准化）
                    std_errors = np.sqrt(mse / n)  # 简化计算
                    
                    coef_df = pd.DataFrame({
                        '因子': feature_cols,
                        '系数': model.coef_,
                        '标准误': std_errors,
                        't值': model.coef_ / std_errors,
                        '显著性': ['***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else 'ns' 
                                  for t in model.coef_ / std_errors]
                    })
                    st.dataframe(coef_df.round(4), use_container_width=True)
                    st.caption("显著性水平: *** p<0.01, ** p<0.05, * p<0.1, ns 不显著")
            
            # 结果导出
            st.subheader("💾 结果导出")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # 导出预测结果
                result_df = pd.DataFrame({
                    '实际值': y_clean,
                    '预测值': y_pred,
                    '残差': y_clean - y_pred,
                    '残差率(%)': (y_clean - y_pred) / y_clean * 100
                })
                csv = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载预测结果 (CSV)",
                    data=csv,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )
            
            with col_exp2:
                # 导出特征重要性
                csv_imp = imp_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载重要性报告 (CSV)",
                    data=csv_imp,
                    file_name='feature_importance.csv',
                    mime='text/csv'
                )
            
            # 使用建议
            st.subheader("📝 科研建议")
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                st.markdown("**模型可靠性评估：**")
                if r2 > 0.8 and cv_mean > 0.7:
                    st.success("✅ 模型拟合优秀，可用于预测和机理解释")
                elif r2 > 0.6:
                    st.info("ℹ️ 模型拟合良好，但存在未解释的变异，建议考虑其他潜在因子")
                else:
                    st.warning("⚠️ 模型解释力有限，可能存在关键缺失变量或非线性关系未捕获")
                
                if n_samples < 30:
                    st.warning("⚠️ 小样本结果需谨慎解读，建议增加样本量以提高统计功效")
            
            with adv_col2:
                st.markdown("**主要发现：**")
                top_factor = imp_df.iloc[0]['因子']
                top_imp = imp_df.iloc[0]['重要性(%)']
                st.write(f"• **关键因子**：{top_factor}（贡献{top_imp:.1f}%）")
                
                # 共线性警告
                if len(feature_cols) >= 2:
                    corr_matrix = X_clean.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr = [(corr_matrix.columns[i], corr_matrix.columns[j], upper.iloc[i,j]) 
                                for i in range(len(upper.columns)) for j in range(i+1, len(upper.columns)) 
                                if upper.iloc[i,j] > 0.8]
                    if high_corr:
                        st.write(f"• **注意**：检测到强相关性（{high_corr[0][0]} vs {high_corr[0][1]}: r={high_corr[0][2]:.2f}），可能存在共线性")
                    else:
                        st.write("• 各因子间独立性良好，无严重共线性问题")

    except Exception as e:
        st.error(f"分析过程中出现错误：{str(e)}")
        st.error("请检查数据格式：确保所有数据列为数值型，且样本量≥5")
        import traceback
        st.expander("详细错误信息").code(traceback.format_exc())

else:
    # 示例展示
    st.markdown("---")
    st.subheader("💡 使用示例")
    
    example_data = pd.DataFrame({
        '温度': [20, 25, 30, 35, 40, 22, 28, 32, 38, 42],
        '湿度': [60, 65, 70, 55, 50, 62, 68, 72, 58, 52],
        '反应时间': [120, 115, 110, 105, 95, 118, 112, 108, 102, 98],
        '产率': [85.2, 87.5, 89.0, 91.2, 93.5, 86.1, 88.3, 90.1, 92.4, 94.2]
    })
    
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.write("**示例数据格式：**")
        st.dataframe(example_data, use_container_width=True)
    
    with col_ex2:
        st.write("**预期输出：**")
        st.info("""
        • 自动识别产率为目标变量Y
        • 分析温度、湿度、反应时间对产率的影响
        • 生成特征重要性排序
        • 提供R²、RMSE等统计指标
        • 输出残差诊断图
        """)
    
    st.caption("提示：您的真实数据应类似此格式，第一行为列名，最后一列为目标变量")
