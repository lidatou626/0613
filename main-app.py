import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from snownlp import SnowNLP

# 设置页面配置
st.set_page_config(
    page_title="5A级景区推荐系统",
    page_icon="🏞️",
    layout="wide"
)

# 页面标题和介绍
st.title("全国5A级景区推荐系统")
st.markdown("基于用户评论的情感分析和内容相似度的景区推荐")

# 初始化状态变量
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False


# 数据加载和预处理函数
@st.cache_data
def load_and_preprocess_data():
    try:
        # 加载数据
        excel_file = pd.ExcelFile('全国5A级景区.xlsx')
        df_scenic = excel_file.parse('5A')
        df_comment = pd.read_excel('景区评论数据集.xlsx')

        # 数据预处理
        df_scenic = df_scenic.dropna(subset=['dth_title'])
        df_comment = df_comment.dropna(subset=['景区名称'])
        merged_df = pd.merge(df_scenic, df_comment, left_on='dth_title', right_on='景区名称', how='outer')

        # 提取关键词和情感分析
        def extract_keywords(text):
            if isinstance(text, float) and pd.isna(text):
                return []
            return jieba.lcut(text)

        def get_sentiment_score(text):
            if isinstance(text, float) and pd.isna(text):
                return None
            return SnowNLP(text).sentiments

        merged_df['关键词'] = merged_df['评论内容'].apply(extract_keywords)
        merged_df['情感得分'] = merged_df['评论内容'].apply(get_sentiment_score)
        merged_df['映射评分'] = merged_df['情感得分'] * 4 + 1  # 映射到1-5分

        st.session_state.data_ready = True
        return merged_df

    except Exception as e:
        st.error(f"数据加载出错: {e}")
        return None


# 模型训练函数
@st.cache_resource
def train_model(df):
    try:
        # 创建Reader和Dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['用户ID', '景区名称', '映射评分']], reader)

        # 划分训练集和测试集
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        # 训练SVD模型
        start_time = time.time()
        algo = SVD(
            n_factors=50,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42
        )
        algo.fit(trainset)
        training_time = time.time() - start_time

        # 内容推荐 - 基于TF-IDF
        scenic_reviews = df.groupby('景区名称')['评论内容'].agg(lambda x: ' '.join(x)).reset_index()

        # 自定义分词器
        def chinese_tokenizer(text):
            return jieba.lcut(text)

        vectorizer = TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            stop_words=['的', '了', '和', '是', '在'],
            ngram_range=(1, 2),
            max_features=5000
        )

        tfidf_matrix = vectorizer.fit_transform(scenic_reviews['评论内容'])
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        scenic_index = {name: i for i, name in enumerate(scenic_reviews['景区名称'])}

        st.session_state.model_ready = True
        return algo, scenic_reviews, similarity_matrix, scenic_index, training_time

    except Exception as e:
        st.error(f"模型训练出错: {e}")
        return None, None, None, None, 0


# 内容推荐函数
def get_similar_scenics(scenic_name, scenic_reviews, similarity_matrix, scenic_index, top_n=5):
    if scenic_name not in scenic_index:
        return pd.DataFrame({"景区名称": [f"抱歉，景区 '{scenic_name}' 不在数据集中。"], "相似度得分": [0]})

    idx = scenic_index[scenic_name]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_scenics = sim_scores[1:top_n + 1]

    result = []
    for i, score in top_scenics:
        result.append({
            '景区名称': scenic_reviews.iloc[i]['景区名称'],
            '相似度得分': score
        })

    return pd.DataFrame(result)


# 混合推荐函数
def hybrid_recommend(user_id, scenic_name, df, algo, scenic_reviews, similarity_matrix, scenic_index, top_n=5,
                     content_weight=0.7, collab_weight=0.3):
    # 内容推荐
    content_rec = get_similar_scenics(scenic_name, scenic_reviews, similarity_matrix, scenic_index, top_n * 2)

    # 协同过滤推荐
    items = df['景区名称'].unique()
    user_items = df[df['用户ID'] == user_id]['景区名称'].tolist()
    unrated_items = [item for item in items if item not in user_items]

    predictions = []
    for item in unrated_items:
        pred = algo.predict(user_id, item)
        predictions.append((item, pred.est))

    collab_rec = pd.DataFrame(predictions, columns=['景区名称', '预测评分'])
    collab_rec = collab_rec.sort_values('预测评分', ascending=False).head(top_n * 2)

    # 混合推荐
    merged_rec = pd.merge(content_rec, collab_rec, on='景区名称', how='outer')
    merged_rec['综合得分'] = (content_weight * merged_rec['相似度得分'].fillna(0) +
                              collab_weight * merged_rec['预测评分'].fillna(0))

    return merged_rec.sort_values('综合得分', ascending=False).head(top_n)


# 主程序流程
with st.spinner("正在加载数据和训练模型..."):
    df = load_and_preprocess_data()

    if df is not None:
        algo, scenic_reviews, similarity_matrix, scenic_index, training_time = train_model(df)

        if algo is not None:
            st.success(f"数据加载和模型训练完成！训练耗时: {training_time:.2f}秒")

            # 显示数据统计信息
            st.subheader("数据统计信息")
            col1, col2 = st.columns(2)
            col1.metric("景区数量", len(df['景区名称'].unique()))
            col2.metric("用户评论数量", len(df))

            # 显示一些热门景区
            st.subheader("热门景区")
            popular_scenics = df.groupby('景区名称')['用户ID'].count().sort_values(ascending=False).head(5)
            st.bar_chart(popular_scenics)

            # 推荐系统界面
            st.subheader("景区推荐")
            recommendation_type = st.selectbox(
                "推荐类型",
                ["基于内容的推荐", "基于用户的推荐", "混合推荐"]
            )

            if recommendation_type == "基于内容的推荐":
                selected_scenic = st.selectbox(
                    "选择景区",
                    sorted(df['景区名称'].unique())
                )
                top_n = st.slider("推荐数量", 1, 10, 5)

                if st.button("获取推荐"):
                    with st.spinner("正在生成推荐..."):
                        recommendations = get_similar_scenics(
                            selected_scenic,
                            scenic_reviews,
                            similarity_matrix,
                            scenic_index,
                            top_n
                        )
                        st.write(f"与 **{selected_scenic}** 相似的景区:")
                        st.dataframe(recommendations.style.format({"相似度得分": "{:.2f}"}))

            elif recommendation_type == "基于用户的推荐":
                user_id = st.text_input("输入用户ID", "11111")
                top_n = st.slider("推荐数量", 1, 10, 5)

                if st.button("获取推荐"):
                    with st.spinner("正在生成推荐..."):
                        # 获取用户已经评价过的景区
                        user_scenics = df[df['用户ID'] == user_id]['景区名称'].unique()

                        if len(user_scenics) == 0:
                            st.warning(f"用户 {user_id} 没有评价记录，无法进行个性化推荐。")
                        else:
                            st.write(f"用户 {user_id} 已评价的景区:")
                            st.write(", ".join(user_scenics))

                            # 生成推荐
                            items = df['景区名称'].unique()
                            user_items = df[df['用户ID'] == user_id]['景区名称'].tolist()
                            unrated_items = [item for item in items if item not in user_items]

                            predictions = []
                            for item in unrated_items:
                                pred = algo.predict(user_id, item)
                                predictions.append((item, pred.est))

                            recommendations = pd.DataFrame(predictions, columns=['景区名称', '预测评分'])
                            recommendations = recommendations.sort_values('预测评分', ascending=False).head(top_n)

                            st.write(f"为用户 {user_id} 推荐的景区:")
                            st.dataframe(recommendations.style.format({"预测评分": "{:.2f}"}))

            else:  # 混合推荐
                user_id = st.text_input("输入用户ID", "11111")
                selected_scenic = st.selectbox(
                    "选择参考景区",
                    sorted(df['景区名称'].unique())
                )
                top_n = st.slider("推荐数量", 1, 10, 5)
                content_weight = st.slider("内容推荐权重", 0.0, 1.0, 0.7, 0.1)
                collab_weight = st.slider("协同过滤权重", 0.0, 1.0, 0.3, 0.1)

                if st.button("获取推荐"):
                    with st.spinner("正在生成混合推荐..."):
                        recommendations = hybrid_recommend(
                            user_id,
                            selected_scenic,
                            df,
                            algo,
                            scenic_reviews,
                            similarity_matrix,
                            scenic_index,
                            top_n,
                            content_weight,
                            collab_weight
                        )

                        st.write(f"混合推荐结果 (用户 {user_id} 可能喜欢的类似 **{selected_scenic}** 的景区):")
                        st.dataframe(recommendations[['景区名称', '相似度得分', '预测评分', '综合得分']].style.format({
                            "相似度得分": "{:.2f}",
                            "预测评分": "{:.2f}",
                            "综合得分": "{:.2f}"
                        }))

            # 景区详情查看
            st.subheader("景区详情")
            selected_scenic_detail = st.selectbox(
                "选择景区查看详情",
                sorted(df['景区名称'].unique())
            )

            if st.button("查看详情"):
                scenic_details = df[df['景区名称'] == selected_scenic_detail].iloc[0]
                st.write(f"### {scenic_details['景区名称']}")

                col1, col2 = st.columns(2)
                col1.metric("平均评分", f"{scenic_details['映射评分']:.1f}/5.0")
                col1.metric("评论数量", len(df[df['景区名称'] == selected_scenic_detail]))

                st.write("**景区简介**")
                st.write(scenic_details.get('景区简介', '暂无简介'))

                st.write("**热门评论**")
                comments = df[df['景区名称'] == selected_scenic_detail]['评论内容'].dropna().head(3)
                for i, comment in enumerate(comments):
                    st.markdown(f"> {comment}")

                st.write("**关键词**")
                keywords = scenic_details.get('关键词', [])
                if keywords:
                    st.markdown(" ".join([f"#{word}" for word in keywords[:10]]))
else:
st.error("数据加载或模型训练失败，请检查数据文件和依赖库。")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
