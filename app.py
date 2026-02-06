import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math

# ------------------
# 設定
# ------------------
st.set_page_config(page_title="Sense Matching App", layout="centered")
st.title("🎨 Sense Palette")
st.caption("Painting your taste from a photo")

# ------------------
# モデル読み込み
# ------------------
reg = joblib.load("sense_app/sense_regressor.joblib")

resnet = models.resnet18(pretrained=True)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------
# データ読み込み
# ------------------
df = pd.read_csv("sense_app/items.csv")

sense_labels = ["brightness", "contrast", "body", "sharpness", "aftertaste"]

# ------------------
# 画像アップロード
# ------------------
uploaded_file = st.file_uploader(
    "写真をアップロードしてください",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="アップロード画像", use_container_width=True)

    # ------------------
    # 特徴量抽出
    # ------------------
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = resnet(x).numpy()

    # ------------------
    # 感性スコア予測
    # ------------------
    sense_vec = reg.predict(feat)[0]

    st.subheader("🧠 推定された感性スコア")
    for label, val in zip(sense_labels, sense_vec):
        st.write(f"{label}: {val:.2f}")

    # ------------------
    # 写真単体 レーダーチャート
    # ------------------
    st.subheader("📊 感性レーダーチャート（写真）")

    values = np.append(sense_vec, sense_vec[0])
    angles = np.linspace(0, 2 * np.pi, len(sense_labels), endpoint=False)
    angles = np.append(angles, angles[0])

    fig1 = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / math.pi, sense_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Photo")

    st.pyplot(fig1)
    plt.close(fig1)

    # ------------------
    # 類似度計算
    # ------------------
    item_vectors = df[sense_labels].values
    sims = cosine_similarity([sense_vec], item_vectors)[0]
    df["similarity"] = sims

    # ------------------
    # レコメンド（上位3件）
    # ------------------
    st.subheader("☕ おすすめコーヒー（TOP 3）")
    coffee_top3 = (
        df[df["item_type"] == "coffee"]
        .sort_values("similarity", ascending=False)
        .head(3)
    )
    st.table(coffee_top3[["item_name", "similarity"]])

    st.subheader("🍰 おすすめフード（TOP 3）")
    food_top3 = (
        df[df["item_type"] == "food"]
        .sort_values("similarity", ascending=False)
        .head(3)
    )
    st.table(food_top3[["item_name", "similarity"]])

    # ------------------
    # 重ね合わせレーダーチャート（TOP1のみ）
    # ------------------
    st.subheader("📊 感性レーダーチャート（写真 × コーヒー × フード）")

    coffee_top1 = coffee_top3.iloc[0][sense_labels].values
    food_top1 = food_top3.iloc[0][sense_labels].values

    def close_circle(x):
        return np.append(x, x[0])

    fig2 = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, close_circle(sense_vec), label="Photo", linewidth=2)
    ax.plot(angles, close_circle(coffee_top1), label="Coffee (Top1)", linewidth=2)
    ax.plot(angles, close_circle(food_top1), label="Food (Top1)", linewidth=2)

    ax.fill(angles, close_circle(sense_vec), alpha=0.15)
    ax.fill(angles, close_circle(coffee_top1), alpha=0.15)
    ax.fill(angles, close_circle(food_top1), alpha=0.15)

    ax.set_thetagrids(angles[:-1] * 180 / math.pi, sense_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Sense Radar Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig2)
    plt.close(fig2)
