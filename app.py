
import streamlit as st
import pandas as pd
import requests
import os
from PIL import Image, ImageStat, ImageFilter
from io import BytesIO

# 전체 너비 레이아웃
st.set_page_config(layout="wide")

CACHE_DIR = "image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return BytesIO(response.content)
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def get_cached_score(
    item_url: str,
    weight_sharpness: float = 1.0,
    weight_center: float = 1.0,
    weight_noise: float = 1.0,
    weight_color: float = 1.0
):
    img_bytes = download_image(item_url)
    if not img_bytes:
        return None
    try:
        image = Image.open(img_bytes)
        return evaluate_clean_product_image(
            image,
            weight_sharpness=weight_sharpness,
            weight_center=weight_center,
            weight_noise=weight_noise,
            weight_color=weight_color
        )
    except:
        return None

def evaluate_clean_product_image(
    pil_img: Image.Image,
    weight_sharpness: float = 1.0,
    weight_center: float = 1.0,
    weight_noise: float = 1.0,
    weight_color: float = 1.0
):
    try:
        img_rgb = pil_img.convert("RGB")
        gray = pil_img.convert("L")

        # Sharpness (전체 이미지의 variance of Laplacian 대체: 이미지의 edge variance로 간주)
        edge_img = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = ImageStat.Stat(edge_img).var[0]

        # Center sharpness
        w, h = gray.size
        center_box = gray.crop((w//4, h//4, 3*w//4, 3*h//4))
        center_edge = center_box.filter(ImageFilter.FIND_EDGES)
        center_sharpness = ImageStat.Stat(center_edge).var[0]

        # Noise (standard deviation of grayscale)
        noise = ImageStat.Stat(gray).stddev[0]

        # Color diversity (R, G, B 각각의 stddev 합산)
        r, g, b = img_rgb.split()
        color_diversity = sum([
            ImageStat.Stat(r).stddev[0],
            ImageStat.Stat(g).stddev[0],
            ImageStat.Stat(b).stddev[0],
        ])

        raw_score = (
            weight_sharpness * sharpness +
            weight_center * center_sharpness -
            weight_noise * noise -
            weight_color * color_diversity
        )

        return {
            "sharpness": float(sharpness),
            "center_sharpness": float(center_sharpness),
            "noise": float(noise),
            "color_diversity": float(color_diversity),
            "score": float(raw_score)
        }
    except Exception:
        return None

PAGE_SIZE = 20
st.title("📦 Catalog Image Recommender")

st.sidebar.header("🔧 Recommendation Weights")
weight_sharpness = st.sidebar.slider("Sharpness", 0.0, 2.0, 2.0, 0.1)
weight_center = st.sidebar.slider("Center Sharpness", 0.0, 2.0, 1.0, 0.1)
weight_noise = st.sidebar.slider("Noise Penalty", 0.0, 2.0, 2.0, 0.1)
weight_color = st.sidebar.slider("Color Diversity Penalty", 0.0, 2.0, 1.0, 0.1)

uploaded_file = st.file_uploader("Upload a CSV", type="csv")
if uploaded_file and "uploaded_file" not in st.session_state:
    st.cache_data.clear()
    st.session_state.clear()
    st.session_state["uploaded_file"] = uploaded_file
    st.session_state["trigger_reload"] = True
    st.rerun()

if st.session_state.get("trigger_reload"):
    del st.session_state["trigger_reload"]

if "image_scores" not in st.session_state:
    st.session_state["image_scores"] = {}

if "uploaded_file" in st.session_state and "result_df" not in st.session_state:
    data = pd.read_csv(st.session_state["uploaded_file"], header=None,
                       names=["catalog_id", "main_img_url", "main_item_no", "item_no", "item_url"])
    st.session_state["data"] = data
    grouped = data.groupby("catalog_id", sort=False)  # 정렬 유지
    # grouped = data.groupby("catalog_id")
    result = []
    for catalog_id, group in grouped:
        if len(group) < 2:
            continue
        main_item = group.iloc[0]
        result.append({
            "catalog_id": catalog_id,
            "main_item_no": main_item["main_item_no"],
            "item_count": len(group),
            "main_item_url": main_item["main_img_url"],
            "selected_item_no": main_item["main_item_no"],
            "selected_url": main_item["main_img_url"]
        })
    st.session_state["result_df"] = pd.DataFrame(result)
    st.session_state["page"] = 0

if "result_df" in st.session_state:
    df = st.session_state["result_df"]
    total_pages = (len(df) - 1) // PAGE_SIZE + 1

    st.subheader("Catalogs")
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ Prev") and st.session_state["page"] > 0:
            st.session_state["page"] -= 1
    with col2:
        if st.button("➡️ Next") and st.session_state["page"] < total_pages - 1:
            st.session_state["page"] += 1

    start = st.session_state["page"] * PAGE_SIZE
    end = start + PAGE_SIZE
    paged_df = df.iloc[start:end]

    for idx in paged_df.index:
        row = df.loc[idx]
        catalog_id = row["catalog_id"]
        main_item_no = row["main_item_no"]
        selected_item_no = row["selected_item_no"]

        with st.expander(f"📁 Catalog {catalog_id}"):
            st.markdown(f"**Main Item No:** `{main_item_no}`  &nbsp;&nbsp;&nbsp; **Selected:** `{selected_item_no}`")
            group = st.session_state["data"][st.session_state["data"]["catalog_id"] == catalog_id]

            col_rec, col_edit = st.columns([1, 1.5])
            with col_rec:
                if st.button("🤖 Recommend", key=f"rec_{catalog_id}"):
                    scores = {}
                    best_item_no = ""
                    best_url = ""
                    best_score = -float("inf")

                    # 1차: 이미지 평가 점수 가져오기
                    for _, item_row in group.iterrows():
                        score = get_cached_score(
                            item_url=item_row["item_url"],
                            weight_sharpness=weight_sharpness,
                            weight_center=weight_center,
                            weight_noise=weight_noise,
                            weight_color=weight_color
                        )
                        if not score:
                            continue
                        scores[item_row["item_no"]] = score

                    # 2차: 점수가 없는 경우 중단
                    if not scores:
                        st.warning("❗ 유효한 이미지 평가 결과가 없습니다.")
                        continue

                    # (1) 루프 전에 모든 값을 수집
                    sharpness_vals = [s["sharpness"] for s in scores.values()]
                    center_vals = [s["center_sharpness"] for s in scores.values()]
                    noise_vals = [s["noise"] for s in scores.values()]
                    color_vals = [s["color_diversity"] for s in scores.values()]

                    # (2) 그 카탈로그 그룹 내에서 min/max 계산
                    sharp_min, sharp_max = min(sharpness_vals), max(sharpness_vals)
                    center_min, center_max = min(center_vals), max(center_vals)
                    noise_min, noise_max = min(noise_vals), max(noise_vals)
                    color_min, color_max = min(color_vals), max(color_vals)

                    total_weight = weight_sharpness + weight_center + weight_noise + weight_color

                    for item_no, s in scores.items():
                        sqrt_sharp = s["sharpness"]
                        sqrt_center = s["center_sharpness"]
                        sqrt_noise = s["noise"]
                        sqrt_color = s["color_diversity"]

                        # 정규화 (0~100 스케일)
                        # max 값 기준으로만 정규화
                        sharp_norm = 100 * (sqrt_sharp - sharp_min) / sharp_max if sharp_max > 0 else 0
                        center_norm = 100 * (sqrt_center - center_min) / center_max if center_max > 0 else 0
                        noise_norm = 100 * (sqrt_noise - noise_min) / noise_max if noise_max > 0 else 0
                        color_norm = 100 * (sqrt_color - color_min) / color_max if color_max > 0 else 0
                        # print(sharp_norm, center_norm, noise_norm, color_norm)

                        raw_score = (
                                weight_sharpness * sharp_norm +
                                weight_center * center_norm +
                                weight_noise * (100 - noise_norm) +
                                weight_color * (100 - color_norm)
                        )
                        s["score"] = raw_score

                        # 모든 항목의 raw_score 계산 후에...
                        raw_scores = [s["score"] for s in scores.values()]
                        raw_min = min(raw_scores)
                        raw_max = max(raw_scores)

                        # 항목별 정규화
                        normalized_score = s["score"] / 4
                        s["score_normalized"] = normalized_score

                        if normalized_score > best_score:
                            best_score = normalized_score
                            best_item_no = item_no
                            best_url = group[group["item_no"] == item_no]["item_url"].values[0]

                    # 4차: 추천 반영
                    if best_item_no:
                        df.at[idx, "selected_item_no"] = best_item_no
                        df.at[idx, "selected_url"] = best_url
                        st.session_state["image_scores"][catalog_id] = scores
                        st.rerun()

            with col_edit:
                if st.button("🛠️ 수정 반영", key=f"edit_{catalog_id}"):
                    st.info("🚧 해당 기능은 서비스 예정입니다.")

            # # 🤖 Recommend 버튼
            # if st.button("🤖 Recommend", key=f"rec_{catalog_id}"):



            items = list(group.iterrows())
            failed_items = []
            valid_items = []

            for _, item_row in items:
                item_no = item_row["item_no"]
                item_url = item_row["item_url"]
                img_bytes = download_image(item_url)
                if not img_bytes:
                    failed_items.append((item_no, item_url))
                    continue
                try:
                    image = Image.open(img_bytes)
                    valid_items.append((item_row, image))
                except:
                    failed_items.append((item_no, item_url))

            for i in range(0, len(valid_items), 8):
                cols = st.columns(8)
                for j, (item_row, image) in enumerate(valid_items[i:i+8]):
                    with cols[j]:
                        item_no = item_row["item_no"]
                        item_url = item_row["item_url"]
                        st.image(image, caption=f"{item_no}", use_container_width=True)

                        score_map = st.session_state["image_scores"].get(catalog_id, {})
                        if item_no in score_map and score_map[item_no] is not None:
                            score = score_map[item_no]
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: #f9f9fb;
                                    border: 1px solid #e0e0e0;
                                    border-radius: 10px;
                                    padding: 10px;
                                    margin-top: 8px;
                                    font-size: 13px;
                                    line-height: 1.5;
                                    box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                                    color: #333333;
                                ">
                                    <b>📊 Score:</b> {score['score_normalized']:.1f} / 100<br>
                                    <b>📈 Sharpness:</b> {score['sharpness']:.1f}<br>
                                    <b>🎯 Center Sharpness:</b> {score['center_sharpness']:.1f}<br>
                                    <b>🧯 Noise:</b> {score['noise']:.1f}<br>
                                    <b>🎨 Color Diversity:</b> {score['color_diversity']:.1f}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        is_selected = (row["selected_item_no"] == item_no)
                        label = "✅ Selected" if is_selected else "Select"
                        if st.button(label, key=f"select_{catalog_id}_{item_no}"):
                            df.at[idx, "selected_item_no"] = item_no
                            df.at[idx, "selected_url"] = item_url
                            st.rerun()

            if failed_items:
                st.warning("⚠️ 이미지 로딩 실패 항목:")
                for item_no, item_url in failed_items:
                    st.markdown(f"- ❌ `{item_no}` | `{item_url}`")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="selected_items.csv", mime="text/csv")
