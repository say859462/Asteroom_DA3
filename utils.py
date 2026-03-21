import matplotlib.pyplot as plt
import json
import csv
import itertools
import random
import os
import torch
from PIL import Image
from panorama_projector import PanoramaProjector
from pathlib import Path
import networkx as nx
import numpy as np
from tqdm import tqdm

# [新增邏輯] 強制使用非互動式後端，防止大規模圖片儲存時與主執行緒衝突導致的 tkinter RuntimeError
import matplotlib
matplotlib.use('Agg')

# Configure fonts to support Chinese characters in plots
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei',
                                   'SimHei', 'Arial Unicode MS']
# Fix the issue where the minus sign '-' is displayed as a square
plt.rcParams['axes.unicode_minus'] = False


def generate_perspective_imgs(pano_path, num_pers=6, fov=120.0, output_size=(512, 512), device="cuda"):
    """
    將單張 360 度全景圖 (Panorama) 均勻切分為多個指定視角的透視圖 (Perspective Images)。
    此函式利用批次運算 (Batch Processing) 最大化影像轉換效率，提供後續深度估計模型更精確的幾何視角。

    【參數說明】
    - pano_path (str | Path): 原始全景圖的檔案路徑。
    - num_pers (int): 預期切分的透視圖數量。預設為 6，代表每 60 度擷取一張畫面，確保視野充分重疊。
    - fov (float): 每個透視圖的視場角 (Field of View)，單位為度。預設 120.0 度可確保廣角覆蓋並維持合理的透視變形。
    - output_size (Tuple[int, int]): 輸出的透視圖解析度 (寬, 高)。預設為 (512, 512) 以平衡運算速度與空間特徵保留。
    - device (str): 執行張量運算的硬體設備，如 "cuda" 或 "cpu"。

    【回傳值】
    - List[Image.Image]: 包含轉換後透視圖的 PIL 影像列表，可直接作為 Depth Anything 3 模型的批次輸入。
    """
    # 1. 初始化投影器 (這會自動載入全景圖並快取網格)
    projector = PanoramaProjector(
        panorama_input=str(pano_path),
        output_size=output_size,
        fov=fov,
        device=device
    )

    # 2. 準備批次角度
    yaws = [i * (360.0 / num_pers) for i in range(num_pers)]
    pitches = [0.0] * num_pers

    # 3. 呼叫批次處理函數獲取所有透視視角的 Tensor [N, C, H, W]
    with torch.no_grad():
        batch_pers_tensor = projector.get_perspectives_batch(yaws, pitches)

    # 4. 後處理：Tensor -> Numpy -> uint8 -> PIL Image
    # 維度轉換: [N, C, H, W] -> [N, H, W, C]
    batch_pers_np = batch_pers_tensor.permute(0, 2, 3, 1).cpu().numpy()
    # 數值還原: 將 0~1 的浮點數轉回 0~255 的整數
    batch_pers_np = np.clip(batch_pers_np * 255, 0, 255).astype(np.uint8)
    # 轉化為 PIL 影像列表供 DA3 使用
    pers_imgs = [Image.fromarray(frame) for frame in batch_pers_np]

    return pers_imgs


def export_house_depth_visualizations(house_id, geometries, output_base_dir="./Depth_Visuals"):
    """
    接收已經推論完成的 geometries 數據，直接進行視覺化輸出，避免重複 Inference。

    【參數說明】
    - house_id (str): 房屋 ID。
    - geometries (dict): 格式為 { pano_rel_path: {"depth": [6,H,W], "orig": [6,H,W,3]} }。
    - output_base_dir (str): 視覺化圖片的輸出根目錄。
    """
    output_root = Path(output_base_dir) / house_id

    if not geometries:
        print(f"無可用的幾何數據，跳過深度圖導出。")
        return

    print(f"\n>>> 正在根據 CSV 列表輸出房屋 {house_id} 的深度預測圖 (直接利用緩存，無須二次推論)...")

    for pano_rel_path, data in tqdm(geometries.items(), desc="Saving Depth Maps"):
        pano_name = Path(pano_rel_path).stem
        pano_save_dir = output_root / pano_name
        pano_save_dir.mkdir(parents=True, exist_ok=True)

        orig_imgs = data["orig"]    # [6, H, W, 3] uint8
        depth_maps = data["depth"]  # [6, H, W] float32

        for i in range(len(orig_imgs)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # 左圖：原始透視圖
            axes[0].imshow(orig_imgs[i])
            axes[0].set_title(f"Original View {i}")
            axes[0].axis('off')

            # 右圖：深度圖 (正規化並套用 magma)
            d_min, d_max = depth_maps[i].min(), depth_maps[i].max()
            depth_norm = (depth_maps[i] - d_min) / (d_max - d_min + 1e-8)
            axes[1].imshow(depth_norm, cmap='magma')
            axes[1].set_title("Predicted Depth")
            axes[1].axis('off')

            plt.tight_layout()
            save_path = pano_save_dir / f"view_{i}.jpg"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f">>> 深度視覺化導出完成：{output_root}")


def save_visualization(img_A, img_B, pts_A, pts_B, save_path, title=""):
    """
    繪製並儲存視覺化結果的輔助函式 (引入彩色 Colormap 以實現 1-to-1 對應)。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 使用彩虹漸層色譜，讓每一個點都有獨一無二的顏色
    cmap = plt.cm.rainbow

    axes[0].imshow(img_A)
    axes[0].set_title("Source (Query Points)")
    if len(pts_A) > 0:
        # 產生 0 ~ N 的索引陣列作為顏色的映射基準
        colors_A = np.arange(len(pts_A))
        axes[0].scatter(pts_A[:, 0], pts_A[:, 1], c=colors_A, cmap=cmap,
                        s=20, alpha=0.9, edgecolors='black', linewidths=0.5)
    axes[0].axis('off')

    axes[1].imshow(img_B)
    axes[1].set_title("Target (Projected Points)")
    if len(pts_B) > 0:
        # 因為成功匹配時 pts_B 與 pts_A 的長度與順序完全一致，相同的索引會對應到相同的顏色
        colors_B = np.arange(len(pts_B))
        axes[1].scatter(pts_B[:, 0], pts_B[:, 1], c=colors_B, cmap=cmap,
                        s=20, alpha=0.9, edgecolors='black', linewidths=0.5)
    axes[1].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_houses_to_individual_csv(dataset_root_path, output_dir_path, negative_ratio=1.0):
    """
    Recursively scans for *_HOTSPOT.json files and generates an individual .csv for each house.
    """
    dataset_dir = Path(dataset_root_path)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    hotspot_files = list(dataset_dir.rglob("*_HOTSPOT.json"))
    houses_data = {}

    for file_path in hotspot_files:
        house_id = "Unknown_House"
        for part in file_path.parts:
            if "DollhouseTask_" in part:
                house_id = part
                break

        if house_id not in houses_data:
            houses_data[house_id] = {
                "all_images": set(), "connected_pairs": set()}

        image_dir = file_path.parent

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                hotspot_list = data.get("HOTSPOTOFROOM", [])
                for item in hotspot_list:
                    source_filename = item.get("IDName")
                    if not source_filename:
                        continue
                    source_full_path = image_dir / source_filename
                    source_rel_path = str(
                        source_full_path.relative_to(dataset_dir))
                    houses_data[house_id]["all_images"].add(source_rel_path)

                    target_filenames = item.get(
                        "ToIDName", {}).get("IDName", [])
                    for target_filename in target_filenames:
                        target_full_path = image_dir / target_filename
                        target_rel_path = str(
                            target_full_path.relative_to(dataset_dir))
                        houses_data[house_id]["all_images"].add(
                            target_rel_path)
                        pair = tuple(
                            sorted([source_rel_path, target_rel_path]))
                        houses_data[house_id]["connected_pairs"].add(pair)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    for house_id, data in houses_data.items():
        connected_pairs = data["connected_pairs"]
        all_images = data["all_images"]
        if not connected_pairs:
            continue

        csv_rows = [[imgA, imgB, 1] for imgA, imgB in connected_pairs]
        all_possible_pairs = set(
            itertools.combinations(sorted(list(all_images)), 2))
        unconnected_pairs = list(all_possible_pairs - connected_pairs)
        target_neg = int(len(connected_pairs) * negative_ratio)
        actual_neg = min(target_neg, len(unconnected_pairs))

        if actual_neg > 0:
            sampled_neg = random.sample(unconnected_pairs, actual_neg)
            for imgA, imgB in sampled_neg:
                csv_rows.append([imgA, imgB, 0])

        file_save_path = output_dir / f"{house_id}_connectivity.csv"
        with open(file_save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image_A', 'Image_B', 'Is_Connected'])
            writer.writerows(csv_rows)
        print(
            f"Saved: {file_save_path.name} (Pos: {len(connected_pairs)}, Neg: {actual_neg})")


def generate_house_graphs(dataset_root_path, output_dir_path):
    """
    Recursively scans for house directories, combines room names from relation.json, 
    builds an undirected connectivity graph, and saves it as an image.
    """
    dataset_dir = Path(dataset_root_path)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    houses = {}
    all_hotspots = list(dataset_dir.rglob("*_HOTSPOT.json"))

    for hp_file in all_hotspots:
        house_id = next(
            (p for p in hp_file.parts if "DollhouseTask_" in p), "Unknown")
        if house_id not in houses:
            houses[house_id] = {"hotspots": [], "relation": None}
        houses[house_id]["hotspots"].append(hp_file)
        rel_file = hp_file.parent / "relation.json"
        if rel_file.exists():
            houses[house_id]["relation"] = rel_file

    for house_id, files in houses.items():
        G = nx.Graph()
        name_map = {}
        if files["relation"]:
            with open(files["relation"], 'r', encoding='utf-8') as f:
                try:
                    rel_data = json.load(f)
                    for pano in rel_data.get("panos", []):
                        img_id = pano.get("id")
                        room_name = pano.get("name", "Unknown Room")
                        display_label = f"{room_name}\n({img_id[-4:]})"
                        name_map[img_id] = display_label
                        name_map[f"{img_id}.jpg"] = display_label
                except Exception as e:
                    print(f"Error reading relation.json for {house_id}: {e}")

        for hp_path in files["hotspots"]:
            with open(hp_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for item in data.get("HOTSPOTOFROOM", []):
                        u_id = item.get("IDName")
                        u_label = name_map.get(u_id, u_id)
                        G.add_node(u_label)
                        targets = item.get("ToIDName", {}).get("IDName", [])
                        for v_id in targets:
                            v_label = name_map.get(v_id, v_id)
                            G.add_edge(u_label, v_label)
                except Exception as e:
                    print(f"Skipping corrupted hotspot file {hp_path}: {e}")

        if G.number_of_nodes() > 0:
            plt.figure(figsize=(20, 12))
            pos = nx.spring_layout(G, k=1.5, iterations=100)
            nx.draw_networkx_edges(
                G, pos, width=1.2, edge_color='#D3D3D3', alpha=0.5)
            nx.draw_networkx_nodes(
                G, pos, node_size=3000, node_color='skyblue', alpha=0.9)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_family='Microsoft JhengHei',
                                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2', alpha=0.7))
            plt.title(f"Space Topology: {house_id}", fontsize=18, pad=20)
            plt.axis('off')
            save_path = output_dir / f"{house_id}_topology.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Topology graph generated: {save_path.name}")


def export_house_topology_json(dataset_root_path, output_dir_path):
    """
    Scans for house directories, builds the topology data by combining 
    relation.json and HOTSPOT.json, and exports the graph to a .json file.
    """
    dataset_dir = Path(dataset_root_path)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    houses = {}
    all_hotspots = list(dataset_dir.rglob("*_HOTSPOT.json"))

    for hp_file in all_hotspots:
        house_id = next(
            (p for p in hp_file.parts if "DollhouseTask_" in p), "Unknown")
        if house_id not in houses:
            houses[house_id] = {"hotspots": [], "relation": None}
        houses[house_id]["hotspots"].append(hp_file)
        rel_file = hp_file.parent / "relation.json"
        if rel_file.exists():
            houses[house_id]["relation"] = rel_file

    for house_id, files in houses.items():
        nodes = {}
        edges = set()
        if files["relation"]:
            with open(files["relation"], 'r', encoding='utf-8') as f:
                try:
                    rel_data = json.load(f)
                    for pano in rel_data.get("panos", []):
                        img_id = pano.get("id")
                        room_name = pano.get("name", "Unknown Room")
                        nodes[img_id] = {
                            "id": img_id, "room_name": room_name, "label": f"{room_name} ({img_id[-4:]})"}
                except Exception as e:
                    print(f"Error reading relation.json for {house_id}: {e}")

        for hp_path in files["hotspots"]:
            with open(hp_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for item in data.get("HOTSPOTOFROOM", []):
                        u_id = item.get("IDName").replace(".jpg", "")
                        if u_id not in nodes:
                            nodes[u_id] = {
                                "id": u_id, "room_name": "Unknown", "label": u_id}
                        targets = item.get("ToIDName", {}).get("IDName", [])
                        for v_full_name in targets:
                            v_id = v_full_name.replace(".jpg", "")
                            if v_id not in nodes:
                                nodes[v_id] = {
                                    "id": v_id, "room_name": "Unknown", "label": v_id}
                            edges.add(tuple(sorted([u_id, v_id])))
                except Exception as e:
                    print(f"Skipping corrupted hotspot file {hp_path}: {e}")

        if edges:
            topology_data = {
                "house_id": house_id,
                "nodes": list(nodes.values()),
                "edges": [{"source": e[0], "target": e[1]} for e in list(edges)]
            }
            save_path = output_dir / f"{house_id}_topology.json"
            with open(save_path, 'w', encoding='utf-8') as out_f:
                json.dump(topology_data, out_f, indent=4, ensure_ascii=False)
            print(f"Successfully exported topology data: {save_path.name}")


if __name__ == "__main__":
    # Example execution
    # process_houses_to_individual_csv('./Dataset', './Dataset/Metadatas', negative_ratio=0.5)
    # generate_house_graphs('./Dataset', './Dataset/Topology_Graphs')
    # export_house_topology_json('./Dataset', './Dataset/Topology_JSONs')
    
    pass
