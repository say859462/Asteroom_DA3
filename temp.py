from utils import (save_visualization, generate_perspective_imgs,
                   export_house_depth_visualizations)
from panorama_projector import PanoramaProjector
from depth_anything_3.api import DepthAnything3
import csv
import time
import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import gc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# 嘗試載入 LightGlue 模組 (Phase 2 局部特徵匹配)
try:
    from LightGlue.lightglue import LightGlue, SuperPoint
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False

# 為了解決 Blackwell (12.0) 硬體不支援舊版 xformers 的問題
# 在載入任何 torch 模組前強制禁用 xformers，切換至原生 PyTorch SDPA 運算
os.environ["XFORMERS_DISABLED"] = "1"


#  獨立的相似度 .CSV 輸出函數
def export_similarity_csv(sim_csv_rows, output_dir, house_id):
    """
    將所有配對的語義相似度與真實標籤輸出為 CSV 檔案，供後續數據分析與 Threshold 決策使用。
    若不需要輸出，可在主程式中將此函數的呼叫註解掉。
    """
    if not sim_csv_rows:
        return

    output_path = Path(output_dir) / f"{house_id}_similarity.csv"
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 寫入表頭
        writer.writerow(
            ['Image_A', 'Image_B', 'Ground_Truth', 'Max_Similarity'])
        writer.writerows(sim_csv_rows)

    print(f">>> 相似度分析數據已導出至：{output_path}")


# [性能優化版] 提取單張全景圖的所有視角特徵 (使用外部 DINOv2)
def extract_pano_features(model_semantic, imgs, device):
    """
    預先提取單張全景圖 (6 個透視圖) 的 CLS Tokens 並進行歸一化。
    [修改邏輯] 使用獨立載入的外部 DINOv2 模型，不依賴 DA3 內部結構以提升穩定性。
    """
    # DINOv2 標準影像預處理 (224x224 是官方最穩定的輸入規格)
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        # 將 6 張視角圖轉換為 Tensor [6, 3, 224, 224]
        tensor = torch.stack([transform(img) for img in imgs]).to(device)

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            # [修正邏輯] 直接呼叫外部 DINOv2 模型的 forward 獲取 CLS Token
            cls_tokens = model_semantic(tensor)  # [6, D]

            # 確保輸出維度為 [6, D]
            if cls_tokens.dim() == 3 and cls_tokens.shape[0] == 1:
                cls_tokens = cls_tokens.squeeze(0)

            # 預先進行 L2 正規化，後續直接做矩陣相乘即為 Cosine Similarity
            cls_tokens = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)

    return cls_tokens


# [極速版] 僅進行純矩陣運算的語義篩選
def check_semantic_similarity_fast(feat_A, feat_B, threshold=0.65):
    """
    Phase 1: 利用預算的特徵進行極速篩選。
    此函數不涉及神經網路運算，純粹為 CPU/GPU 矩陣相乘。
    """
    # 計算 6x6 個視角的交叉相似度矩陣 (矩陣內積即 Cosine Similarity)
    sim_matrix = torch.matmul(feat_A, feat_B.T)
    max_sim = sim_matrix.max().item()
    is_passed = max_sim >= threshold
    return is_passed, max_sim


# [新增模組] 提取局部特徵 (使用 SuperPoint)
def extract_local_features(extractor, imgs, device):
    """
    Phase 2: 預先提取 6 個透視圖的 SuperPoint 角點特徵。
    SuperPoint 要求輸入為灰階影像 (1, 1, H, W)。
    """
    transform = T.Compose([
        T.Resize((512, 512)),
        T.Grayscale(num_output_channels=1),  # LightGlue 官方預設處理為單通道灰階
        T.ToTensor(),
    ])

    feats_list = []
    with torch.no_grad():
        for img in imgs:
            img_tensor = transform(img).unsqueeze(0).to(device)
            feats = extractor.extract(img_tensor)
            feats_list.append(feats)

    return feats_list


# [新增模組] 局部幾何特徵匹配 (使用 LightGlue)
def check_lightglue_match(matcher, feats_A, feats_B, threshold=15):
    """
    Phase 2: 執行 LightGlue 稀疏特徵點匹配。
    只要 A 的任何一個視角與 B 的任何一個視角能產生 >= threshold 個 Inlier 匹配點，即放行。
    """
    max_matches = 0
    with torch.no_grad():
        for feat_a in feats_A:
            for feat_b in feats_B:
                # LightGlue 匹配
                matches01 = matcher({"image0": feat_a, "image1": feat_b})
                matches_tensor = matches01['matches']

                # 獲取配對點數量 (相容不同版本的 LightGlue 輸出張量維度)
                num_matches = matches_tensor.shape[-2]

                if num_matches > max_matches:
                    max_matches = num_matches

                # 提早停止 (Early Stopping)：只要找到一組達標，直接放行以節省算力
                if max_matches >= threshold:
                    return True, max_matches

    return False, max_matches


#  獨立的深度一致性驗證函數
def check_depth_consistency(u_proj, v_proj, z_proj, target_depth_map, tolerance=0.15):
    """
    驗證投影點的深度一致性 (Occlusion Check)。
    藉由比對投影點在目標相機座標系下的深度 (z_proj) 與目標影像在該像素位置預測的深度 (target_depth_map)，
    來判斷該點是否被牆壁或障礙物遮擋。

    【參數說明】
    - u_proj (np.ndarray): 投影至目標影像的 x 座標陣列
    - v_proj (np.ndarray): 投影至目標影像的 y 座標陣列
    - z_proj (np.ndarray): 投影點在目標相機座標系下的 z 深度陣列 (投影深度)
    - target_depth_map (np.ndarray): 目標影像的深度預測圖 (H, W)
    - tolerance (float): 深度容差比例，預設 0.15 表示允許 15% 的估計誤差

    【回傳值】
    - not_occluded_mask (np.ndarray): 布林陣列，True 代表未被遮擋 (有效連通)
    """
    H, W = target_depth_map.shape

    # 確保座標在影像範圍內，避免 index out of bounds (在此之前通常已做過 in_fov 檢查，但保險起見再做一次 clip)
    u_int = np.clip(np.round(u_proj).astype(int), 0, W - 1)
    v_int = np.clip(np.round(v_proj).astype(int), 0, H - 1)

    # 取得目標影像在投影位置的「預測深度」
    d_pred = target_depth_map[v_int, u_int]

    # 判斷是否遮擋：
    # 如果投影過來的深度 z_proj 小於或等於目標預測深度 (加上容差)，代表點在目標視角的可見表面上或前方 -> 未遮擋
    # 若 z_proj 顯著大於 d_pred，代表前方有牆壁遮擋
    not_occluded_mask = z_proj <= d_pred * (1 + tolerance)

    return not_occluded_mask


# [新增輔助函式] 將原來的 evaluate_connectivity 核心邏輯封裝為單向驗證
def _project_and_verify_single_direction(prediction_pose, pers_idx_A, pers_idx_B, query_grid_size=(20, 20), conf_threshold=0.5, depth_mode="relative_only", prediction_metric=None, occlusion_tolerance=0.20):
    """
    原 evaluate_connectivity 的核心邏輯，執行單向投影 (A -> B)，並回傳「視角重疊比例」。
    """
    total_query_points = 0
    visible_in_B_points = 0

    #  儲存視覺化用的最佳視角配對資訊
    best_vis = {
        'count': -1,
        'idx_A': pers_idx_A[0],
        'idx_B': pers_idx_B[0],
        'pts_A': np.empty((0, 2)),
        'pts_B': np.empty((0, 2))
    }
    max_valid_A = -1

    # 遍歷房間 A 的 6 個透視圖視角
    for i in pers_idx_A:

        depth_A = prediction_pose.depth[i]         # [H, W]
        conf_A = prediction_pose.conf[i]           # [H, W] 提取信心度矩陣

        # 使用 np.copy 避免修改到原始 prediction 的記憶體，保護矩陣不被二次污染
        ext_A = np.copy(prediction_pose.extrinsics[i])       # [3, 4]
        int_A = prediction_pose.intrinsics[i]      # [3, 3]

        H, W = depth_A.shape

        # 1. 均勻撒點 (Query Points)
        # Spread query_grid_size[0] points between [0,W-1]
        x_coords = np.linspace(0, W - 1, query_grid_size[0], dtype=int)
        y_coords = np.linspace(0, H - 1, query_grid_size[1], dtype=int)

        # 組合成 query points 的 x , y 座標
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)

        u, v = grid_x.flatten(), grid_y.flatten()
        # Image coord (v,u)
        d = depth_A[v, u]
        c = conf_A[v, u]

        # 根據 depth_mode 進行幾何與尺度切換 ，只有需要metric depth時才會啟動，讓3D資訊可以對齊精確單位 ---
        scale_factor = 1.0  # 預設不縮放
        # NOTE: 有可能當前的metric轉換不是最佳解。
        if depth_mode == "hybrid" and prediction_metric is not None:
            depth_M = prediction_metric.depth[i]

            # DA3METRIC 需要透過焦距正規化轉換為公尺
            focal_length = (int_A[0, 0] + int_A[1, 1]) / 2.0
            depth_M = (focal_length * depth_M) / 300.0

            # 找出有效點以計算縮放係數 (Scale Factor)
            valid_scale_mask = (depth_A > 0) & (depth_M > 0)
            if np.sum(valid_scale_mask) > 0:
                scale_factor = np.median(
                    depth_M[valid_scale_mask] / depth_A[valid_scale_mask])
                # 將相機平移向量 t 放大到 Metric 尺度
                ext_A[:, 3] = ext_A[:, 3] * scale_factor
                # 將後續投影運算的深度替換為真實公尺
                d = depth_M[v, u]
        # --------------------------------------------------------

        # 結合深度有效性與模型信心度進行雙重過濾
        valid_mask = (d > 0) & (c > conf_threshold)
        u, v, d = u[valid_mask], v[valid_mask], d[valid_mask]
        total_query_points += len(u)

        if len(u) == 0:
            continue

        # 如果這是目前最多有效深度的 A 視角，且尚未找到投影成功的 B，先用它當作預設視覺化圖
        if len(u) > max_valid_A:
            max_valid_A = len(u)
            if best_vis['count'] <= 0:
                best_vis['idx_A'] = i
                best_vis['pts_A'] = np.stack([u, v], axis=1)

        # 2. 反投影至相機座標系 A
        # 運算邏輯: P_cam_A = d * inv(K_A) * [u, v, 1]^T
        # PS: [u,v,1] is the direction to the 3D coord , it's not a specific coord
        inv_K_A = np.linalg.inv(int_A)  # 內參反矩陣
        p2d_homo = np.stack([u, v, np.ones_like(u)], axis=0)  # 齊次座標系(擴張一維度)
        p_cam_A = inv_K_A @ p2d_homo * d

        # 3. 轉換至全局世界座標系
        # 運算邏輯: P_world = inv(R_A) * (P_cam_A - t_A)
        R_A, t_A = ext_A[:, :3], ext_A[:, 3:]  # Rotaion , Translation
        p_world = np.linalg.inv(R_A) @ (p_cam_A - t_A)

        # 4. 投影至房間 B 的所有 6 個透視圖視角
        is_visible_anywhere = np.zeros(len(u), dtype=bool)

        for j in pers_idx_B:
            # 必須使用 np.copy 以免改到原始資料，並且在 hybrid 模式下同步放大 B 的平移向量
            ext_B = np.copy(prediction_pose.extrinsics[j])
            if depth_mode == "hybrid":
                ext_B[:, 3] = ext_B[:, 3] * scale_factor

            int_B = prediction_pose.intrinsics[j]
            depth_B = prediction_pose.depth[j]  # 提取目標影像的深度圖供遮擋驗證使用
            R_B, t_B = ext_B[:, :3], ext_B[:, 3:]

            # 4.1 轉換至相機座標系 B
            # 運算邏輯: P_cam_B = R_B * P_world + t_B
            # 將A圖像的query points 世界座標，投影回圖像B
            p_cam_B = R_B @ p_world + t_B

            # 投影回B相機坐標系 可能會有點在相機後面，我們不考慮這些點
            z_B = p_cam_B[2, :]

            # 確保點在相機前方
            front_mask = z_B > 0

            # 4.2 投影至 2D 影像平面 B
            # 運算邏輯: [u_proj*z, v_proj*z, z]^T = K_B * P_cam_B
            # 齊次座標歸一化: u_B = (u_proj*z) / z, v_B = (v_proj*z) / z
            # 相機座標 -> 影像B平面座標
            p2d_B = int_B @ p_cam_B
            u_B = p2d_B[0, :] / (z_B + 1e-6)
            v_B = p2d_B[1, :] / (z_B + 1e-6)

            # 4.3 檢查投影點是否在視角範圍 (FOV) 內
            in_fov = (u_B >= 0) & (u_B < W) & (v_B >= 0) & (v_B < H)

            # 4.4 深度一致性驗證 (Occlusion Check)
            not_occluded_mask = check_depth_consistency(
                u_B, v_B, z_B, depth_B, tolerance=occlusion_tolerance
            )

            # 擷取並儲存成功投影且「未被遮擋」的點，以供視覺化
            valid_B_mask = front_mask & in_fov & not_occluded_mask

            count_B = np.sum(valid_B_mask)
            if count_B > best_vis['count']:
                best_vis['count'] = count_B
                best_vis['idx_A'] = i
                best_vis['idx_B'] = j
                best_vis['pts_A'] = np.stack(
                    [u[valid_B_mask], v[valid_B_mask]], axis=1)
                best_vis['pts_B'] = np.stack(
                    [u_B[valid_B_mask], v_B[valid_B_mask]], axis=1)

            # 圖像B的某個透視圖可以看到query points，且沒有被牆壁遮住
            is_visible_anywhere |= valid_B_mask

        visible_in_B_points += np.sum(is_visible_anywhere)

    if total_query_points == 0:
        return 0.0, best_vis

    # 計算重疊比例並回傳
    overlap_ratio = visible_in_B_points / total_query_points
    return overlap_ratio, best_vis


# TODO: query points的撒點選擇也很重要，有效率的撒點可以提高效率。
# TODO: confidence 有沒有可能DA3都預測是很低的情況，導致沒有空間點可以匹配 ?
def evaluate_connectivity(prediction_pose, pers_idx_A, pers_idx_B, overlap_threshold=0.05, query_grid_size=(20, 20), conf_threshold=0.5, depth_mode="relative_only", prediction_metric=None, occlusion_tolerance=0.20, use_bidirectional=True):
    """
    透過幾何重投影 (Geometric Reprojection) 驗證兩個空間之間的視覺連通性。
    演算法會在來源空間 (A) 的深度圖上建立均勻網格 (Query Points)，結合 DA3 提供的信心度 (Confidence) 過濾不可靠的預測，
    將有效點反投影至 3D 世界座標系後，再次投影至目標空間 (B) 的所有透視視角中，藉由計算落在目標視野內的點數比例來判定空間是否相連。

    【參數說明】
    - prediction_pose (Prediction): 原為 prediction，DA3 模型輸出的全局預測結果，包含深度圖 (depth)、信心度 (conf)、內參 (intrinsics) 與外參 (extrinsics)。
    - pers_idx_A (List[int]): 來源空間 A 在全局預測結果中對應的多個透視圖索引。
    - pers_idx_B (List[int]): 目標空間 B 在全局預測結果中對應的多個透視圖索引。
    - overlap_threshold (float): 判定為連通的最低視野重疊率門檻。預設 0.05 代表至少有 5% 的點能在目標空間中被觀測到。
    - query_grid_size (Tuple[int, int]): 來源視圖上均勻撒點的網格維度 (X, Y)。預設 (20, 20) 可確保產生 400 個解析度獨立的查詢點，維持效能穩定。
    - conf_threshold (float): 深度預測的最低信心度門檻。預設為 0.5，用於剔除模糊或反光區域的不穩定深度值，以提升幾何重投影的精準度。
    - depth_mode (str): 選擇深度推論模式 ("relative_only", "hybrid")。
    - prediction_metric (Prediction): 僅在 "hybrid" 模式下提供，負責提供真實尺度的深度。
    - occlusion_tolerance (float): [新增參數] 深度驗證容差，預設已放寬至 0.20。
    - use_bidirectional (bool): [消融實驗超參數] 控制是否啟用雙向驗證 (A->B 及 B->A)。

    【回傳值】
    - Tuple[bool, dict]: 回傳 (是否連通, 視覺化所需的字典包)。
    """
    # 1. 執行 A -> B 投影
    overlap_A2B, best_vis_A2B = _project_and_verify_single_direction(
        prediction_pose, pers_idx_A, pers_idx_B, query_grid_size, conf_threshold, depth_mode, prediction_metric, occlusion_tolerance
    )

    if use_bidirectional:
        # 2. 執行 B -> A 投影 (反向驗證)
        overlap_B2A, _ = _project_and_verify_single_direction(
            prediction_pose, pers_idx_B, pers_idx_A, query_grid_size, conf_threshold, depth_mode, prediction_metric, occlusion_tolerance
        )
        # 3. 嚴格判定：兩個方向的視野重疊率都必須大於重疊門檻，才能判定為真連通
        is_connected = (overlap_A2B > overlap_threshold) and (
            overlap_B2A > overlap_threshold)
    else:
        # 單向判定 (退回舊版邏輯)
        is_connected = (overlap_A2B > overlap_threshold)

    return is_connected, best_vis_A2B


def evaluate_single_house(dataset_root, house_csv_path, model_pose, device="cuda", query_grid_size=(20, 20), conf_threshold=0.5, depth_mode="relative_only", model_metric=None, output_dir="./Plots", export_depth=False, depth_out_dir="./Depth_Visuals", semantic_threshold=0.3, use_semantic=True, use_bidirectional=True, use_lightglue=True, lightglue_threshold=15, **kwargs):
    """
    執行單一棟房屋評估 ，評估所有空間全景圖連通性。
    【架構變更】: 三級漏斗過濾架構 (3-Stage Funnel)
    Phase 1 (Global Semantic): DINOv2 全域相似度預篩選 (閾值極度放寬，擋掉最離譜配對)
    Phase 2 (Local Geometric): LightGlue 局部特徵匹配 (防禦白牆與 DA3 幾何幻覺)
    Phase 3 (3D Pose & Depth) : DA3 幾何投影與遮擋驗證

    【參數說明】
    ...
    - use_semantic (bool): [消融實驗] 是否啟用 Phase 1 (DINOv2)
    - use_lightglue (bool): [消融實驗] 是否啟用 Phase 2 (LightGlue)
    - use_bidirectional (bool): [消融實驗] 是否啟用雙向幾何驗證
    """
    start_time = time.time()  # 計算花費時間

    # 防呆機制：若使用者未安裝 lightglue 卻啟用了該功能，自動退回降級模式
    global LIGHTGLUE_AVAILABLE
    if use_lightglue and not LIGHTGLUE_AVAILABLE:
        print("\n[警告] 未偵測到 lightglue 套件！請先執行: pip install lightglue")
        print(">> 已自動為您停用 Phase 2 (LightGlue) 局部匹配機制...\n")
        use_lightglue = False

    dataset_dir = Path(dataset_root)
    csv_path = Path(house_csv_path)

    # 1. 讀取單棟房屋的 CSV 檔案
    rows = []
    unique_panos_dict = {}

    if not csv_path.exists():
        print(f"找不到指定的 metadata 檔案：{csv_path}")
        return 0.0, 0.0, 0.0, 0.0, 0, 0.0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            unique_panos_dict[row['Image_A']] = None
            unique_panos_dict[row['Image_B']] = None

    unique_panos = list(unique_panos_dict.keys())
    num_panos = len(unique_panos)
    house_id = csv_path.stem.replace("_connectivity", "")

    # 幾何與特徵緩存
    house_geometries = {}
    pano_imgs_cache = {}
    pano_features_cache = {}       # DINOv2 全域特徵
    pano_local_features_cache = {}  # SuperPoint 局部特徵

    sim_csv_rows = []

    # === [預先載入特徵提取模型] ===
    model_semantic, extractor = None, None

    if use_semantic:
        print(">>> 載入外部 DINOv2 模型進行全域語義快取...")
        model_semantic = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitl14').to(device).eval()

    if use_lightglue:
        print(">>> 載入外部 SuperPoint 模型進行局部角點特徵快取...")
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # 2. 預處理：統一生成透視圖並預算語義特徵 (O(N))
    print(f"\n--- 開始處理房屋：{house_id} ---")
    for pano_rel_path in tqdm(unique_panos, desc="Preprocessing All Panos"):
        pano_full_path = dataset_dir / pano_rel_path
        if pano_full_path.exists():
            # 生成透視圖 (FOV 90)
            imgs = generate_perspective_imgs(
                pano_full_path, fov=90, device=device)
            pano_imgs_cache[pano_rel_path] = imgs

            # 若啟用 Phase 1，進行全域特徵提取
            if use_semantic:
                pano_features_cache[pano_rel_path] = extract_pano_features(
                    model_semantic, imgs, device)

            # 若啟用 Phase 2，進行局部特徵提取
            if use_lightglue:
                pano_local_features_cache[pano_rel_path] = extract_local_features(
                    extractor, imgs, device)

    # 特徵快取完成，極致釋放外部模型資源以騰出顯存給 Phase 3 的 DA3 推論
    print(">>> 特徵提取完成，釋放 DINOv2 與 SuperPoint 顯存資源 (CPU offload + GC)...")
    if model_semantic is not None:
        model_semantic = model_semantic.to('cpu')
        del model_semantic
    if extractor is not None:
        extractor = extractor.to('cpu')
        del extractor

    gc.collect()           # 強制執行 Python 垃圾回收
    torch.cuda.empty_cache()  # 強制清空 CUDA 顯存碎片

    # 載入輕量級的 LightGlue 匹配器 (保留在顯存中供 Pair-wise 比對使用)
    matcher = None
    if use_lightglue:
        matcher = LightGlue(features='superpoint').eval().to(device)

    # 統計變數
    total_correct, total_samples = 0, 0
    true_positives, true_negatives = 0, 0
    false_positives, false_negatives = 0, 0
    total_gt_connected, total_gt_disconnected = 0, 0

    semantic_rejected_count = 0
    lightglue_rejected_count = 0

    # 3. 驗證連通性配對
    print("驗證連通性配對 (3-Stage Funnel Evaluation)...")
    for row in tqdm(rows, desc="Evaluating Pairs"):
        pA, pB = row['Image_A'], row['Image_B']
        ground_truth = int(row['Is_Connected'])

        if ground_truth == 1:
            total_gt_connected += 1
        else:
            total_gt_disconnected += 1

        is_connected_pred = 0
        is_passed = True
        rejected_phase = 0  # 記錄被哪一階段過濾掉

        max_sim = 1.0       # DINOv2 Similarity
        max_matches = 0     # LightGlue Match Count

        vis_data = {
            'count': -1, 'idx_A': 0, 'idx_B': 6,
            'pts_A': np.empty((0, 2)), 'pts_B': np.empty((0, 2))
        }

        vis_idx_A = 0
        vis_idx_B = 6

        imgs_A = pano_imgs_cache[pA]
        imgs_B = pano_imgs_cache[pB]
        vis_imgs_ref = imgs_A + imgs_B

        panoA_name = Path(pA).stem
        panoB_name = Path(pB).stem

        # ========================================================
        # Phase 1: DINOv2 全域語義篩選 (垃圾桶防線)
        # ========================================================
        if use_semantic:
            feat_A = pano_features_cache[pA]
            feat_B = pano_features_cache[pB]
            is_passed, max_sim = check_semantic_similarity_fast(
                feat_A, feat_B, threshold=semantic_threshold)
            sim_csv_rows.append(
                [panoA_name, panoB_name, ground_truth, f"{max_sim:.4f}"])

            if not is_passed:
                rejected_phase = 1

        # ========================================================
        # Phase 2: LightGlue 局部特徵匹配 (防白牆與異質場景幻覺)
        # ========================================================
        if is_passed and use_lightglue:
            feat_A_local = pano_local_features_cache[pA]
            feat_B_local = pano_local_features_cache[pB]

            is_passed, max_matches = check_lightglue_match(
                matcher, feat_A_local, feat_B_local, threshold=lightglue_threshold)

            if not is_passed:
                rejected_phase = 2

        # ========================================================
        # Phase 3: DA3 幾何與深度驗證 (最終決策)
        # ========================================================
        if is_passed:
            idx_A, idx_B = list(range(0, 6)), list(range(6, 12))

            with torch.no_grad():
                pred_pose = model_pose.inference(vis_imgs_ref)
                pred_metric = model_metric.inference(
                    vis_imgs_ref) if depth_mode == "hybrid" and model_metric else None

            if export_depth:
                if pA not in house_geometries:
                    house_geometries[pA] = {
                        "depth": pred_pose.depth[0:6], "orig": pred_pose.processed_images[0:6]}
                if pB not in house_geometries:
                    house_geometries[pB] = {
                        "depth": pred_pose.depth[6:12], "orig": pred_pose.processed_images[6:12]}

            is_conn_bool, raw_vis_data = evaluate_connectivity(
                pred_pose, idx_A, idx_B, depth_mode=depth_mode, prediction_metric=pred_metric,
                occlusion_tolerance=0.20, use_bidirectional=use_bidirectional)

            is_connected_pred = 1 if is_conn_bool else 0

            vis_data['pts_A'] = raw_vis_data['pts_A']
            vis_data['pts_B'] = raw_vis_data['pts_B']
            vis_data['count'] = raw_vis_data['count']
            vis_idx_A = raw_vis_data['idx_A']
            vis_idx_B = raw_vis_data['idx_B']

        else:
            # 統計被提早攔截的數量
            if rejected_phase == 1:
                semantic_rejected_count += 1
            elif rejected_phase == 2:
                lightglue_rejected_count += 1

            is_connected_pred = 0

        # 處理統計數據 (以 Recall 導向詳細分類)
        is_correct = (is_connected_pred == ground_truth)
        do_visualize = True
        pred_type = ""

        if is_correct:
            total_correct += 1
            if ground_truth == 1:
                true_positives += 1
                pred_type = "TP"
            else:
                true_negatives += 1
                pred_type = "TN"
        else:
            if is_connected_pred == 1 and ground_truth == 0:
                false_positives += 1
                pred_type = "FP"
            elif is_connected_pred == 0 and ground_truth == 1:
                false_negatives += 1
                pred_type = "FN"

        # 視覺化儲存
        if do_visualize:
            save_path = str(Path(output_dir) / house_id / pred_type /
                            f"{pred_type}_{panoA_name}_vs_{panoB_name}.jpg")

            # 在圖片標題清楚註記是被哪一層漏斗攔截的
            reject_str = ""
            if rejected_phase == 1:
                reject_str = f" [Phase 1: DINO Sim {max_sim:.2f}]"
            elif rejected_phase == 2:
                reject_str = f" [Phase 2: LightGlue {max_matches} pts]"
            else:
                reject_str = f" [Phase 3: DA3 Passed]" if is_connected_pred else f" [Phase 3: DA3 Failed]"

            # 若判斷為 TN 或 FN，載入原始全景圖進行視覺化，並清空投影點位以防座標錯亂
            if pred_type in ["TN", "FN"]:
                img_A_out = Image.open(dataset_dir / pA).convert("RGB")
                img_B_out = Image.open(dataset_dir / pB).convert("RGB")
                pts_A_out = np.empty((0, 2))
                pts_B_out = np.empty((0, 2))
            else:
                img_A_out = vis_imgs_ref[vis_idx_A]
                img_B_out = vis_imgs_ref[vis_idx_B]
                pts_A_out = vis_data['pts_A']
                pts_B_out = vis_data['pts_B']

            save_visualization(img_A_out, img_B_out,
                               pts_A_out, pts_B_out, save_path,
                               title=f"GT: {ground_truth} | Pred: {is_connected_pred} ({pred_type}){reject_str} | Match: {max(0, vis_data['count'])}")

        total_samples += 1
        torch.cuda.empty_cache()

    # 導出相似度 CSV (僅在有啟用語義篩選且有資料時輸出)
    export_similarity_csv(sim_csv_rows, output_dir, house_id)

    if export_depth:
        export_house_depth_visualizations(
            house_id, house_geometries, output_base_dir=depth_out_dir)

    # 計算 Recall, Precision 與 F1-Score
    recall = (true_positives / total_gt_connected *
              100) if total_gt_connected > 0 else 0.0
    precision = (true_positives / (true_positives + false_positives)
                 * 100) if (true_positives + false_positives) > 0 else 0.0
    accuracy = (total_correct / total_samples *
                100) if total_samples > 0 else 0.0

    # 計算 F1-Score：精確率和召回率的調和平均數，防止模型靠「全猜正類」來刷高 Recall
    f1_score = (2 * precision * recall / (precision + recall)
                ) if (precision + recall) > 0 else 0.0

    elapsed_time = time.time() - start_time

    print("\n" + "="*55)
    print(f"房屋 {csv_path.name} 評估報告")
    print(f"推論架構      ：3-Stage Funnel (Global -> Local -> Geometry)")
    print(f"深度推論模式  ：{depth_mode.upper()}")
    print(
        f"P1 語義攔截   ：{'[啟用]' if use_semantic else '[停用]'} (門檻 {semantic_threshold})")
    print(
        f"P2 局部匹配   ：{'[啟用]' if use_lightglue else '[停用]'} (門檻 {lightglue_threshold} pts)")
    print(f"P3 雙向驗證   ：{'[啟用]' if use_bidirectional else '[停用]'}")
    print(f"全景圖數量    ：{num_panos}")
    print(f"總測試配對數  ：{total_samples}")
    print(f"-------------------------------------------------------")
    print(f">> P1 (DINOv2) 攔截數    : {semantic_rejected_count}")
    print(f">> P2 (LightGlue) 攔截數 : {lightglue_rejected_count}")
    print(f"-------------------------------------------------------")
    print(f"真實連通配對 (GT Positives) : {total_gt_connected}")
    print(f"真實不連通   (GT Negatives) : {total_gt_disconnected}")
    print(f"正確預測連通 (TP)           : {true_positives}")
    print(f"正確預測不連通 (TN)         : {true_negatives}")
    print(f"誤判為連通 (FP - Hard/Soft) : {false_positives}")
    print(f"漏判連通 (FN - 致命錯誤)    : {false_negatives}")
    print(f"-------------------------------------------------------")
    print(f"模型召回率 (Recall)         : {recall:.2f}%")
    print(f"模型精確率 (Precision)      : {precision:.2f}%")
    print(f"模型 F1-Score               : {f1_score:.2f}%  <== 終極評估指標")
    print(f"模型準確率 (Accuracy)       : {accuracy:.2f}%")
    print(f"總運算時間                  : {elapsed_time:.2f} 秒")
    out_dir_str = str(Path(output_dir) / house_id).replace("/", "\\")
    print(f"視覺化結果已分類儲存至      : {out_dir_str}/{{TN|TP|FP|FN}}/")
    print("="*55)

    return f1_score, recall, precision, accuracy, total_samples, elapsed_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DA3 連通性評估腳本 (3-Stage Funnel 架構)")
    parser.add_argument("--depth_mode", type=str,
                        choices=["relative_only", "hybrid"], default="relative_only")

    # Phase 1: DINOv2 全域閾值 (建議調低至 0.3，僅作為最寬鬆的防線)
    parser.add_argument("--semantic_threshold", type=float,
                        default=0.3, help="Phase 1 (DINOv2) 門檻")

    # Phase 2: LightGlue 局部閾值 (15 代表要找到 15 個特徵角點才算過關)
    parser.add_argument("--lightglue_threshold", type=int,
                        default=15, help="Phase 2 (LightGlue) 匹配點數量門檻")

    # 供消融實驗使用的動態開關
    parser.add_argument("--disable_semantic", action="store_true",
                        help="停用 Phase 1 (DINOv2) 語義相似性預篩選")
    parser.add_argument("--disable_lightglue", action="store_true",
                        help="停用 Phase 2 (LightGlue) 局部特徵匹配")
    parser.add_argument("--disable_bidirectional", action="store_true",
                        help="停用 Phase 3 (DA3) 雙向幾何驗證 (退回單向 A->B)")

    parser.add_argument("--dataset_root", type=str, default="./Dataset")
    parser.add_argument("--house_csv", type=str,
                        default="./Dataset/Metadatas/DollhouseTask_65826_NoOutdoor_connectivity.csv")
    parser.add_argument("--output_dir", type=str,
                        default="./Plots/2026_03_23_7")
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--depth_out_dir", type=str, default="./Depth_Visuals")
    args = parser.parse_args()

    os.environ["DA3_LOG_LEVEL"] = "WARN"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    da3_pose = DepthAnything3.from_pretrained(
        "depth-anything/DA3-LARGE-1.1").to(device).eval()
    da3_metric = DepthAnything3.from_pretrained(
        "depth-anything/DA3METRIC-LARGE").to(device).eval() if args.depth_mode == "hybrid" else None

    evaluate_single_house(
        dataset_root=args.dataset_root, house_csv_path=args.house_csv,
        model_pose=da3_pose, device=device, depth_mode=args.depth_mode,
        model_metric=da3_metric, output_dir=args.output_dir,
        export_depth=args.export_depth, depth_out_dir=args.depth_out_dir,

        # 傳遞門檻與開關參數
        semantic_threshold=args.semantic_threshold,
        lightglue_threshold=args.lightglue_threshold,
        use_semantic=not args.disable_semantic,
        use_lightglue=not args.disable_lightglue,
        use_bidirectional=not args.disable_bidirectional
    )

    # ---------------------------------------------------------
    # 【未來擴充指南】如果您未來想要跑一個資料夾下所有的 CSV，只需加上這段：
    #
    # metadata_dir = Path(args.dataset_root) / 'Metadatas'
    # all_csvs = list(metadata_dir.glob("*_connectivity.csv"))
    # global_f1_sum = 0
    # global_recall_sum = 0
    # global_precision_sum = 0
    # global_total_houses = 0
    # global_time = 0.0
    #
    # for csv_file in all_csvs:
    #      f1, recall, prec, acc, total, elapsed = evaluate_single_house(
    #          args.dataset_root, csv_file, da3_pose, device,
    #          query_grid_size=(20, 20), conf_threshold=0.5,
    #          depth_mode=args.depth_mode, model_metric=da3_metric,
    #          output_dir=args.output_dir,
    #          semantic_threshold=args.semantic_threshold,
    #          lightglue_threshold=args.lightglue_threshold,
    #          use_semantic=not args.disable_semantic,
    #          use_lightglue=not args.disable_lightglue,
    #          use_bidirectional=not args.disable_bidirectional)
    #      global_f1_sum += f1
    #      global_recall_sum += recall
    #      global_precision_sum += prec
    #      global_total_houses += 1
    #      global_time += elapsed
    #
    # print(f"所有房屋平均 F1-Score: {(global_f1_sum / global_total_houses):.2f}%")
    # print(f"所有房屋平均 Recall: {(global_recall_sum / global_total_houses):.2f}%")
    # print(f"所有房屋平均 Precision: {(global_precision_sum / global_total_houses):.2f}%")
    # print(f"所有房屋總運算時間: {global_time:.2f} 秒")
    # --------------------------------------------------------
