import csv
import time
import torch
import argparse  # [新增邏輯] 匯入 argparse 模組
import os  # [新增邏輯] 匯入 os 用於建立視覺化資料夾
import matplotlib.pyplot as plt  # [新增邏輯] 匯入 matplotlib 用於視覺化
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# 匯入 DA3 與工具組
from depth_anything_3.api import DepthAnything3
from panorama_projector import PanoramaProjector
from utils import (save_visualization, generate_perspective_imgs,
                   export_house_depth_visualizations)


# TODO: query points的撒點選擇也很重要，有效率的撒點可以提高效率。
# TODO: confidence 有沒有可能DA3都預測是很低的情況，導致沒有空間點可以匹配 ?
def evaluate_connectivity(prediction_pose, pers_idx_A, pers_idx_B, overlap_threshold=0.05, query_grid_size=(20, 20), conf_threshold=0.5, depth_mode="relative_only", prediction_metric=None):
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

    【回傳值】
    - Tuple[bool, dict]: 回傳 (是否連通, 視覺化所需的字典包)。
    """
    total_query_points = 0
    visible_in_B_points = 0

    # [新增邏輯] 儲存視覺化用的最佳視角配對資訊
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

        # [新增邏輯] 使用 np.copy 避免修改到原始 prediction 的記憶體，保護矩陣不被二次污染
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

        # [新增邏輯] 如果這是目前最多有效深度的 A 視角，且尚未找到投影成功的 B，先用它當作預設視覺化圖
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
            # [新增邏輯] 必須使用 np.copy 以免改到原始資料，並且在 hybrid 模式下同步放大 B 的平移向量
            ext_B = np.copy(prediction_pose.extrinsics[j])
            if depth_mode == "hybrid":
                ext_B[:, 3] = ext_B[:, 3] * scale_factor

            int_B = prediction_pose.intrinsics[j]
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

            # [新增邏輯] 擷取並儲存成功投影的點，以供視覺化
            valid_B_mask = front_mask & in_fov
            count_B = np.sum(valid_B_mask)
            if count_B > best_vis['count']:
                best_vis['count'] = count_B
                best_vis['idx_A'] = i
                best_vis['idx_B'] = j
                best_vis['pts_A'] = np.stack(
                    [u[valid_B_mask], v[valid_B_mask]], axis=1)
                best_vis['pts_B'] = np.stack(
                    [u_B[valid_B_mask], v_B[valid_B_mask]], axis=1)

            # 圖像B的某個透視圖可以看到query points
            is_visible_anywhere |= valid_B_mask

        visible_in_B_points += np.sum(is_visible_anywhere)

    if total_query_points == 0:
        return False, best_vis
    is_connected = (visible_in_B_points /
                    total_query_points) > overlap_threshold
    return is_connected, best_vis


def evaluate_single_house(dataset_root, house_csv_path, model_pose, device="cuda", query_grid_size=(20, 20), conf_threshold=0.5, depth_mode="relative_only", model_metric=None, window_size=4, stride=1, output_dir="./Plots", export_depth=False, depth_out_dir="./Depth_Visuals"):
    """
    執行單一棟房屋評估 (優化版：收集推論結果以供深度視覺化，避免二次 Inference)。

    【參數說明】
    - dataset_root (str | Path): 資料集的根目錄路徑，用於解析圖片相對路徑。
    - house_csv_path (str | Path): 記錄該棟房屋測試配對與真實連通性標籤 (Ground Truth) 的 CSV 檔案路徑。
    - model_pose (DepthAnything3): 已初始化並載入權重的 DA3 模型實例。
    - device (str): 模型與張量運算所使用的硬體設備，預設為 "cuda"。
    - query_grid_size (Tuple[int, int]): 控制連通性驗證時的撒點密度，預設為 (20, 20)。
    - conf_threshold (float): 控制擷取 Query Point 的最低信心度門檻。
    - depth_mode (str): 控制深度推論模式 ("relative_only", "hybrid")。
    - model_metric (DepthAnything3): 當選擇 "hybrid" 模式時，需傳入的 Metric 深度模型。
    - window_size (int): 滑動窗口大小，代表一次推論的 Batch 包含幾張全景圖 (每張又切 6 透視)。
    - stride (int): 滑動窗口的步長，控制 Batch 與 Batch 之間的 Overlap 程度。
    - output_dir (str): [新增參數] 視覺化圖片輸出的根目錄路徑。
    """
    start_time = time.time()  # 計算花費時間

    dataset_dir = Path(dataset_root)
    csv_path = Path(house_csv_path)

    # 1. 讀取單棟房屋的 CSV 檔案
    rows = []
    # [修改邏輯] 使用 dict 來取代 set，因為 Python 3.7+ 的 dict 可以完美保留插入順序 (即看房軌跡)
    unique_panos_dict = {}

    if not csv_path.exists():
        print(f"找不到指定的 metadata 檔案：{csv_path}")
        return 0, 0, 0, 0.0

    # Fetch out all the panos
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            # 注意：CSV內的表頭仍為 Image_A/B，但我們在程式內明確將其視為 Pano
            # 利用 dict 的 key 不重複特性，同時保留原本在 CSV 中出現的順序 (即真實的人類走動路徑)
            unique_panos_dict[row['Image_A']] = None
            unique_panos_dict[row['Image_B']] = None

    # TODO : 圖片的排序最好是依照一個人逛房子的路徑排序，可以讓DA3的幾何一致性最大化，因為有相同圖像的overlap作為anchor參考
    # [修改邏輯] 直接提取 dict 的 keys 轉為 list，即可完美保留人類走動看房的路徑順序，無需使用 sorted 打亂
    unique_panos = list(unique_panos_dict.keys())

    # 計算全景圖數量
    num_panos = len(unique_panos)
    house_id = csv_path.stem.replace("_connectivity", "")

    # [新增邏輯] 初始化幾何緩存，避免 export_depth 時重複推論
    house_geometries = {}

    # 2. 統一生成該房屋的「所有」透視圖，並存入快取
    pano_imgs_cache = {}
    print(f"\n--- 開始處理房屋：{house_id} ---")
    for pano_rel_path in tqdm(unique_panos, desc="Caching Perspectives"):
        pano_full_path = dataset_dir / pano_rel_path
        if pano_full_path.exists():
            pano_imgs_cache[pano_rel_path] = generate_perspective_imgs(
                pano_full_path, fov=90, device=device)

    # [新增邏輯] 2. 滑動窗口批次推論 (Sliding Window Inference)
    # 將推論結果存在一個字典中，Key 為 (PanoA_rel, PanoB_rel) 的 Tuple
    pair_prediction_cache = {}

    print(f"執行滑動窗口幾何推理...")
    # 將相鄰的全景圖放在同一個推論 Batch 中，使 DA3 透過 Cross-attention 捕捉幾何一致性
    for i in tqdm(range(0, max(1, num_panos - window_size + 1), stride), desc="Sliding Windows"):
        window_pano_paths = unique_panos[i: i + window_size]
        batch_imgs = []
        for p in window_pano_paths:
            batch_imgs.extend(pano_imgs_cache[p])

        with torch.no_grad():
            pred_pose = model_pose.inference(batch_imgs)
            pred_metric = model_metric.inference(
                batch_imgs) if depth_mode == "hybrid" and model_metric else None

            # [優化邏輯] 將推論結果存入緩存
            for idx, p_path in enumerate(window_pano_paths):
                if p_path not in house_geometries:
                    house_geometries[p_path] = {
                        "depth": pred_pose.depth[idx*6: (idx+1)*6],
                        "orig": pred_pose.processed_images[idx*6: (idx+1)*6]
                    }

                # 建立局部索引 (每張全景圖佔 6 個索引)
                for idx_target, p_target in enumerate(window_pano_paths):
                    if p_path == p_target:
                        continue
                    pair_prediction_cache[(p_path, p_target)] = (
                        pred_pose, pred_metric,
                        list(range(idx*6, (idx+1)*6)),
                        list(range(idx_target*6, (idx_target+1)*6)),
                        batch_imgs
                    )

    total_correct, total_samples = 0, 0
    false_positives, false_negatives = 0, 0

    # 4. 驗證連通性配對與分類視覺化 (TN, TP, FP, FN)
    print("驗證連通性配對與分類視覺化...")
    for row in tqdm(rows, desc="Evaluating Pairs"):
        pA, pB = row['Image_A'], row['Image_B']
        ground_truth = int(row['Is_Connected'])

        # [修改邏輯] 嘗試從滑動窗口快取中獲取具備「強空間一致性」的推論結果
        if (pA, pB) in pair_prediction_cache:
            pred_pose, pred_metric, idx_A, idx_B, imgs_ref = pair_prediction_cache[(
                pA, pB)]
        else:
            # 如果 CSV 裡的配對在滑動窗口中沒出現過 (例如跨樓層或距離極遠)，則回退到原本的單獨成對 (Pair-wise) 推論
            imgs_ref = pano_imgs_cache[pA] + pano_imgs_cache[pB]
            idx_A, idx_B = list(range(0, 6)), list(range(6, 12))

            # 將推論移入迴圈內，作為備用方案
            with torch.no_grad():
                pred_pose = model_pose.inference(imgs_ref)
                pred_metric = model_metric.inference(
                    imgs_ref) if depth_mode == "hybrid" and model_metric else None

            if pA not in house_geometries:
                house_geometries[pA] = {
                    "depth": pred_pose.depth[0:6], "orig": pred_pose.processed_images[0:6]}
            if pB not in house_geometries:
                house_geometries[pB] = {
                    "depth": pred_pose.depth[6:12], "orig": pred_pose.processed_images[6:12]}

        # 傳遞深度模式參數、metric prediction 以及局部視角索引
        # 同時接收連通性布林值與視覺化資料
        is_conn_bool, vis_data = evaluate_connectivity(
            pred_pose, idx_A, idx_B, depth_mode=depth_mode, prediction_metric=pred_metric)
        is_connected_pred = 1 if is_conn_bool else 0

        # [修改邏輯] 處理統計數據並設定 Positive / Negative 與 TP/TN/FP/FN (移除輸出數量限制)
        is_correct = (is_connected_pred == ground_truth)
        do_visualize = True  # 設為全量輸出，不論正確與否皆輸出視覺化圖
        pred_type = ""

        if is_correct:
            total_correct += 1
            pred_type = "TP" if ground_truth == 1 else "TN"
        else:
            if is_connected_pred == 1 and ground_truth == 0:
                false_positives += 1
                pred_type = "FP"
            elif is_connected_pred == 0 and ground_truth == 1:
                false_negatives += 1
                pred_type = "FN"

        # [修改邏輯] 執行視覺化儲存，套用自訂的 output_dir 路徑
        if do_visualize:
            panoA_name = Path(pA).stem
            panoB_name = Path(pB).stem
            # 檔案路徑格式: {output_dir}/{house_id}/{pred_type}/{pred_type}_imgA_vs_imgB.jpg
            save_path = str(Path(output_dir) / house_id / pred_type /
                            f"{pred_type}_{panoA_name}_vs_{panoB_name}.jpg")

            # 從當前的 batch_imgs_ref 中提取對應的 PIL Image
            save_visualization(imgs_ref[vis_data['idx_A']], imgs_ref[vis_data['idx_B']],
                               vis_data['pts_A'], vis_data['pts_B'], save_path,
                               title=f"GT: {ground_truth} | Pred: {is_connected_pred} ({pred_type}) | Project Points: {max(0, vis_data['count'])}")

        total_samples += 1

        # 強制釋放此回合的 GPU 記憶體緩存，確保穩定性
        torch.cuda.empty_cache()

    # [優化邏輯] 在所有預測完成後導出深度圖，避免二次 Inference
    if export_depth:
        export_house_depth_visualizations(
            house_id, house_geometries, output_base_dir=depth_out_dir)

    # 4. 輸出該房屋的準確率報告
    elapsed_time = time.time() - start_time

    # [修改邏輯] 報表列印出精確符合使用者要求的格式
    print("\n" + "="*45)
    print(f"房屋 {csv_path.name} 評估報告")
    print(f"推論模式    ：{depth_mode.upper()}")
    print(f"滑動窗口設定：Window Size = {window_size}, Stride = {stride}")
    print(f"全景圖數量  ：{num_panos}")
    print(f"總測試配對數：{total_samples}")
    print(f"正確預測數  ：{total_correct}")
    print(f"誤判為連通 (False Positives)：{false_positives}")
    print(f"誤判不連通 (False Negatives)：{false_negatives}")
    accuracy = (total_correct / total_samples) * \
        100 if total_samples > 0 else 0
    print(f"模型準確率  ：{accuracy:.2f}%")
    print(f"總運算時間  ：{elapsed_time:.2f} 秒")
    # 將反斜線標準化以符合期望的輸出格式
    out_dir_str = str(Path(output_dir) / house_id).replace("/", "\\")
    print(f"視覺化結果已分類儲存至：{out_dir_str}/{{TN|TP|FP|FN}}/")
    print("="*45)

    return accuracy, total_correct, total_samples, elapsed_time


if __name__ == "__main__":
    # 使用 argparse 讓使用者可以透過終端機傳遞參數
    parser = argparse.ArgumentParser(description="DA3 連通性評估腳本 (完整效能優化版)")
    parser.add_argument("--depth_mode", type=str,
                        choices=["relative_only", "hybrid"], default="relative_only",
                        help="選擇深度推理模式: 'relative_only' 或 'hybrid'")
    parser.add_argument("--window_size", type=int, default=4, help="滑動窗口大小")
    parser.add_argument("--stride", type=int, default=1, help="滑動窗口步長")
    parser.add_argument("--dataset_root", type=str,
                        default="./Dataset", help="資料集根目錄路徑")
    parser.add_argument("--house_csv", type=str,
                        default="./Dataset/Metadatas/DollhouseTask_65826_NoOutdoor_connectivity.csv",
                        help="單棟房屋的測試配對 CSV 檔案路徑")
    parser.add_argument("--output_dir", type=str,
                        default="./Plots/2026_03_21", help="視覺化圖片的輸出根目錄路徑")
    parser.add_argument(
        "--export_depth", action="store_true", help="啟用則導出深度品質視覺化")
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
        model_metric=da3_metric, window_size=args.window_size, stride=args.stride,
        output_dir=args.output_dir, export_depth=args.export_depth, depth_out_dir=args.depth_out_dir
    )

    # ---------------------------------------------------------
    # 【未來擴充指南】如果您未來想要跑一個資料夾下所有的 CSV，只需加上這段：
    #
    # metadata_dir = Path(args.dataset_root) / 'Metadatas'
    # all_csvs = list(metadata_dir.glob("*_connectivity.csv"))
    # global_correct = 0
    # global_total = 0
    # global_time = 0.0
    #
    # for csv_file in all_csvs:
    #     acc, correct, total, elapsed = evaluate_single_house(
    #         args.dataset_root, csv_file, da3_pose, device,
    #         query_grid_size=(20, 20), conf_threshold=0.5,
    #         depth_mode=args.depth_mode, model_metric=da3_metric,
    #         window_size=args.window_size, stride=args.stride,
    #         output_dir=args.output_dir)
    #     global_correct += correct
    #     global_total += total
    #     global_time += elapsed
    #
    # print(f"所有房屋總準確率: {(global_correct / global_total) * 100:.2f}%")
    # print(f"所有房屋總運算時間: {global_time:.2f} 秒")
    # --------------------------------------------------------
