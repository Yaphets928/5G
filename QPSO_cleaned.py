#!/usr/bin/env python3
import os
os.environ["XI"] = "10"   # 改成 2 / 4 / 6 / 8 / 10

# QPSA_UPA_V4_patched.py  — minimal changes: merge grid_tx into offset; bandwidth-weighted objective; UE plotting (black dots) restored; PSO function added
# 相比v12 1、根据审稿意见，扩充XI

import json, math, hashlib, time
import numpy as np

# [INIT-SPACING] 放大初始UPA间距，减少clip（k>=1），仅影响初始化；不改变后续逻辑/注释/打印
INIT_SPACING_MULT_MICRO = 1.30  # 微站：建议 1.2~1.5；默认 1.30
INIT_SPACING_MULT_MACRO = 1.20  # 宏站：建议 1.1~1.4；默认 1.20

# ================= UA-OT (Theorem 1) 动态负载权重：全局状态 =================
# 目的：将论文 ISAC_UAV_OT 的 UA 判别式中 z(U)=1/U 融入离散 OT 代价。
# 做法：在每次 UA-OT 调用前，使用上一轮的行和 U_prev（每个基站承担的 UE 数）
#       来构造动态代价 C_{j,k} = -(alpha * R_{j,k}) / max(U_prev[j], eps)。
UA_PREV_LOADS = None  # 上一次 UA 的行和（单位：UE 数量）。若 None，则用目标行边际 b_marg 作为初值
UA_EPS_LOAD = 1e-9    # 防止除零

# ---- [CRE-INIT] one-shot initializer (power-weighted RSRP + dB bias + capacity repair) ----
CRE_BIAS_DB = 9.0  # set to 0.0 to disable one-shot CRE initialization
CRE_INIT_USED = False

def _ua_cre_init(H, params, Jm, Mm, MM, cre_db=CRE_BIAS_DB):
    '''
    Build one-shot UA by CRE: use power-weighted RSRP (P_j * ||H||_F^2),
    add micro dB-bias, greedy assign under capacity, then simple overload repair.
    Only for the *first* UA. All subsequent UA steps still use lp_and_integerize().
    '''
    import numpy as np
    Jm = int(Jm); JM = int(params['JM']); J = Jm + JM
    K = len(H)
    Pj = np.asarray(params['Pj'], float)

    # Power-weighted RSRP-like metric
    R0p = np.zeros((J, K), dtype=float)
    for j in range(J):
        for k in range(K):
            R0p[j, k] = Pj[j] * (np.linalg.norm(H[k][j], 'fro') ** 2)

    # dB-domain bias for micro rows
    eps = 1e-12
    score = 10.0 * np.log10(np.maximum(R0p, eps))
    if cre_db and float(cre_db) != 0.0:
        score[:Jm, :] += float(cre_db)

    # Preferences
    pref = np.argsort(-score, axis=0)  # (J, K), descending by score

    # Station capacities
    cap = np.concatenate([np.full(Jm, int(Mm)), np.full(J - Jm, int(MM))]).astype(int)

    a = np.zeros((J, K), dtype=int)
    load = np.zeros(J, dtype=int)

    # Greedy under capacities
    for k in range(K):
        assigned = False
        for rank in range(J):
            j = int(pref[rank, k])
            if load[j] < cap[j]:
                a[j, k] = 1
                load[j] += 1
                assigned = True
                break
        if not assigned:
            # No capacity anywhere -> leave unassigned (blocked)
            pass

    # Simple overload repair: drop worst from overloaded BS and reassign
    for j in range(J):
        over = int(load[j] - cap[j])
        if over > 0:
            users = np.where(a[j, :] == 1)[0]
            # ascending by score -> drop worst
            worst_users = users[np.argsort(score[j, users])]
            for k in worst_users[:over]:
                a[j, k] = 0
                load[j] -= 1
                placed = False
                for rank in range(J):
                    jj = int(pref[rank, k])
                    if load[jj] < cap[jj]:
                        a[jj, k] = 1
                        load[jj] += 1
                        placed = True
                        break
                if not placed:
                    # No space anywhere; leave unassigned (blocked)
                    pass

    return a
# -------------------------------------------------------------------------


# === Monte Carlo channel epoch (for random AoD/AoA/Sigma per run) ===
CHAN_EPOCH = 0  # will be set in main() during MC loop

if False:  # [MC-quiet] disable plotting
    pass

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

C = 3e8
SEED = 1
np.random.seed(SEED)
# 供 PSO 代码使用的全局 RNG（保持与给定片段一致，提供 rand/randn 接口）
global_rng = np.random.RandomState(SEED)

def string_tf(b): return "true" if b else "false"

def steering_matrix_from_pos_3d(p_elems, lam, thetas, phis):
    # 相位：x*sinθ*cosφ + y*cosθ；列归一化 1/sqrt(N)
    p = np.asarray(p_elems, float)
    N = p.shape[0]
    ux = np.sin(thetas) * np.cos(phis)
    uy = np.cos(thetas)
    phase = (p[:, [0]] @ ux[None, :]) + (p[:, [1]] @ uy[None, :])
    return np.exp(-1j * (2*np.pi/lam) * phase) / np.sqrt(max(1, N))

def fspl_power(d, fc):
    lam = C / fc; d = max(float(d), 1e-2)
    return (lam/(4*np.pi*d))**2

def stable_seed(*parts):
    # Inject CHAN_EPOCH so that each MC channel run randomizes seeds without touching callers
    parts = (globals().get('CHAN_EPOCH', 0),) + parts
    s = "|".join([("{:.4f}".format(float(x)) if isinstance(x,(int,float,np.floating)) else str(x)) for x in parts])
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:4], "little", signed=False)

def stable_seed_noepoch(*parts):
    s = "|".join([("{:.4f}".format(float(x)) if isinstance(x,(int,float,np.floating)) else str(x)) for x in parts])
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:4], "little", signed=False)

def wmmse_given_a(H, a, params, max_iter=3, tol=1e-3):
    """
    Weighted-WMMSE under fixed integer UA a.

    Notes
    -----
    1) Returns spectral efficiencies R[j,k] = log2(1+SINR) in bits/s/Hz.
       The outer function inner_wmmse_cvxp() is responsible for multiplying
       the BS bandwidths B_j to form the final bandwidth-weighted WSR.

    2) The transmit update adopts the corrected per-BS sum-power budget:
           sum_k a[j,k] * ||w_{j,k}||^2 <= Pj[j].
       Hence, all active beams transmitted by the same BS j are jointly coupled
       through one shared Lagrange multiplier (implemented by bisection).

    3) To remain consistent with the revised main.tex after the bandwidth fix,
       a BS-specific positive weight eta_j proportional to B_j is introduced
       in the WMMSE transmit update. The global constant 1/ln(2) is omitted
       since it does not affect the optimizer.
    """
    Jm, JM = params["Jm"], params["JM"]
    J = Jm + JM
    K = len(H)
    Nr = params["Nr_ue"]
    Nt_m, Nt_M = params["Nt_micro"], params["Nt_macro"]
    sigma2 = params["sigma2"]
    Pj = params["Pj"]

    # ----- BS bandwidth weights for the weighted-WMMSE transmit update -----
    # The factor 1/ln(2) is a common positive constant and is omitted.
    Bfac_raw = np.concatenate([
        np.full(Jm, float(params["B_micro"])),
        np.full(JM, float(params["B_macro"]))
    ])
    # normalize for numerical stability only
    Bfac = Bfac_raw / max(float(np.max(Bfac_raw)), 1.0)

    # ----- macro/micro orthogonality model -----
    def _count_as_interf(j_ref: int, jj: int, Jm: int) -> bool:
        # If the served BS is a micro BS: all micros interfere, macros are orthogonal
        if j_ref < Jm:
            return jj < Jm
        # If the served BS is the macro BS: only same macro-station multiuser interference remains
        else:
            return jj == j_ref

    # beamformers
    w = [[np.zeros((Nt_m if j < Jm else Nt_M, 1), dtype=complex) for _ in range(K)] for j in range(J)]

    # ----- initialization: dominant right singular vector on each associated link -----
    for k in range(K):
        j_star = int(np.argmax(a[:, k]))
        if a[j_star, k] > 0.5:
            Hj = H[k][j_star]
            _, _, Vh = np.linalg.svd(Hj, full_matrices=False)
            w[j_star][k] = Vh.conj().T[:, [0]]

    u = [np.ones((Nr, 1), dtype=complex) for _ in range(K)]
    omega = np.ones((K, 1))

    for _ in range(max_iter):

        # =========================================================
        # 1) update receive beamformers u
        # =========================================================
        for k in range(K):
            if np.sum(a[:, k]) < 0.5:
                u[k] = np.zeros((Nr, 1), dtype=complex)
                continue

            j_star = int(np.argmax(a[:, k]))
            Hj = H[k][j_star]

            Rk = sigma2 * np.eye(Nr, dtype=complex)
            for jj in range(J):
                if not _count_as_interf(j_star, jj, Jm):
                    continue
                Hj2 = H[k][jj]
                Sj = np.zeros((Hj2.shape[1], Hj2.shape[1]), dtype=complex)
                for m in range(K):
                    Sj += w[jj][m] @ w[jj][m].conj().T
                Rk = Rk + Hj2 @ Sj @ Hj2.conj().T

            v = Hj @ w[j_star][k]
            try:
                u_k = np.linalg.solve(Rk, v)
            except Exception:
                u_k = np.linalg.pinv(Rk) @ v

            un = np.sqrt(np.real(u_k.conj().T @ u_k))[0, 0]
            u[k] = u_k / (un if un > 1e-12 else 1.0)

        # =========================================================
        # 2) update MSE weights omega
        # =========================================================
        # Positive user/BS weights do not change the optimizer omega*=1/e.
        for k in range(K):
            if np.sum(a[:, k]) < 0.5:
                omega[k, 0] = 0.0
                continue

            j_star = int(np.argmax(a[:, k]))
            Hj = H[k][j_star]
            v = Hj @ w[j_star][k]

            interf = 0.0
            for jj in range(J):
                if not _count_as_interf(j_star, jj, Jm):
                    continue
                for m in range(K):
                    if jj == j_star and m == k:
                        continue
                    term = (u[k].conj().T @ (H[k][jj] @ w[jj][m])).item()
                    interf += float(np.abs(term) ** 2)

            ukv = (u[k].conj().T @ v)[0, 0]
            ukuk = (u[k].conj().T @ u[k])[0, 0].real
            sig = np.abs(ukv) ** 2
            e = 1.0 - 2.0 * np.real(ukv) + (sig + interf + sigma2 * ukuk)
            omega[k, 0] = 1.0 / max(e, 1e-9)

        # =========================================================
        # 3) update transmit beamformers w under per-BS sum-power
        # =========================================================
        # For each BS j, all active beams {w[j][k]} are jointly updated
        # under one shared power budget Pj[j], with one shared Lagrange
        # multiplier found by bisection.
        for j in range(J):
            Nt = Nt_m if j < Jm else Nt_M
            eta_j = Bfac[j]

            A = np.zeros((Nt, Nt), dtype=complex)
            b = [np.zeros((Nt, 1), dtype=complex) for _ in range(K)]
            assigned_users = [k for k in range(K) if a[j, k] > 0.5]

            if len(assigned_users) == 0:
                continue

            for k in assigned_users:
                Hj = H[k][j]
                A += eta_j * omega[k, 0] * (
                    Hj.conj().T @ (u[k] @ u[k].conj().T) @ Hj
                )

            for k in assigned_users:
                Hj = H[k][j]
                b[k] = eta_j * omega[k, 0] * (Hj.conj().T @ u[k])

            def total_power(lmbd):
                X = np.linalg.pinv(A + lmbd * np.eye(Nt))
                p = 0.0
                for kk in assigned_users:
                    wjk = X @ b[kk]
                    p += float(np.real(wjk.conj().T @ wjk)[0, 0])
                return p, X

            pow0, X0 = total_power(0.0)

            if pow0 <= Pj[j]:
                for k in assigned_users:
                    w[j][k] = X0 @ b[k]
            else:
                l, r, Xbest = 1e-12, 1e4, None
                for _ in range(40):
                    mid = 0.5 * (l + r)
                    powm, Xm = total_power(mid)
                    if powm > Pj[j]:
                        l = mid
                    else:
                        r, Xbest = mid, Xm

                X = Xbest if Xbest is not None else np.linalg.pinv(A + r * np.eye(Nt))
                for k in assigned_users:
                    w[j][k] = X @ b[k]

        # optional early stop
        # (kept inactive now because max_iter is already small in the outer loop)

    # =========================================================
    # 4) return spectral efficiencies (bits/s/Hz)
    # =========================================================
    R = np.zeros((J, K))
    for j in range(J):
        for k in range(K):
            if a[j, k] < 0.5:
                continue

            v = H[k][j] @ w[j][k]
            num = float(np.abs((u[k].conj().T @ v).item()) ** 2)
            den = sigma2

            for jj in range(J):
                if not _count_as_interf(j, jj, Jm):
                    continue
                for m in range(K):
                    if jj == j and m == k:
                        continue
                    den += float(np.abs((u[k].conj().T @ (H[k][jj] @ w[jj][m])).item()) ** 2)

            R[j, k] = np.log2(1.0 + num / max(1e-12, den))

    return R, w, u, omega

def lp_and_integerize(R, Mm, MM, Jm, *compat_ignore):
    """
    OT + dummy BS + 动态负载版本（对齐 HetNet_MA_ICC 的 1/U 设计）:

      - OT 阶段：cost 使用 R / U_j（z(U)=1/U），并通过 dummy blocking BS
        把“阻塞”一起纳入行约束；
      - rounding 阶段：在 Σ_j a_{j,k} <= 1、Σ_k a_{j,k} <= cap_real[j] 下，
        按 OT 软关联 a_frac 与真实速率 R 相结合的打分做 column-wise 贪心；
      - Patch C：在 rounding 后，再做一小轮 “只用 surrogate R 的 1-opt 局部搜索”，
        尽量让被阻塞的是全局贡献最差的一批 UE，缓解几何上明显的“舍近求远”。

    输入
    ----
    R   : (J,K) 带宽加权代理速率矩阵 (bps)（现在来自 simplified SINR）
    Mm  : 微站最大并发 UE 数
    MM  : 宏站最大并发 UE 数
    Jm  : 微站个数 (宏站 JM = J - Jm)

    输出
    ----
    a_frac : (J,K) 软关联（去掉 dummy 行后的 OT 连续解）
    a_int  : (J,K) 0/1 整数 UA（给 WMMSE 用）
    """
    import numpy as np
    global UA_PREV_LOADS, UA_EPS_LOAD

    if "UA_PREV_LOADS" not in globals():
        UA_PREV_LOADS = None
    if "UA_EPS_LOAD" not in globals():
        UA_EPS_LOAD = 1e-3

    R = np.asarray(R, float)
    J_real, K = R.shape
    Jm = int(Jm)
    JM = int(J_real - Jm)
    Mm = int(Mm)
    MM = int(MM)

    # ---- 0) 实际 BS 容量向量 & dummy BS 决策 ----
    cap_real = np.concatenate([
        np.full(Jm, Mm, dtype=int),
        np.full(JM, MM, dtype=int)
    ])
    total_cap = int(cap_real.sum())

    has_dummy = False
    if K > total_cap:
        # UE 多于总容量：显式加入 dummy blocking BS，容量 = K - total_cap
        dummy_cap = K - total_cap
        cap_ext = np.concatenate([cap_real, np.array([dummy_cap], dtype=int)])
        J_ext = J_real + 1
        dummy_idx = J_ext - 1
        has_dummy = True
    else:
        # UE 不多于容量：不强制 dummy，cap_ext 等比例缩放到总和 = K
        cap_ext = cap_real.copy()
        J_ext = J_real
        dummy_idx = None

    # ---- 1) OT 的行/列边际 ----
    a_marg = np.ones(K, dtype=float)  # 每 UE 质量 = 1

    if has_dummy:
        b_marg = cap_ext.astype(float)
    else:
        cap_sum = float(cap_ext.sum())
        if cap_sum <= 0:
            b_marg = np.ones(J_ext, dtype=float) * (float(K) / max(J_ext, 1))
        else:
            b_marg = cap_ext.astype(float) * (float(K) / cap_sum)

    # ---- 2) 构造 R_ext，并按论文 z(U)=1/U 做动态负载加权 ----
    if has_dummy:
        R_ext = np.vstack([R, np.zeros((1, K), dtype=float)])
    else:
        R_ext = R.copy()

    if UA_PREV_LOADS is None:
        cap_sum_real = max(float(cap_real.sum()), 1e-12)
        U_prev_real = cap_real.astype(float) * (float(K) / cap_sum_real)
    else:
        prev = np.asarray(UA_PREV_LOADS, float)
        if prev.shape[0] != J_real:
            cap_sum_real = max(float(cap_real.sum()), 1e-12)
            U_prev_real = cap_real.astype(float) * (float(K) / cap_sum_real)
        else:
            s_prev = max(float(prev.sum()), 1e-12)
            U_prev_real = prev * (float(K) / s_prev)

    UA_EPS = float(UA_EPS_LOAD)
    denom_real = np.maximum(U_prev_real, UA_EPS)

    if has_dummy:
        denom_ext = np.concatenate([denom_real, np.array([1.0])])
    else:
        denom_ext = denom_real

    R_base_ext = R_ext / denom_ext[:, None]

    R_max = float(np.max(R_base_ext)) if R_base_ext.size > 0 else 1.0
    if R_max <= 0:
        R_scaled = np.zeros_like(R_base_ext)
    else:
        R_scaled = R_base_ext / R_max

    # ---- 3) Sinkhorn 迭代：求 OT 软解 ----
    eps = 0.05
    eps_eff = max(1e-4, 0.5 * eps)
    tiny = 1e-18
    iters = 300
    tol = 1e-6

    C = R_scaled - np.max(R_scaled, axis=0, keepdims=True)
    Kmat = np.exp(C / eps_eff) + tiny  # (J_ext, K)

    u = np.ones(J_ext, dtype=float) / max(J_ext, 1)
    v = np.ones(K, dtype=float) / max(K, 1)

    for _ in range(iters):
        Kv = Kmat @ v + tiny
        u_new = b_marg / Kv
        Ku = Kmat.T @ u_new + tiny
        v_new = a_marg / Ku

        if max(np.max(np.abs(u_new - u)), np.max(np.abs(v_new - v))) < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new

    P_ext = (u[:, None] * Kmat) * v[None, :]
    col_sum = np.maximum(P_ext.sum(axis=0, keepdims=True), 1e-12)
    A_ext = P_ext / col_sum

    a_frac = A_ext[:J_real, :]  # 去掉 dummy 行

    # ---- 4) column-wise rounding：在容量约束下最大化速率 ----
    a_int = np.zeros((J_real, K), dtype=int)
    load = np.zeros(J_real, dtype=int)

    score_real = a_frac * R  # OT 权重 × 真实 surrogate 速率

    ue_best_score = np.max(score_real, axis=0)
    ue_order = np.argsort(-ue_best_score, kind="mergesort")

    for k in ue_order:
        best_j = -1
        best_val = 0.0
        for j in range(J_real):
            if load[j] >= cap_real[j]:
                continue
            val = score_real[j, k]
            if val > best_val:
                best_val = val
                best_j = j
        if best_j >= 0 and best_val > 0.0:
            a_int[best_j, k] = 1
            load[best_j] += 1
        else:
            # 所有 BS 满载，或这个 UE 实在太差 => blocked
            continue

    # ---- 5) surrogate-1-opt 局部搜索（只用 R 做 WSR 细调） ----
    # 目标函数（基于 surrogate R）
    def _obj(A):
        return float((R * A).sum())

    cur_val = _obj(a_int)

    # 5.1 用完剩余容量：把 blocked UE 直接接到有空位且有正贡献的 BS 上
    assigned = a_int.sum(axis=0).astype(bool)
    for k in range(K):
        if assigned[k]:
            continue
        best_j = -1
        best_gain = 0.0
        for j in range(J_real):
            if load[j] >= cap_real[j]:
                continue
            gain = R[j, k]
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j >= 0 and best_gain > 1e-12:
            a_int[best_j, k] = 1
            load[best_j] += 1
            assigned[k] = True
            cur_val += best_gain

    # 5.2 对仍被阻塞的 UE，尝试替换某 BS 上“最差的 UE”（严格增益才接受）
    assigned = a_int.sum(axis=0).astype(bool)
    blocked_indices = np.where(~assigned)[0]

    for k in blocked_indices:
        best_gain = 0.0
        best_j = -1
        best_k_old = -1
        for j in range(J_real):
            if load[j] <= 0:
                continue
            served_indices = np.where(a_int[j, :] == 1)[0]
            if served_indices.size == 0:
                continue
            scores_served = R[j, served_indices]
            idx_min = int(np.argmin(scores_served))
            k_old = int(served_indices[idx_min])

            gain = R[j, k] - R[j, k_old]
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_k_old = k_old

        if best_gain > 1e-12 and best_j >= 0:
            a_int[best_j, best_k_old] = 0
            a_int[best_j, k] = 1
            # load[best_j] 不变
            assigned[k] = True
            assigned[best_k_old] = False
            cur_val += best_gain


    # ---- 6) 用软行和更新 UA_PREV_LOADS，供下一次 OT 使用 ----
    UA_PREV_LOADS = a_frac.sum(axis=1).astype(float)

    return a_frac, a_int


def make_rx_array(center, Nr):
    center = np.asarray(center, float)
    if Nr==1: return center[None,:]
    vr = np.array([[0.0,0.0],[0.08,0.0]], float)  # 8 cm 间距（按需求）
    return center[None,:] + vr[:Nr,:]

def build_H_for_link_tx_UPA(p_tx_abs, p_tx_center, p_rx_elems, lam_tx, lam_rx, amp_scale, rng_local, L):
    # 合并后：tx_eff = center + offset，此处 p_tx_abs 已是绝对坐标（center + offset）
    offset = p_tx_abs - p_tx_center
    tx_eff = p_tx_center + offset

    # AoD/AoA 采样：θ、φ ∈ [0, π]
    thetas_t = rng_local.random(L) * np.pi; phis_t = rng_local.random(L) * np.pi
    thetas_r = rng_local.random(L) * np.pi; phis_r = rng_local.random(L) * np.pi

    F = steering_matrix_from_pos_3d(tx_eff, lam_tx, thetas_t, phis_t)
    center_rx = np.mean(p_rx_elems, axis=0, keepdims=True)  # 去中心仅影响公共相位
    G = steering_matrix_from_pos_3d(p_rx_elems - center_rx, lam_rx, thetas_r, phis_r)

    alpha = (rng_local.standard_normal(L) + 1j*rng_local.standard_normal(L)) / np.sqrt(2.0*L)
    Sigma = np.diag(alpha)
    return amp_scale * (G @ Sigma @ F.conj().T)

def build_channels(UE_pos, micro_centers, macro_center, ma_off_m, ma_off_M, params):
    K = UE_pos.shape[0]; Jm, JM = params["Jm"], params["JM"]
    Nr = params["Nr_ue"]; Nt_m, Nt_M = params["Nt_micro"], params["Nt_macro"]
    fc_m = params["fc"];  fc_M = params.get("fc_macro", params["fc"])
    L = params["L_paths"]
    sh_m = params.get("shadow_std_dB_micro", 7); sh_M = params.get("shadow_std_dB_macro", 8)
    lam_m, lam_M = C/fc_m, C/fc_M

    # 绝对坐标：center + offset（grid 已并入 offset）
    tx_mic_abs = np.zeros((Jm, Nt_m, 2), float)
    for j in range(Jm): tx_mic_abs[j,:,:] = micro_centers[j,:] + ma_off_m[j,:,:]
    tx_mac_abs = macro_center[None,:] + ma_off_M  # (Nt_M,2)

    H = [[None for _ in range(Jm+JM)] for _ in range(K)]
    for k in range(K):
        rx_center = UE_pos[k,:2]
        rx_elems_m = make_rx_array(rx_center, Nr); rx_elems_M = rx_elems_m.copy()
        # 微站
        for j in range(Jm):
            seed = stable_seed(1,'farfield-upa','m',j,'k',k,rx_center[0],rx_center[1], micro_centers[j,0], micro_centers[j,1], fc_m, L, sh_m)
            rng_s = np.random.default_rng(seed)
            d = np.linalg.norm(rx_center - micro_centers[j,:2])
            beta_pow = fspl_power(d, fc_m) * (10.0**(0.1 * rng_s.standard_normal() * sh_m))
            amp = np.sqrt(beta_pow)
            H[k][j] = build_H_for_link_tx_UPA(tx_mic_abs[j,:,:], micro_centers[j,:], rx_elems_m, lam_m, lam_m, amp, rng_s, L)
        # 宏站
        seed = stable_seed(1,'farfield-upa','M',Jm,'k',k,rx_center[0],rx_center[1], macro_center[0], macro_center[1], fc_M, L, sh_M)
        rng_M = np.random.default_rng(seed)
        dM = np.linalg.norm(rx_center - macro_center[:2])
        beta_pow_M = fspl_power(dM, fc_M) * (10.0**(0.1 * rng_M.standard_normal() * sh_M))
        amp_M = np.sqrt(beta_pow_M)
        H[k][Jm] = build_H_for_link_tx_UPA(tx_mac_abs, macro_center, rx_elems_M, lam_M, lam_M, amp_M, rng_M, L)
    return H

def inner_wmmse_cvxp(UE_pos, micro_centers, macro_center,
                      ma_offsets_m, ma_offsets_M, params, use_cvx):
    """
    内层：给定几何 & MA 位置，做 UA + WMMSE 交替（弱耦合版）：

      - 先用简化 SINR 构造一个与 UA 无关的基础 surrogate 速率矩阵 R_base；
      - 每一轮迭代：
          * 用当前 surrogate R_sur 做 OT + rounding，得到整数 UA a_int；
          * 在 a_int 下调用 WMMSE，得到真实谱效率 R_speff；
          * 用 WMMSE 的结果修正当前 UA 上的 surrogate：对 a_int==1 的边，
            用 B * R_speff 覆盖 R_base，其它边仍然保留 R_base；
          * 若 UA 不再变化，则提前停止。
      - 返回在整个交替过程中 WSR 最好的那一轮 (R_speff, a)。
    """
    import numpy as _np
    global UA_PREV_LOADS

    Jm, JM = params["Jm"], params["JM"]; J = Jm + JM
    Mm, MM = params["M_micro"], params["M_macro"]

    # ---------- 1) 构建物理信道 ----------
    H = build_channels(UE_pos, micro_centers, macro_center,
                       ma_offsets_m, ma_offsets_M, params)
    K = UE_pos.shape[0]

    # 带宽向量（bps = B * bits/s/Hz）
    Bvec = _np.concatenate([
        _np.full(Jm, params["B_micro"]),
        _np.full(JM, params["B_macro"])
    ])

    # ---------- 2) 基础 surrogate：简化 SINR ----------
    sigma2 = float(params["sigma2"])
    Pj = _np.asarray(params["Pj"], dtype=float)

    def _count_as_interf_sur(j_ref: int, jj: int, Jm: int) -> bool:
        # micro UE：所有 micro 互相干扰，macro 与 micro 正交
        if j_ref < Jm:
            return (jj < Jm) and (jj != j_ref)
        # macro UE：只考虑“同一宏站”干扰；这里只有 1 个宏站 => 实际无干扰
        else:
            return False

    S_base = _np.zeros((J, K), dtype=float)
    for k in range(K):
        g = _np.zeros(J, dtype=float)
        for j in range(J):
            h_kj = H[k][j]
            g[j] = Pj[j] * (_np.linalg.norm(h_kj, ord='fro') ** 2)
        for j in range(J):
            interf = 0.0
            for jj in range(J):
                if _count_as_interf_sur(j, jj, Jm):
                    interf += g[jj]
            sinr = g[j] / max(sigma2 + interf, 1e-12)
            S_base[j, k] = _np.log2(1.0 + sinr)

    R_base = (Bvec[:, None]) * S_base  # bps，作为 OT 的基础代价（越大越好）

    # ---------- 3) 初始化 UA_PREV_LOADS：用 CRE 做一次粗 UA ----------
    a0 = _ua_cre_init(H, params, Jm, Mm, MM, CRE_BIAS_DB)
    cap_vec = _np.concatenate([
        _np.full(Jm, Mm, float),
        _np.full(JM, MM, float)
    ])
    UA_PREV_LOADS = _np.minimum(a0.sum(axis=1).astype(float), cap_vec)

    # ---------- 4) UA–WMMSE 交替（弱耦合） ----------
    max_outer = int(params.get("max_iter_wmmse", 3))
    tol_wm = float(params.get("tol_wmmse", 1e-3))

    R_sur = R_base.copy()
    a_prev = None
    best_obj = -1.0
    best_Rspeff = None
    best_a = None
    best_w = best_u = None

    for _ in range(max_outer):
        # (a) OT + rounding 得到整数 UA
        a_frac, a_int = lp_and_integerize(R_sur, Mm, MM, Jm,
                                          use_cvx and HAS_CVXPY)
        # 更新下一轮 OT 的行先验（动态 1/U）
        UA_PREV_LOADS = a_frac.sum(axis=1).astype(float)

        # (b) 在当前 UA 下跑一次 WMMSE
        R_speff, w, u, omega = wmmse_given_a(H, a_int, params,
                                             max_iter=3, tol=tol_wm)

        obj = float(_np.sum((Bvec[:, None] * R_speff) * a_int))
        if obj > best_obj:
            best_obj = obj
            best_Rspeff = R_speff.copy()
            best_a = a_int.copy()
            best_w, best_u = w, u

        # (c) 若 UA 没变，则可以提前停止
        if a_prev is not None and _np.array_equal(a_int, a_prev):
            break
        a_prev = a_int.copy()

        # (d) 构造下一轮的 surrogate：已服务边用 B*R_speff，其余仍用 R_base
        R_sur = R_base.copy()
        mask = (a_int > 0.5)
        R_sur[mask] = (Bvec[:, None] * R_speff)[mask]

    # ---------- 5) 返回在整个交替过程中 WSR 最好的那一轮 ----------
    if best_Rspeff is None:
        # 理论上不会发生，但防御：退回到基础 surrogate 的一次 OT + WMMSE
        a_frac, best_a = lp_and_integerize(R_base, Mm, MM, Jm,
                                           use_cvx and HAS_CVXPY)
        UA_PREV_LOADS = a_frac.sum(axis=1).astype(float)
        best_Rspeff, best_w, best_u, _ = wmmse_given_a(H, best_a, params,
                                                       max_iter=3, tol=tol_wm)
        best_obj = float(_np.sum((Bvec[:, None] * best_Rspeff) * best_a))

    return best_Rspeff, best_a, best_w, best_u, best_obj, H


def evalUnified(v, slot_weights, slot_ues, params, packer, use_cvx, slot_epoch_base=None):
    """
    Generic Xi-slot unified objective.

    slot_weights : array-like, shape (Xi,)
    slot_ues     : list of UE arrays, length Xi

    Notes
    -----
    To make the horizon test non-trivial, each slot uses an independent channel epoch
    derived deterministically from the snapshot base epoch. The day/night traffic states
    may repeat, but the per-slot small-scale channel realization is allowed to differ.
    """
    macro, micro, ma_slots_m, ma_slots_M = packer.unpack(v)
    area = params["area"]; MRm = params["MA_range_micro"]; MRmM = params["MA_range_macro"]
    Xi = packer.Xi

    if not (0 <= macro[0] <= area and 0 <= macro[1] <= area):
        return -1e9
    if np.any(micro < 0) or np.any(micro > area):
        return -1e9
    if np.any(np.abs(ma_slots_m) > MRm + 1e-9):
        return -1e9
    if np.any(np.abs(ma_slots_M) > MRmM + 1e-9):
        return -1e9
    if len(slot_weights) != Xi or len(slot_ues) != Xi:
        raise ValueError(f"slot_weights / slot_ues length must equal Xi={Xi}")

    old_epoch = globals().get('CHAN_EPOCH', 0)
    total = 0.0
    try:
        for s in range(Xi):
            if slot_epoch_base is not None:
                globals()['CHAN_EPOCH'] = int(slot_epoch_base) * 1000 + (s + 1)
                params['chan_epoch'] = globals()['CHAN_EPOCH']
            _, _, _, _, obj_s, _ = inner_wmmse_cvxp(
                slot_ues[s], micro, macro, ma_slots_m[s], ma_slots_M[s], params, use_cvx
            )
            total += float(slot_weights[s]) * float(obj_s)
    finally:
        globals()['CHAN_EPOCH'] = old_epoch
        params['chan_epoch'] = old_epoch
    return total

def run_qpso_unified(init_vec, eval_fn, params, packer):
    start = time.time()
    dim = init_vec.size
    numP = params["numParticles"]; maxI = params["maxIterPSO"]
    c1, c2 = params["c1"], params["c2"]
    beta0, beta1 = params["beta_start"], params["beta_end"]
    reheat_patience, reheat_ratio = 5, 0.3

    Ppos = np.tile(init_vec[None,:], (numP,1)) + 0.1*np.random.standard_normal((numP, dim))
    Ppos[0,:] = init_vec.copy()
    for p in range(numP): Ppos[p,:] = packer.clip(Ppos[p,:])

    print(f"[Unified-QPSO] init: numP={numP}, dim={dim} — evaluating initial swarm...")
    Pbest = Ppos.copy(); PbestVal = np.array([eval_fn(Ppos[i,:]) for i in range(numP)])
    Gidx = int(np.argmax(PbestVal)); Gbest = Ppos[Gidx,:].copy(); GbestVal = PbestVal[Gidx]

    no_imp=0
    for it in range(1, maxI+1):
        beta_t = beta0 + (beta1 - beta0) * ((it-1)/max(1, maxI-1))
        mbest = np.mean(Pbest, axis=0)
        vals = np.zeros(numP)
        print(f"[Unified-QPSO] iter {it}/{maxI} — beta={beta_t:.3f} — updating & evaluating...")
        for p in range(numP):
            p_i = (c1/(c1+c2))*Pbest[p,:] + (c2/(c1+c2))*Gbest
            u = np.clip(np.random.random(dim), 1e-12, 1-1e-12)
            sgn = np.where(np.random.random(dim) < 0.5, -1.0, 1.0)
            step = beta_t * np.abs(mbest - Ppos[p,:]) * np.log(1.0/u)
            x_new = p_i + sgn * step
            x_new = packer.clip(x_new)
            Ppos[p,:] = x_new
            vals[p] = eval_fn(x_new)
            if vals[p] > PbestVal[p]: PbestVal[p], Pbest[p,:] = vals[p], x_new.copy()

        mx_idx = int(np.argmax(vals))
        if vals[mx_idx] > GbestVal + 1e-12:
            GbestVal = float(vals[mx_idx]); Gbest = Ppos[mx_idx,:].copy(); no_imp = 0
        else:
            no_imp += 1
        if reheat_patience>0 and no_imp>=reheat_patience and numP>=4:
            order = np.argsort(vals); worst_idx = order[:max(1, int(round(reheat_ratio*numP)))]
            for p in worst_idx:
                noise = 0.5*(np.random.random(dim)-0.5)
                Ppos[p,:] = packer.clip(Gbest + noise)
                Pbest[p,:] = Ppos[p,:]; PbestVal[p] = eval_fn(Ppos[p,:])
            no_imp = 0; print(f"[Unified-QPSO]  reheat applied to {len(worst_idx)} particles.")
        print(f"[Unified-QPSO]  best={GbestVal:.6f}  mean={vals.mean():.6f}  std={vals.std():.6f}")
    print(f"[Unified-QPSO] Total execution time: {time.time()-start:.4f} seconds")
    return Gbest, GbestVal

# -------- PSO（标准粒子群，与给定片段保持一致；仅新增必要注释，不改动原有流程/打印） --------
def run_pso_unified(init_vec, eval_fn, params, packer):
    start_time = time.time()
    dim = init_vec.size
    numP, maxI = params['numParticles'], params['maxIterPSO']
    w0, c1, c2 = params['w_inertia'], params['c1'], params['c2']

    # 初始化群体与速度（按给定片段使用 global_rng.rand/randn 接口）
    Ppos = np.tile(init_vec, (numP,1)) + 0.1*global_rng.randn(numP, dim)
    Ppos[0] = init_vec.copy()
    Ppos = np.array([packer.clip(Ppos[p]) for p in range(numP)])
    Pvel = np.zeros_like(Ppos)

    print(f"[Unified] init: numP={numP}, dim={dim} — evaluating initial swarm...", flush=True)
    Pbest = Ppos.copy(); PbestVal = np.array([eval_fn(Ppos[i]) for i in range(numP)])
    Gidx = int(np.argmax(PbestVal)); Gbest = Ppos[Gidx].copy(); GbestVal = float(PbestVal[Gidx])

    for it in range(maxI):
        print(f"[Unified] iter {it+1}/{maxI} — evaluating swarm...", flush=True)
        vals = np.array([eval_fn(Ppos[p]) for p in range(numP)])

        # 更新个体/全局最好
        for p in range(numP):
            if vals[p] > PbestVal[p]:
                PbestVal[p], Pbest[p] = float(vals[p]), Ppos[p].copy()
        idx = int(np.argmax(vals))
        if vals[idx] > GbestVal:
            GbestVal, Gbest = float(vals[idx]), Ppos[idx].copy()

        print(f"[Unified] PSO iter {it+1}/{maxI}: {GbestVal:.6e}", flush=True)

        # 速度与位置更新（按片段使用 global_rng.rand）
        r1, r2 = global_rng.rand(numP,dim), global_rng.rand(numP,dim)
        Pvel = w0*Pvel + c1*r1*(Pbest-Ppos) + c2*r2*(Gbest-Ppos)
        Ppos = np.array([packer.clip(Ppos[p]+Pvel[p]) for p in range(numP)])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"[Unified-PSO] Total execution time: {execution_time:.4f} seconds", flush=True)
    return Gbest, GbestVal

class UnifiedPack:
    def __init__(self, params):
        self.Jm   = params["Jm"]
        self.Nt_m = params["Nt_micro"]
        self.Nt_M = params["Nt_macro"]
        self.area = params["area"]
        self.MRm  = params["MA_range_micro"]
        self.MRmM = params["MA_range_macro"]
        self.fc_m = params["fc"]
        self.fc_M = params.get("fc_macro", params["fc"])
        self.Xi   = int(params.get("Xi", 2))

    def pack(self, macro, micro, ma_slots_m, ma_slots_M):
        return np.concatenate([macro.ravel(), micro.ravel(),
                               ma_slots_m.ravel(), ma_slots_M.ravel()])

    def unpack(self, v):
        Jm, Nt_m, Nt_M, Xi = self.Jm, self.Nt_m, self.Nt_M, self.Xi
        off = 0
        macro = v[off:off+2]; off += 2
        micro = v[off:off+2*Jm].reshape(Jm, 2); off += 2*Jm
        ma_slots_m = v[off:off+Xi*Jm*Nt_m*2].reshape(Xi, Jm, Nt_m, 2); off += Xi*Jm*Nt_m*2
        ma_slots_M = v[off:off+Xi*Nt_M*2].reshape(Xi, Nt_M, 2)
        return macro, micro, ma_slots_m, ma_slots_M

    def _pairwise_ok(self, pts, lam):
        D = pts[:,None,:] - pts[None,:,:]
        d2 = np.sum(D*D, axis=-1)
        iu = np.triu_indices(len(pts), 1)
        return np.all(d2[iu] >= (lam/2.0)**2 - 1e-12)

    def clip(self, v):
        macro, micro, ma_slots_m, ma_slots_M = self.unpack(v.copy())
        Jm, Xi = self.Jm, self.Xi
        area, MRm, MRmM = self.area, self.MRm, self.MRmM
        macro = np.clip(macro, 0.0, area)
        micro = np.clip(micro, 0.0, area)
        ma_slots_m = np.clip(ma_slots_m, -MRm, MRm)
        ma_slots_M = np.clip(ma_slots_M, -MRmM, MRmM)

        lam_m = C/self.fc_m; lam_M = C/self.fc_M

        for s in range(Xi):
            for jj in range(Jm):
                arr = ma_slots_m[s]
                pts = micro[jj][None,:] + arr[jj]
                if not self._pairwise_ok(pts, lam_m):
                    req = lam_m/2.0
                    max_push_iter = 100
                    for itt in range(max_push_iter):
                        pts = micro[jj][None,:] + arr[jj]
                        diff = pts[:,None,:] - pts[None,:,:]
                        d2 = (diff**2).sum(axis=2)
                        iu = np.triu_indices(arr.shape[1], k=1)
                        d = np.sqrt(d2[iu])
                        viol_idx = np.where(d < req)[0]
                        if viol_idx.size == 0:
                            break
                        order = np.argsort(d[viol_idx])
                        for idx in viol_idx[order]:
                            i, j = iu[0][idx], iu[1][idx]
                            pi, pj = pts[i].copy(), pts[j].copy()
                            delta = pj - pi
                            norm = float(np.linalg.norm(delta))
                            if norm < 1e-12:
                                angle = (stable_seed("repel", s, jj, i, j) % 3600)/3600.0 * 2*np.pi
                                u = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                            else:
                                u = delta / norm
                            need = (req - norm)
                            if need <= 0:
                                continue
                            shift = 0.5 * need * u
                            arr[jj,i] -= shift
                            arr[jj,j] += shift
                            arr[jj] = np.clip(arr[jj], -MRm, MRm)
                    pts = micro[jj][None,:] + arr[jj]
                    if not self._pairwise_ok(pts, lam_m):
                        print(f"[CLIP] MICRO slot{s} cell{jj}: pairwise repair failed after max_push_iter={max_push_iter}, fallback to circle reset.")
                        Nt = self.Nt_m
                        ang = np.linspace(0, 2*np.pi, Nt, endpoint=False)
                        Rneed = (lam_m/4.0)/np.sin(np.pi/Nt)
                        R = min(MRm, max(Rneed, 1e-3))
                        new = np.column_stack([R*np.cos(ang), R*np.sin(ang)])
                        arr[jj,:,:] = np.clip(new, -MRm, MRm)
                ma_slots_m[s] = arr

        def macro_fix(ma_M, slot_idx):
            pts = macro[None, :] + ma_M
            if not self._pairwise_ok(pts, lam_M):
                req = lam_M / 2.0
                max_push_iter = 100
                for itt in range(max_push_iter):
                    pts = macro[None, :] + ma_M
                    diff = pts[:, None, :] - pts[None, :, :]
                    d2 = (diff ** 2).sum(axis=2)
                    iu = np.triu_indices(ma_M.shape[0], k=1)
                    d = np.sqrt(d2[iu])
                    viol_idx = np.where(d < req)[0]
                    if viol_idx.size == 0:
                        break
                    order = np.argsort(d[viol_idx])
                    for idx in viol_idx[order]:
                        i, j = iu[0][idx], iu[1][idx]
                        pi, pj = pts[i].copy(), pts[j].copy()
                        delta = pj - pi
                        norm = float(np.linalg.norm(delta))
                        if norm < 1e-12:
                            angle = (stable_seed("repelM", slot_idx, i, j) % 3600) / 3600.0 * 2 * np.pi
                            u = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                        else:
                            u = delta / norm
                        need = (req - norm)
                        if need <= 0:
                            continue
                        shift = 0.5 * need * u
                        ma_M[i] -= shift
                        ma_M[j] += shift
                        ma_M = np.clip(ma_M, -MRmM, MRmM)
                pts = macro[None, :] + ma_M
                if not self._pairwise_ok(pts, lam_M):
                    print(f"[CLIP] MACRO slot{slot_idx}: pairwise repair failed after max_push_iter={max_push_iter}, fallback to circle reset.")
                    Nt = self.Nt_M
                    ang = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
                    Rneed = (lam_M / 4.0) / np.sin(np.pi / Nt)
                    R = min(MRmM, max(Rneed, 1e-3))
                    new = np.column_stack([R * np.cos(ang), R * np.sin(ang)])
                    ma_M[:, :] = np.clip(new, -MRmM, MRmM)
            return ma_M

        for s in range(Xi):
            ma_slots_M[s] = macro_fix(ma_slots_M[s], s)

        return self.pack(macro, micro, ma_slots_m, ma_slots_M)

def main():
    XI = int(os.getenv("XI", "2"))
    if XI < 2 or XI % 2 != 0:
        raise ValueError("For v13, XI must be an even integer >= 2, e.g., 2,4,6,8,10.")
    rep = XI // 2
    slot_types = [0, 1] * rep   # 0=day, 1=night, alternating
    slot_weights = np.array([0.3 / rep, 0.7 / rep] * rep, dtype=float)

    UE_day, UE_night = 15, 30
    area = 100.0
    micro_init = np.array([[20,20],[80,20],[50,80]], float)
    macro_init = np.array([50,50], float)

    params = dict(
        area=area, Jm=3, JM=1,
        Nt_micro=8, Nt_macro=4, Nr_ue=2, sigma2=1e-9,
        fc=3.5e9, fc_macro=2.0e9,
        B_micro=100e6, B_macro=20e6,
        M_micro=8, M_macro=4,
        max_iter_wmmse=4, tol_wmmse=1e-3,
        numParticles=50, maxIterPSO=50,
        w_inertia=0.7, c1=1.8, c2=1.2, beta_start=1.15, beta_end=0.7,
        MA_range_micro=0.6, MA_range_macro=0.8,
        Pj=np.array([80.0,80.0,80.0,40.0]),
        L_paths=10,
        Xi=XI,
    )

    FAST_MODE = os.getenv("FAST_MODE","0") == "1"
    if os.getenv("PSO_PARTICLES"): params["numParticles"] = int(os.getenv("PSO_PARTICLES"))
    if os.getenv("PSO_ITERS"):     params["maxIterPSO"]   = int(os.getenv("PSO_ITERS"))
    if FAST_MODE:
        params["numParticles"] = min(params["numParticles"], 4)
        params["maxIterPSO"]   = min(params["maxIterPSO"], 3)
    print(f"[Config] Xi={XI}, slot_types={slot_types}, slot_weights={slot_weights.tolist()}")
    print(f"[Config] numParticles={params['numParticles']}, maxIterPSO={params['maxIterPSO']}, HAS_CVXPY={string_tf(HAS_CVXPY)}")

    UE1 = np.random.random((UE_day,3)) * area
    MC_UE_NIGHT_LIST = [36]
    UE2_base = np.random.random((max(MC_UE_NIGHT_LIST),3)) * area

    SCENARIOS = []
    EPOCHS = list(range(1, 11))
    Jm, Nt_m, Nt_M = params["Jm"], params["Nt_micro"], params["Nt_macro"]
    lam_m = C/params["fc"]; lam_M = C/params["fc_macro"]
    lam_m_eff = INIT_SPACING_MULT_MICRO * lam_m
    lam_M_eff = INIT_SPACING_MULT_MACRO * lam_M

    def _upa_grid(Nt, lam):
        Nx = int(np.floor(np.sqrt(Nt))); Ny = int(np.ceil(Nt/max(1, Nx)))
        dx = lam/2.0; dy = lam/2.0
        xs = (np.arange(Nx) - (Nx-1)/2.0) * dx
        ys = (np.arange(Ny) - (Ny-1)/2.0) * dy
        X, Y = np.meshgrid(xs, ys)
        return np.column_stack([X.ravel(), Y.ravel()])[:Nt,:]

    grid_m = _upa_grid(Nt_m, lam_m_eff)
    grid_M = _upa_grid(Nt_M, lam_M_eff)
    MRm, MRmM = params["MA_range_micro"], params["MA_range_macro"]

    for _E in EPOCHS:
        rng_geo   = np.random.RandomState(stable_seed_noepoch("GEO",   _E))
        rng_shift = np.random.RandomState(stable_seed_noepoch("SHIFT", _E))

        micro_init_e = rng_geo.random((Jm,2)) * area
        macro_init_e = rng_geo.random(2) * area

        ma_init_m_e2 = np.zeros((Jm, Nt_m, 2), float)
        for j in range(Jm):
            t = (rng_shift.random(2)*2 - 1.0) * MRm
            ma_init_m_e2[j,:,:] = grid_m + t[None,:]
        tM = (rng_shift.random(2)*2 - 1.0) * MRmM
        ma_init_M_e2 = grid_M + tM[None,:]

        SCENARIOS.append(dict(epoch=_E,
                              micro_init=micro_init_e,
                              macro_init=macro_init_e,
                              ma_init_m=ma_init_m_e2,
                              ma_init_M=ma_init_M_e2))

    import hashlib
    def _hash_scenarios(batch):
        h = hashlib.sha256()
        h.update(str(len(batch)).encode())
        for s in batch:
            h.update(int(s['epoch']).to_bytes(8, 'little', signed=True))
            for k in ('micro_init','macro_init','ma_init_m','ma_init_M'):
                a = np.asarray(s[k], dtype=np.float64)
                h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.hexdigest()
    _SC_HASH = _hash_scenarios(SCENARIOS)
    print(f"[MC] SCENARIOS hash (sha256/16) = {_SC_HASH[:16]}, count={len(SCENARIOS)}")

    def _slot_epoch(base_epoch, slot_idx):
        return int(base_epoch) * 1000 + (slot_idx + 1)

    MC_ue_averages = []
    ALL_MC_RESULTS = []

    for _MC_UN in MC_UE_NIGHT_LIST:
        params["B_micro"] = 100e6
        vals_this_UN = []

        for _MC_EPOCH in range(1, len(EPOCHS)+1):
            print(f"[MC] Snapshot {_MC_EPOCH}/{len(EPOCHS)} | Xi={XI}")
            global CHAN_EPOCH
            CHAN_EPOCH = SCENARIOS[_MC_EPOCH-1]['epoch']
            params["chan_epoch"] = CHAN_EPOCH

            micro_init = SCENARIOS[_MC_EPOCH-1]['micro_init']
            macro_init = SCENARIOS[_MC_EPOCH-1]['macro_init']
            ma_init_m  = SCENARIOS[_MC_EPOCH-1]['ma_init_m']
            ma_init_M  = SCENARIOS[_MC_EPOCH-1]['ma_init_M']
            UE2 = UE2_base[:_MC_UN, :]

            # slot-wise replicated traffic states, but each slot has its own MA variables
            slot_ues = [UE1 if t == 0 else UE2 for t in slot_types]
            ma_init_m_slots = np.repeat(ma_init_m[None, ...], XI, axis=0)
            ma_init_M_slots = np.repeat(ma_init_M[None, ...], XI, axis=0)

            packer = UnifiedPack(params)
            init_vec = packer.pack(macro_init, micro_init, ma_init_m_slots, ma_init_M_slots)
            eval_unified = lambda v: evalUnified(v, slot_weights, slot_ues, params, packer, use_cvx=True,
                                                 slot_epoch_base=SCENARIOS[_MC_EPOCH-1]['epoch'])

            _t0 = time.time()
            best_vec, _ = run_qpso_unified(init_vec, eval_unified, params, packer)
            _elapsed = time.time() - _t0

            best_macro, best_mic, ma_slots_m, ma_slots_M = packer.unpack(best_vec)

            slot_objs = []
            slot_rates = []
            old_epoch = CHAN_EPOCH
            for s in range(XI):
                CHAN_EPOCH = _slot_epoch(SCENARIOS[_MC_EPOCH-1]['epoch'], s)
                params['chan_epoch'] = CHAN_EPOCH
                _, _, _, _, obj_s, _ = inner_wmmse_cvxp(slot_ues[s], best_mic, best_macro,
                                                        ma_slots_m[s], ma_slots_M[s], params, use_cvx=True)
                slot_objs.append(float(obj_s))
                slot_rates.append(float(slot_weights[s]) * float(obj_s))
            CHAN_EPOCH = old_epoch
            params['chan_epoch'] = CHAN_EPOCH
            totalVal = float(np.sum(slot_rates))

            vals_this_UN.append(totalVal)
            ALL_MC_RESULTS.append((_MC_UN, _MC_EPOCH, totalVal))

            def _all_zero(arr, tol=1e-12):
                return np.all(np.abs(arr) < tol)

            for s in range(XI):
                if _all_zero(ma_slots_M[s]):
                    print(f"[Check] Macro (slot {s}) MA offsets fell back to origin (all zeros).")
                for jj in range(Jm):
                    if _all_zero(ma_slots_m[s, jj]):
                        print(f"[Check] Micro #{jj} (slot {s}) MA offsets fell back to origin (all zeros).")

            print(f"[MC-Run] Night UE {_MC_UN} — snapshot {_MC_EPOCH}/{len(EPOCHS)} — elapsed {_elapsed:.4f}s — weighted sum-rate {totalVal:.6e}")
            print(f"[MC-Run] slot raw objs = {[float(x) for x in slot_objs]}")
            print(f"[MC-Run] slot weighted objs = {[float(x) for x in slot_rates]}")

            # representative plotting: first day-like slot and first night-like slot
            try:
                import matplotlib.pyplot as plt

                day_slot = slot_types.index(0)
                nig_slot = slot_types.index(1)

                CHAN_EPOCH = _slot_epoch(SCENARIOS[_MC_EPOCH-1]['epoch'], day_slot)
                params['chan_epoch'] = CHAN_EPOCH
                R_day, a_day, _, _, _, _ = inner_wmmse_cvxp(UE1, best_mic, best_macro,
                                                            ma_slots_m[day_slot], ma_slots_M[day_slot],
                                                            params, use_cvx=True)

                CHAN_EPOCH = _slot_epoch(SCENARIOS[_MC_EPOCH-1]['epoch'], nig_slot)
                params['chan_epoch'] = CHAN_EPOCH
                R_nig, a_nig, _, _, _, _ = inner_wmmse_cvxp(UE2, best_mic, best_macro,
                                                            ma_slots_m[nig_slot], ma_slots_M[nig_slot],
                                                            params, use_cvx=True)

                CHAN_EPOCH = old_epoch
                params['chan_epoch'] = CHAN_EPOCH

                def _plot_scene_with_assoc(title, micro_centers, macro_center, ma_m, ma_M, ue, a_int, R_speff, params):
                    import numpy as _np
                    plt.figure(figsize=(5.8,5.8))
                    plt.scatter(macro_center[0], macro_center[1], marker='^', s=90, label='Macro Center')
                    for j in range(micro_centers.shape[0]):
                        plt.scatter(micro_centers[j,0], micro_centers[j,1], marker='s', s=70, label='Micro Center' if j==0 else None)
                    M_pts = macro_center[None,:] + ma_M
                    plt.scatter(M_pts[:,0], M_pts[:,1], s=18, label='Macro MAs')
                    for j in range(micro_centers.shape[0]):
                        pts = micro_centers[j][None,:] + ma_m[j]
                        plt.scatter(pts[:,0], pts[:,1], s=12, label=f'Micro{j} MAs' if j==0 else None)

                    Jm = params["Jm"]; JM = params["JM"]; J = Jm + JM
                    Bvec = _np.concatenate([_np.full(Jm, params["B_micro"]), _np.full(JM, params["B_macro"])])
                    bs_to_ues = {j: [] for j in range(J)}
                    for k in range(a_int.shape[1]):
                        col = a_int[:, k]
                        if _np.sum(col) != 0:
                            j = int(_np.argmax(col)); bs_to_ues[j].append(k)
                    print(f"[Assoc - {title}] Per-BS served UE indices:")
                    for j in range(J):
                        label = f"Micro#{j}" if j < Jm else "Macro#0"
                        print(f"  {label:<8} -> {bs_to_ues[j]}")
                    row_loads = a_int.sum(axis=1)
                    cap_vec = [params["M_micro"]]*Jm + [params["M_macro"]]*JM
                    print("[SANITY] loads=", row_loads.tolist(), "cap=", cap_vec)
                    print(f"[Rates - {title}] Per-UE rates (bps):")
                    for k in range(len(ue)):
                        col = a_int[:, k]
                        if _np.sum(col) == 1:
                            j = int(_np.argmax(col))
                            rate_k = float(Bvec[j] * R_speff[j, k])
                        else:
                            rate_k = 0.0
                        print(f"  UE[{k:02d}] = {rate_k:.6e}")
                    for k in range(len(ue)):
                        col = a_int[:, k]
                        if _np.sum(col) == 0:
                            plt.text(ue[k,0]+0.5, ue[k,1]+0.5, str(k), fontsize=8)
                            continue
                        j = int(_np.argmax(col))
                        p_bs = micro_centers[j] if j < Jm else macro_center
                        plt.plot([p_bs[0], ue[k,0]], [p_bs[1], ue[k,1]], linewidth=0.7, alpha=0.7)
                        plt.text(ue[k,0]+0.5, ue[k,1]+0.5, str(k), fontsize=8)
                    plt.scatter(ue[:,0], ue[:,1], s=14, marker='.', c='k', label='UE')
                    plt.title(title); plt.xlim(0, area); plt.ylim(0, area); plt.gca().set_aspect('equal', 'box')
                    plt.legend(loc='best', fontsize=8); plt.grid(True, ls='--', alpha=0.4)

                _plot_scene_with_assoc(f'Day-like Slot #{day_slot}', best_mic, best_macro,
                                       ma_slots_m[day_slot], ma_slots_M[day_slot], UE1, a_day, R_day, params)
                _plot_scene_with_assoc(f'Night-like Slot #{nig_slot}', best_mic, best_macro,
                                       ma_slots_m[nig_slot], ma_slots_M[nig_slot], UE2, a_nig, R_nig, params)
                plt.show()
            except Exception:
                pass

            if _MC_EPOCH == len(EPOCHS):
                avg_val = float(np.mean(vals_this_UN)) if len(vals_this_UN)>0 else float('nan')
                print(f"[MC] Night UE {_MC_UN} — {len(EPOCHS)}-run average weighted sum-rate: {avg_val}")
                MC_ue_averages.append((_MC_UN, avg_val))

    if len(MC_ue_averages) == len(MC_UE_NIGHT_LIST):
        print("[MC] Summary: average weighted sum-rates (bps)")
        for UN, avgU in MC_ue_averages:
            print(f"  Night UE {UN}: avg {avgU}")

    print("[MC] Detailed per-run results (nightUE, epoch, weighted_sum_rate_bps):")
    for (pw, ep, val) in ALL_MC_RESULTS:
        print(f"  ({pw:>6}, {ep:2d})  {val:.6e}")


if __name__ == "__main__":
    main() 
