import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous

# coding=utf-8
rcParams['font.family'] = ['sans-serif']
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False

# ============================================================
# Device
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# ============================================================
# System parameters (from text 2)
# ============================================================
Nt = 15
Nr = 15
M = 30
L = 7  # 3 DU + 2 RU + 2 TU

sigma2 = 1e-9
Pmax = 1.0

# Communication / computation parameters
B = 1.0
delta_t = 0.5
c_l = 1e3
F_B = 5e3
P_user_max =1
F_loc_max = 1e3

# Sensing
Gamma_s = 3000.0

# Penalty weights
mu_s = 1.0
mu_c = 1.0

# UAV trajectory
H = 50.0
x_c = -35.0
y_c = 0.0
R_uav = 18.0
phi0 = np.pi
N_slot = 25

# Channel parameters
beta0 = 10 ** (-20 / 10)
alpha_r = 2.2
alpha_d = 3.5
alpha0 = 1e-4

# PPO / PGA parameters
ACTION_DIM = 2 * M + 3 * L + 2

PGA_RHO_LIST = [8e-5 for _ in range(15)]

PGA_ITERS = len(PGA_RHO_LIST)
PGA_RHO = PGA_RHO_LIST[0]

# State scaling
POS_SCALE = 100.0
CHANNEL_SCALE = 1e4

# ============================================================
# Plot utilities (keep first-text style)
# ============================================================
def plot_evaluation_rewards(evaluate_rewards, save_path=None):
    plt.figure(figsize=(12, 8))
    eval_episodes = range(1, len(evaluate_rewards) + 1)
    plt.plot(eval_episodes, evaluate_rewards, 'b-', linewidth=2, label='评估奖励', alpha=0.8)
    plt.scatter(eval_episodes, evaluate_rewards, color='red', s=30, alpha=0.6, label='评估点')

    if len(evaluate_rewards) >= 5:
        moving_avg = []
        for i in range(len(evaluate_rewards)):
            if i < 4:
                moving_avg.append(np.mean(evaluate_rewards[:i + 1]))
            else:
                moving_avg.append(np.mean(evaluate_rewards[i - 4:i + 1]))
        plt.plot(eval_episodes, moving_avg, 'g--', linewidth=2, label='移动平均（窗口=5）', alpha=0.7)

    plt.xlabel('评估次数', fontsize=14)
    plt.ylabel('评估奖励', fontsize=14)
    plt.title('UAV-STAR-RIS PPO-PGA训练评估奖励曲线', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    if evaluate_rewards:
        y_min = min(evaluate_rewards)
        y_max = max(evaluate_rewards)
        y_range = y_max - y_min if y_max > y_min else 1.0
        plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    if len(evaluate_rewards) > 10:
        plt.xticks(range(0, len(evaluate_rewards) + 1, max(1, len(evaluate_rewards) // 10)))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"评估奖励图已保存至: {save_path}")
        data_save_path = save_path.rsplit('.', 1)[0] + '_data.csv'
        np.savetxt(data_save_path, evaluate_rewards, delimiter=',', header='Evaluation_Reward', comments='')
        print(f"评估奖励数据已保存至: {data_save_path}")
    plt.show()


def plot_training_progress(evaluate_rewards, total_steps, eval_freq, save_path=None):
    if len(evaluate_rewards) == 0:
        return

    plt.figure(figsize=(12, 8))
    training_steps = [i * eval_freq for i in range(1, len(evaluate_rewards) + 1)]
    plt.plot(training_steps, evaluate_rewards, 'b-o', linewidth=2, markersize=4, label='评估奖励')

    if len(evaluate_rewards) >= 3:
        z = np.polyfit(training_steps, evaluate_rewards, 1)
        p = np.poly1d(z)
        plt.plot(training_steps, p(training_steps), 'r--', alpha=0.8, label=f'趋势线 (斜率: {z[0]:.4f})')

    plt.xlabel('训练步数', fontsize=14)
    plt.ylabel('评估奖励', fontsize=14)
    plt.title(f'训练进度 (总步数: {total_steps})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    stats_text = (
        f"统计信息:\n"
        f"评估次数: {len(evaluate_rewards)}\n"
        f"最高奖励: {max(evaluate_rewards):.4f}\n"
        f"平均奖励: {np.mean(evaluate_rewards):.4f}\n"
        f"最后奖励: {evaluate_rewards[-1]:.4f}"
    )
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                 verticalalignment='top', fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练进度图已保存至: {save_path}")
    plt.show()


def plot_episode_rewards(episode_rewards_list, save_path=None):
    if not episode_rewards_list:
        print('没有可用的episode_rewards数据')
        return {}

    episode_total_rewards = [float(np.sum(rewards)) for rewards in episode_rewards_list]

    plt.figure(figsize=(12, 8))
    episodes = range(1, len(episode_total_rewards) + 1)
    plt.plot(episodes, episode_total_rewards, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode总奖励随迭代次数的变化')
    plt.grid(True, alpha=0.3)

    total_reward = float(np.sum(episode_total_rewards))
    stats_text = (
        f"统计信息:\n"
        f"总Episode数: {len(episode_total_rewards)}\n"
        f"总时隙数: {sum(len(rewards) for rewards in episode_rewards_list)}\n"
        f"总奖励: {total_reward:.4f}"
    )
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                 verticalalignment='top', fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Episode奖励图已保存至: {save_path}")
    plt.show()

    return {
        'total_episodes': len(episode_rewards_list),
        'total_steps': sum(len(rewards) for rewards in episode_rewards_list),
        'total_reward': total_reward,
    }


def save_episode_data(episode_rewards_list, filename='episode_rewards_data.csv'):
    import pandas as pd
    data = []
    for episode_idx, rewards in enumerate(episode_rewards_list, 1):
        for step_idx, reward in enumerate(rewards, 1):
            data.append({'episode': episode_idx, 'step': step_idx, 'reward': float(reward)})
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Episode奖励数据已保存至: {filename}")

    summary_data = []
    for episode_idx, rewards in enumerate(episode_rewards_list, 1):
        summary_data.append({'episode': episode_idx, 'total_reward': float(np.sum(rewards)), 'steps': len(rewards)})
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(filename.replace('.csv', '_summary.csv'), index=False)
    print(f"Episode汇总数据已保存至: {filename.replace('.csv', '_summary.csv')}")
    return df, df_summary


def print_episode_summary(episode, env, episode_rewards, uav_trajectory, ris_history, user_alloc_history):
    print('\n' + '=' * 60)
    print(f'Episode {episode} 完成 - 总奖励: {sum(episode_rewards):.4f}')
    print(f'总时隙数: {len(episode_rewards)}')
    print('4. 性能指标:')
    print(f'   总奖励: {sum(episode_rewards):.4f}')
    print(f'   平均时隙奖励: {np.mean(episode_rewards):.4f}')
    print('=' * 60 + '\n')

# ============================================================
# Basic utilities
# ============================================================
def steering_vector(N, angle):
    idx = torch.arange(N, device=device).float()
    return torch.exp(1j * np.pi * idx * torch.cos(angle)).reshape(-1, 1)


def array_response(num_ant, phi):
    idx = torch.arange(num_ant, device=device).float()
    return torch.exp(-1j * 2 * np.pi * 0.5 * idx * phi).reshape(-1, 1)


def get_uav_position(n=0, N=N_slot):
    angle_uav = 2 * np.pi * n / N + phi0
    q_uav_x = x_c + R_uav * np.cos(angle_uav)
    q_uav_y = y_c + R_uav * np.sin(angle_uav)
    return torch.tensor([q_uav_x, q_uav_y, H], dtype=torch.float32, device=device)


def get_ris_position():
    return torch.tensor([0.0, 0.0, 25.0], dtype=torch.float32, device=device)


def get_target_position():
    return torch.tensor([-40.0, 0.0, 0.0], dtype=torch.float32, device=device)


def proj_power(w):
    norm = torch.norm(w)
    factor = torch.minimum(torch.tensor(1.0, device=device),
                           torch.sqrt(torch.tensor(Pmax, device=device)) / (norm + 1e-12))
    return factor * w


def complex_to_real_vec(x):
    return torch.cat([x.real.reshape(-1), x.imag.reshape(-1)], dim=0)

# ============================================================
# PGA inner solver (from text 2, no DU)
# ============================================================
def update_ul(h_eff, A0, w, Pl):
    Ry = Pl * (h_eff @ h_eff.conj().T) + A0 @ w @ w.conj().T @ A0.conj().T + sigma2 * torch.eye(Nr, dtype=torch.cfloat, device=device)
    ul = torch.linalg.solve(Ry, h_eff)
    ul = ul / (torch.norm(ul) + 1e-12)
    return ul


def update_u(A0, w):
    Rs = A0 @ w @ w.conj().T @ A0.conj().T + sigma2 * torch.eye(Nr, dtype=torch.cfloat, device=device)
    u = torch.linalg.solve(Rs, A0 @ w)
    u = u / (torch.norm(u) + 1e-12)
    return u


def compute_sinr(ul, h, A0, w, Pl):
    hu = ul.conj().T @ h
    signal = Pl * torch.abs(hu) ** 2
    sensing_term = ul.conj().T @ (A0 @ w @ w.conj().T @ A0.conj().T) @ ul
    noise_term = sigma2 * (ul.conj().T @ ul)
    interference = sensing_term + noise_term
    return (signal / (interference + 1e-30)).real.squeeze()


def compute_sensing_snr(u, A0, w):
    num = u.conj().T @ A0 @ w @ w.conj().T @ A0.conj().T @ u
    den = sigma2 * torch.norm(u) ** 2
    return (num / (den + 1e-12)).real.squeeze()


def compute_components(w, h_eff_list, A0, Pl_list, tau_list, f_loc_list):
    C_off_list = []
    C_loc_list = []
    sinr_list = []

    ul_list = [update_ul(h_eff_list[l], A0, w, Pl_list[l]) for l in range(L)]
    u = update_u(A0, w)

    F_comm = torch.tensor(0.0, device=device)
    for l in range(L):
        sinr_l = compute_sinr(ul_list[l], h_eff_list[l], A0, w, Pl_list[l])
        sinr_list.append(sinr_l)
        C_off_l = c_l * tau_list[l] * delta_t * B * torch.log2(1.0 + sinr_l)
        C_loc_l = delta_t * f_loc_list[l]
        C_off_list.append(C_off_l)
        C_loc_list.append(C_loc_l)
        F_comm = F_comm + C_off_l + C_loc_l

    gamma_s = compute_sensing_snr(u, A0, w)
    P_s = -torch.relu(torch.tensor(Gamma_s, device=device) - gamma_s)

    P_c = torch.tensor(0.0, device=device)
    comp_margins = []
    for l in range(L):
        margin_l = tau_list[l] * delta_t * F_B - C_off_list[l]
        comp_margins.append(margin_l)
        P_c = P_c + torch.minimum(torch.tensor(0.0, device=device), margin_l)

    reward = F_comm + mu_s * P_s + mu_c * P_c

    return {
        'reward': reward,
        'F_comm': F_comm,
        'C_off_list': C_off_list,
        'C_loc_list': C_loc_list,
        'sinr_list': sinr_list,
        'gamma_s': gamma_s,
        'P_s': P_s,
        'P_c': P_c,
        'comp_margins': comp_margins,
        'ul_list': ul_list,
        'u': u,
    }


def compute_pga_grad(w, h_eff_list, A0, Pl_list, tau_list):
    grad = torch.zeros_like(w)
    ul_list = [update_ul(h_eff_list[l], A0, w, Pl_list[l]) for l in range(L)]
    u = update_u(A0, w)

    for l in range(L):
        ul = ul_list[l]
        h = h_eff_list[l]
        Pl = Pl_list[l]

        Sl = Pl * torch.abs(ul.conj().T @ h) ** 2
        Dl = ul.conj().T @ (A0 @ w @ w.conj().T @ A0.conj().T + sigma2 * torch.eye(Nr, dtype=torch.cfloat, device=device)) @ ul
        core = A0.conj().T @ ul @ ul.conj().T @ A0 @ w
        grad_gamma = -Sl / (Dl ** 2 + 1e-12) * core

        sinr_l = compute_sinr(ul, h, A0, w, Pl)
        xi = c_l * tau_list[l] * delta_t * B
        grad_Coff_l = (xi / (torch.log(torch.tensor(2.0, device=device)) * (1.0 + sinr_l))) * grad_gamma

        grad = grad + grad_Coff_l

        C_off_l = c_l * tau_list[l] * delta_t * B * torch.log2(1.0 + sinr_l)
        margin_l = tau_list[l] * delta_t * F_B - C_off_l
        if margin_l.item() < 0:
            grad = grad - mu_c * grad_Coff_l

    gamma_s = compute_sensing_snr(u, A0, w)
    if gamma_s.item() <= Gamma_s:
        grad_s = (A0.conj().T @ u @ u.conj().T @ A0 @ w) / (sigma2 * torch.norm(u) ** 2 + 1e-12)
        grad = grad + mu_s * grad_s

    return grad


def traditional_pga_inner(h_eff_list, A0, Pl_list, tau_list,
                          pga_iters=PGA_ITERS, rho_list=PGA_RHO_LIST):
    w = torch.ones(Nt, 1, dtype=torch.cfloat, device=device)
    w = proj_power(w)
    obj_history = []

    if isinstance(rho_list, torch.Tensor):
        rho_seq = rho_list.detach().cpu().numpy().tolist()
    else:
        rho_seq = list(rho_list)

    pga_iters = min(pga_iters, len(rho_seq))

    for k in range(pga_iters):
        dummy_f = torch.zeros(L, device=device)
        comps = compute_components(w, h_eff_list, A0, Pl_list, tau_list, dummy_f)
        obj_history.append(comps['reward'].item())

        grad = compute_pga_grad(w, h_eff_list, A0, Pl_list, tau_list)

        rho_k = float(rho_seq[k])
        w = w + rho_k * grad
        w = proj_power(w)

    dummy_f = torch.zeros(L, device=device)
    comps = compute_components(w, h_eff_list, A0, Pl_list, tau_list, dummy_f)
    obj_history.append(comps['reward'].item())
    return w, obj_history

# ============================================================
# PPO environment (first-text interface, second-text system)
# ============================================================
class UAVSTARRISEnvironment:
    def __init__(self):
        self.device = device
        self.num_users = L
        self.num_ris_elements = M
        self.max_time_slots = N_slot
        self.current_time_slot = 0
        self.N_t = Nt
        self.N_r = Nr
        self.beta0 = beta0
        self.alpha_r = alpha_r
        self.alpha_d = alpha_d
        self.alpha_0 = alpha0
        self.target_pos = np.array([-40.0, 0.0], dtype=np.float32)
        self.ris_position = np.array([0.0, 0.0], dtype=np.float32)
        self.ris_height = 25.0
        self.uav_height = H
        self.trajectory_center = np.array([x_c, y_c], dtype=np.float32)
        self.trajectory_radius = R_uav
        self.time_slot_duration = delta_t
        self.bandwidth = B
        self.max_transmit_power = P_user_max
        self.uav_compute_capacity = F_B
        self.c_l = c_l
        self.C_l = c_l

        self.direct_user_1 = np.array([-25, 16], dtype=np.float32)
        self.direct_user_2 = np.array([-48, -20], dtype=np.float32)
        self.direct_user_3 = np.array([-30, -8], dtype=np.float32)
        self.reflect_user_1 = np.array([-10, 10], dtype=np.float32)
        self.reflect_user_2 = np.array([-8, -8], dtype=np.float32)
        self.transmit_user_1 = np.array([5, 5], dtype=np.float32)
        self.transmit_user_2 = np.array([3, -3], dtype=np.float32)

        self.state_dim = self._build_state_dim()
        self.action_dim = ACTION_DIM
        self.last_info = None
        self.reset()

    def _build_state_dim(self):
        env0 = self._generate_base_env(0)
        return int(self._build_state(env0).numel())

    def _generate_base_env(self, n=0):
        UAV_pos = get_uav_position(n=n, N=N_slot)
        RIS_pos = get_ris_position()
        target_pos = get_target_position()

        DU_pos = torch.tensor([[-25, 16, 0], [-48, -20, 0], [-30, -8, 0]], device=device).float()
        RU_pos = torch.tensor([[-10, 10, 0], [-8, -8, 0]], device=device).float()
        TU_pos = torch.tensor([[5, 5, 0], [3, -3, 0]], device=device).float()
        user_all = torch.cat([DU_pos, RU_pos, TU_pos], dim=0)

        d_u = torch.norm(UAV_pos - RIS_pos)
        phi_M = (UAV_pos[0] - RIS_pos[0]) / (d_u + 1e-12)
        gM = array_response(M, phi_M)
        gNr = array_response(Nr, phi_M)
        Hu = torch.sqrt(torch.tensor(beta0, device=device)) * (d_u ** (-alpha_r / 2)) * (gNr @ gM.conj().T)

        h_ris_list = []
        h_d_list = []
        for i in range(L):
            user = user_all[i]
            d_l = torch.norm(user - RIS_pos)
            phi_l = (RIS_pos[0] - user[0]) / (d_l + 1e-12)
            g_l = array_response(M, phi_l)
            h_l = torch.sqrt(torch.tensor(beta0, device=device)) * (d_l ** (-alpha_r / 2)) * g_l
            h_ris_list.append(h_l)

            if i < 3:
                d_du = torch.norm(UAV_pos - user)
                phi_du = (UAV_pos[0] - user[0]) / (d_du + 1e-12)
                g_du = array_response(Nr, phi_du)
                h_d = torch.sqrt(torch.tensor(beta0, device=device)) * (d_du ** (-alpha_d / 2)) * g_du
            else:
                h_d = torch.zeros(Nr, 1, dtype=torch.cfloat, device=device)
            h_d_list.append(h_d)

        d_s = torch.norm(UAV_pos - target_pos)
        theta = torch.arccos(UAV_pos[2] / (torch.sqrt(d_s ** 2 + UAV_pos[2] ** 2) + 1e-12))
        at = steering_vector(Nt, theta)
        ar = steering_vector(Nr, theta)
        A0 = torch.sqrt(torch.tensor(alpha0, device=device) * d_s ** (-alpha_r)) * (ar @ at.conj().T)

        return {
            'slot': n,
            'UAV_pos': UAV_pos,
            'RIS_pos': RIS_pos,
            'target_pos': target_pos,
            'user_all': user_all,
            'Hu': Hu,
            'h_ris_list': h_ris_list,
            'h_d_list': h_d_list,
            'A0': A0,
        }

    def _build_state(self, env):
        parts = []
        parts.append(env['UAV_pos'].float().reshape(-1) / POS_SCALE)
        parts.append(CHANNEL_SCALE * complex_to_real_vec(env['Hu']))
        for h_l in env['h_ris_list']:
            parts.append(CHANNEL_SCALE * complex_to_real_vec(h_l))
        for h_d in env['h_d_list']:
            parts.append(CHANNEL_SCALE * complex_to_real_vec(h_d))
        return torch.cat(parts, dim=0).float()

    def _parse_action(self, raw_action):
        raw_action = torch.as_tensor(raw_action, dtype=torch.float32, device=device).reshape(-1)
        idx = 0
        lambda_logits = raw_action[idx: idx + 2]
        idx += 2
        lambda_pair = torch.softmax(lambda_logits, dim=0)
        lambda_t = lambda_pair[0].reshape(1)
        lambda_r = lambda_pair[1].reshape(1)

        theta_t_raw = raw_action[idx: idx + M]
        idx += M
        theta_t = 2 * np.pi * torch.sigmoid(theta_t_raw)

        theta_r_raw = raw_action[idx: idx + M]
        idx += M
        theta_r = 2 * np.pi * torch.sigmoid(theta_r_raw)

        P_raw = raw_action[idx: idx + L]
        idx += L
        Pl_list = P_user_max * torch.sigmoid(P_raw)

        tau_raw = raw_action[idx: idx + L]
        idx += L
        tau_list = torch.softmax(tau_raw, dim=0)

        F_raw = raw_action[idx: idx + L]
        idx += L
        f_loc_list = F_loc_max * torch.sigmoid(F_raw)

        phi_t = torch.sqrt(lambda_t + 1e-12) * torch.exp(1j * theta_t)
        phi_r = torch.sqrt(lambda_r + 1e-12) * torch.exp(1j * theta_r)

        return {
            'lambda_t': lambda_t,
            'lambda_r': lambda_r,
            'theta_t': theta_t,
            'theta_r': theta_r,
            'Phi_t': torch.diag(phi_t),
            'Phi_r': torch.diag(phi_r),
            'Pl_list': Pl_list,
            'tau_list': tau_list,
            'f_loc_list': f_loc_list,
        }

    def _build_effective_channels(self, env, action_vars):
        Hu = env['Hu']
        Phi_t = action_vars['Phi_t']
        Phi_r = action_vars['Phi_r']
        h_eff_list = []
        for l in range(L):
            h_l = env['h_ris_list'][l]
            h_d = env['h_d_list'][l]
            if l < 3:
                h_eff = h_d + Hu @ Phi_r @ h_l
            elif l < 5:
                h_eff = Hu @ Phi_r @ h_l
            else:
                h_eff = Hu @ Phi_t @ h_l
            h_eff_list.append(h_eff)
        return h_eff_list

    def _evaluate_slot(self, env, raw_action, pga_iters=PGA_ITERS, pga_rho=PGA_RHO):
        action_vars = self._parse_action(raw_action)
        h_eff_list = self._build_effective_channels(env, action_vars)
        w_opt, pga_obj_history = traditional_pga_inner(
            h_eff_list=h_eff_list,
            A0=env['A0'],
            Pl_list=action_vars['Pl_list'],
            tau_list=action_vars['tau_list'],
            pga_iters=pga_iters,
            rho_list=PGA_RHO_LIST,
        )
        comps = compute_components(
            w=w_opt,
            h_eff_list=h_eff_list,
            A0=env['A0'],
            Pl_list=action_vars['Pl_list'],
            tau_list=action_vars['tau_list'],
            f_loc_list=action_vars['f_loc_list'],
        )
        info = {
            'reward': comps['reward'].item(),
            'F_comm': comps['F_comm'].item(),
            'gamma_s': comps['gamma_s'].item(),
            'P_s': comps['P_s'].item(),
            'P_c': comps['P_c'].item(),
            'lambda_t': action_vars['lambda_t'].item(),
            'lambda_r': action_vars['lambda_r'].item(),
            'mean_power': action_vars['Pl_list'].mean().item(),
            'sum_tau': action_vars['tau_list'].sum().item(),
            'mean_f_loc': action_vars['f_loc_list'].mean().item(),
            'pga_obj_history': pga_obj_history,
            'w_opt': w_opt,
            'action_vars': action_vars,
            'h_eff_list': h_eff_list,
        }
        return comps['reward'].detach(), info

    def reset(self):
        self.current_time_slot = 0
        self.current_env = self._generate_base_env(self.current_time_slot)
        self.uav_position = self.current_env['UAV_pos'][:2].detach().cpu().numpy().tolist()
        self.last_info = None
        return self._build_state(self.current_env).detach().cpu().numpy().astype(np.float32)

    def step(self, action):
        reward, info = self._evaluate_slot(self.current_env, action, pga_iters=PGA_ITERS, pga_rho=PGA_RHO)
        self.last_info = info

        episode_reward = float(reward.item())
        self.current_time_slot += 1
        done = self.current_time_slot >= self.max_time_slots

        if not done:
            self.current_env = self._generate_base_env(self.current_time_slot)
            self.uav_position = self.current_env['UAV_pos'][:2].detach().cpu().numpy().tolist()
            next_state = self._build_state(self.current_env).detach().cpu().numpy().astype(np.float32)
        else:
            next_state = self._build_state(self.current_env).detach().cpu().numpy().astype(np.float32)

        return next_state, episode_reward, done, info

# ============================================================
# PPO evaluation / training (keep first-text method)
# ============================================================
def evaluate_policy(args, env, agent, state_norm, return_components=False):
    """
    Evaluate policy over several episodes.

    When return_components=True, this function also returns the separated reward
    components averaged over evaluation episodes:
        reward = F_comm + mu_s * P_s + mu_c * P_c
    """
    times = 3
    total_reward = 0.0
    component_records = []

    for eval_ep in range(times):
        state = env.reset()
        if args.use_state_norm:
            state = state_norm(state, update=False)

        episode_reward = 0.0
        done = False

        sum_F_comm = 0.0
        sum_P_s = 0.0
        sum_P_c = 0.0
        sum_gamma_s = 0.0
        sum_weighted_P_s = 0.0
        sum_weighted_P_c = 0.0
        steps = 0

        while not done:
            action = agent.evaluate(state)
            if args.policy_dist == 'Beta':
                action = 2 * (action - 0.5)

            next_state, reward, done, info = env.step(action)

            if args.use_state_norm:
                next_state = state_norm(next_state, update=False)

            reward = float(np.real(reward))
            episode_reward += reward

            F_comm_i = float(info['F_comm'])
            P_s_i = float(info['P_s'])
            P_c_i = float(info['P_c'])
            gamma_s_i = float(info['gamma_s'])

            sum_F_comm += F_comm_i
            sum_P_s += P_s_i
            sum_P_c += P_c_i
            sum_gamma_s += gamma_s_i
            sum_weighted_P_s += mu_s * P_s_i
            sum_weighted_P_c += mu_c * P_c_i
            steps += 1

            state = next_state

        reconstructed_reward = sum_F_comm + sum_weighted_P_s + sum_weighted_P_c

        component_records.append({
            'eval_ep_inner': eval_ep + 1,
            'episode_reward': episode_reward,
            'sum_F_comm': sum_F_comm,
            'sum_P_s': sum_P_s,
            'sum_P_c': sum_P_c,
            'sum_mu_s_P_s': sum_weighted_P_s,
            'sum_mu_c_P_c': sum_weighted_P_c,
            'reconstructed_reward': reconstructed_reward,
            'reward_reconstruct_error': episode_reward - reconstructed_reward,
            'avg_slot_reward': episode_reward / max(steps, 1),
            'avg_F_comm': sum_F_comm / max(steps, 1),
            'avg_P_s': sum_P_s / max(steps, 1),
            'avg_P_c': sum_P_c / max(steps, 1),
            'avg_mu_s_P_s': sum_weighted_P_s / max(steps, 1),
            'avg_mu_c_P_c': sum_weighted_P_c / max(steps, 1),
            'avg_gamma_s': sum_gamma_s / max(steps, 1),
            'steps': steps,
        })

        total_reward += episode_reward

    avg_reward = total_reward / times

    if return_components:
        avg_components = {
            'eval_reward': avg_reward,
            'eval_F_comm': float(np.mean([x['sum_F_comm'] for x in component_records])),
            'eval_P_s': float(np.mean([x['sum_P_s'] for x in component_records])),
            'eval_P_c': float(np.mean([x['sum_P_c'] for x in component_records])),
            'eval_mu_s_P_s': float(np.mean([x['sum_mu_s_P_s'] for x in component_records])),
            'eval_mu_c_P_c': float(np.mean([x['sum_mu_c_P_c'] for x in component_records])),
            'eval_reconstructed_reward': float(np.mean([x['reconstructed_reward'] for x in component_records])),
            'eval_reward_reconstruct_error': float(np.mean([x['reward_reconstruct_error'] for x in component_records])),
            'eval_avg_slot_reward': float(np.mean([x['avg_slot_reward'] for x in component_records])),
            'eval_avg_F_comm': float(np.mean([x['avg_F_comm'] for x in component_records])),
            'eval_avg_P_s': float(np.mean([x['avg_P_s'] for x in component_records])),
            'eval_avg_P_c': float(np.mean([x['avg_P_c'] for x in component_records])),
            'eval_avg_mu_s_P_s': float(np.mean([x['avg_mu_s_P_s'] for x in component_records])),
            'eval_avg_mu_c_P_c': float(np.mean([x['avg_mu_c_P_c'] for x in component_records])),
            'eval_gamma_s': float(np.mean([x['avg_gamma_s'] for x in component_records])),
            'eval_times': times,
            'eval_steps_per_episode': int(component_records[0]['steps']) if component_records else 0,
        }
        return avg_reward, avg_components

    return avg_reward


def main(args, env_name, number, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = UAVSTARRISEnvironment()
    eval_env = UAVSTARRISEnvironment()

    args.state_dim = env.state_dim
    args.action_dim = env.action_dim
    args.max_action = 1.0
    args.max_episode_steps = env.max_time_slots
    # max_train_steps 仅用于日志和画图：真正训练终止由 max_episodes 控制。
    args.max_train_steps = int(args.max_episodes * args.max_episode_steps)

    print('环境参数:')
    print(f'状态维度: {args.state_dim}')
    print(f'动作维度: {args.action_dim}')
    print(f'最大动作值: {args.max_action}')
    print(f'最大时隙数: {args.max_episode_steps}')
    print(f'最大Episode数: {args.max_episodes}')
    print(f'总训练步数/时隙数: {args.max_train_steps}')
    print(f'评估频率: 每 {args.evaluate_freq} 个Episode')
    print(f'PGA层数: {PGA_ITERS}, PGA步长: {PGA_RHO}')

    evaluate_num = 0
    evaluate_rewards = []
    evaluate_component_records = []
    total_steps = 0
    episode_count = 0
    episode_rewards_list = []

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    eval_reward = 0.0
    while episode_count < args.max_episodes:
        state = env.reset()
        if args.use_state_norm:
            state = state_norm(state)
        if args.use_reward_scaling:
            reward_scaling.reset()

        episode_steps = 0
        done = False
        episode_rewards = []
        uav_trajectory = []
        ris_phases_history = []
        user_allocations_history = []

        while not done and episode_steps < args.max_episode_steps:
            uav_trajectory.append((env.uav_position[0], env.uav_position[1]))

            a, a_logprob = agent.choose_action(state)
            next_state, r, done, info = env.step(a)

            ris_phases_history.append({
                'reflect_mean': float(np.mean(info['action_vars']['theta_r'].detach().cpu().numpy())),
                'transmit_mean': float(np.mean(info['action_vars']['theta_t'].detach().cpu().numpy())),
            })

            user_alloc = []
            for i in range(env.num_users):
                user_alloc.append({
                    'offload_ratio': info['action_vars']['tau_list'][i].item(),
                    'transmit_power': info['action_vars']['Pl_list'][i].item(),
                    'compute_freq': info['action_vars']['f_loc_list'][i].item(),
                })
            user_allocations_history.append(user_alloc)

            if args.use_state_norm:
                next_state = state_norm(next_state)

            r = float(np.real(r))
            if args.use_reward_scaling:
                r = float(reward_scaling(r))

            episode_rewards.append(r)
            # dw 表示“非时间上限导致的终止”。当前 episode 固定 40 个时隙，
            # 最后一个时隙属于自然截断，不应当作为 dead/win 终止。
            dw = done and ((episode_steps + 1) != args.max_episode_steps)
            replay_buffer.store(state, a, a_logprob, r, next_state, dw, done)

            state = next_state
            episode_steps += 1
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0


        episode_count += 1
        episode_rewards_list.append(episode_rewards.copy())
        total_episode_reward = float(np.sum(episode_rewards)) if episode_rewards else 0.0

        # 按 episode 评估：evaluate_freq=100 表示每 100 个 episode 评估一次。
        if episode_count % args.evaluate_freq == 0:
            evaluate_num += 1
            eval_reward, eval_comps = evaluate_policy(
                args, eval_env, agent, state_norm, return_components=True
            )
            eval_reward = float(np.real(eval_reward))
            evaluate_rewards.append(eval_reward)

            eval_record = {
                'evaluate_num': evaluate_num,
                'episode': episode_count,
                'total_steps': total_steps,
                'mu_s': mu_s,
                'mu_c': mu_c,
            }
            eval_record.update(eval_comps)
            evaluate_component_records.append(eval_record)

            print(
                f'评估次数: {evaluate_num}	'
                f'评估奖励: {eval_reward:.4f}	'
                f'F_comm: {eval_comps["eval_F_comm"]:.4f}	'
                f'mu_s*P_s: {eval_comps["eval_mu_s_P_s"]:.4f}	'
                f'mu_c*P_c: {eval_comps["eval_mu_c_P_c"]:.4f}	'
                f'P_s: {eval_comps["eval_P_s"]:.4f}	'
                f'P_c: {eval_comps["eval_P_c"]:.4f}	'
                f'gamma_s: {eval_comps["eval_gamma_s"]:.4f}	'
                f'Episode: {episode_count}	'
                f'总步数: {total_steps}'
            )

            os.makedirs('./results', exist_ok=True)
            np.save(
                f'./results/PPO_PGA_{args.policy_dist}_seed_{seed}.npy',
                np.array(evaluate_rewards)
            )

            # 额外保存每次评估 reward 的分项，便于判断 reward 上升来自通信项还是惩罚项改善。
            try:
                import pandas as pd
                df_eval_components = pd.DataFrame(evaluate_component_records)
                df_eval_components.to_csv(
                    f'./results/evaluation_reward_components_seed_{seed}.csv',
                    index=False,
                    encoding='utf-8-sig'
                )
                print(
                    f'评估reward分项数据已保存至: '
                    f'./results/evaluation_reward_components_seed_{seed}.csv'
                )
            except Exception as e:
                print(f'保存评估reward分项CSV失败: {e}')

        if episode_rewards:
            print_episode_summary(episode_count, env, episode_rewards, uav_trajectory, ris_phases_history, user_allocations_history)
        print('\n' + '=' * 60)
        print(f'Episode {episode_count} 完成!')
        print(f'总时隙数: {len(episode_rewards)}/{args.max_episode_steps}')
        print(f'所有时隙奖励总和: {total_episode_reward:.6f}')
        print('=' * 60 + '\n')

    final_eval_reward = float(np.real(evaluate_policy(args, eval_env, agent, state_norm)))
    print(f'训练完成! 最终评估奖励: {final_eval_reward:.2f}')

    if evaluate_rewards:
        os.makedirs('./plots', exist_ok=True)
        plot_evaluation_rewards(evaluate_rewards, './plots/final_evaluation_rewards.png')
        plot_training_progress(evaluate_rewards, total_steps, args.evaluate_freq * args.max_episode_steps, './plots/final_training_progress.png')

    if episode_rewards_list:
        stats = plot_episode_rewards(episode_rewards_list, './plots/final_episode_rewards.png')
        save_episode_data(episode_rewards_list, f'./results/episode_rewards_seed_{seed}.csv')
        print('\n=== 训练完成统计 ===')
        print(f"总Episode数: {stats['total_episodes']}")
        print(f"总时隙数: {stats['total_steps']}")
        print(f"总奖励: {stats['total_reward']:.6f}")
        print('=' * 30 + '\n')

    if evaluate_component_records:
        try:
            import pandas as pd
            os.makedirs('./results', exist_ok=True)
            df_eval_components = pd.DataFrame(evaluate_component_records)
            df_eval_components.to_csv(
                f'./results/evaluation_reward_components_seed_{seed}.csv',
                index=False,
                encoding='utf-8-sig'
            )
            print(
                f'最终评估reward分项数据已保存至: '
                f'./results/evaluation_reward_components_seed_{seed}.csv'
            )
        except Exception as e:
            print(f'最终保存评估reward分项CSV失败: {e}')

    os.makedirs('./models', exist_ok=True)
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
    }, f'./models/UAV_STAR_RIS_PPO_PGA_{args.policy_dist}_seed_{seed}_final.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('UAV-STAR-RIS PPO-PGA训练参数')
    parser.add_argument('--max_episodes', type=int, default=5000)
    # max_train_steps 会在 main() 中由 max_episodes * N_slot 自动计算，不再手动控制训练结束。
    parser.add_argument('--max_train_steps', type=int, default=None)
    parser.add_argument('--evaluate_freq', type=int, default=100)  # 每100个episode评估一次
    parser.add_argument('--save_freq', type=int, default=10)

    parser.add_argument('--policy_dist', type=str, default='Gaussian')
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--mini_batch_size', type=int, default=64)
    parser.add_argument('--hidden_width', type=int, default=512)
    parser.add_argument('--lr_a', type=float, default=1e-4)
    parser.add_argument('--lr_c', type=float, default=1e-4)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--K_epochs', type=int, default=10)

    parser.add_argument('--use_adv_norm', type=bool, default=True)
    parser.add_argument('--use_state_norm', type=bool, default=True)
    parser.add_argument('--use_reward_norm', type=bool, default=False)
    parser.add_argument('--use_reward_scaling', type=bool, default=True)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--use_lr_decay', type=bool, default=True)
    parser.add_argument('--use_grad_clip', type=bool, default=True)
    parser.add_argument('--use_orthogonal_init', type=bool, default=True)
    parser.add_argument('--set_adam_eps', type=float, default=True)
    parser.add_argument('--use_tanh', type=float, default=True)

    args = parser.parse_args()
    main(args, env_name='UAV-STAR-RIS-PPO-PGA', number=1, seed=42)
