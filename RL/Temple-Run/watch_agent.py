"""Tonton agent bermain.  python watch_agent.py [agent|random|human]

Kontrol : +/- ubah kecepatan, P pause, Esc keluar
Human   : panah kiri/kanan pindah lajur, panah atas/spasi lompat
Logika game tetap dari runner_env.py, file ini hanya untuk tampilan.
"""
import math
import os
import sys

import numpy as np
import pygame

from runner_env import (RunnerEnv, LANES, H, VIEW, WALL, LEFT, STAY, RIGHT, JUMP,
                        N_ACTIONS, ACTION_NAMES)

# ---------------- ukuran layar & kamera ----------------
GW, GH, PANEL = 480, 720, 250           # area game + panel info di kanan
HY = 215                                # garis horizon
Y_PLAYER = GH - 95                      # posisi kaki pemain di layar
D_PLAYER = 2.0                          # jarak kamera ke baris pemain
C = D_PLAYER * (Y_PLAYER - HY)          # konstanta proyeksi perspektif
LANE_PX = 130                           # lebar lajur di baris pemain
W0 = LANE_PX * LANES / 2 * D_PLAYER     # setengah lebar jalan x jarak

# ---------------- warna ----------------
SKY_TOP, SKY_HORIZON, FOG = (38, 52, 92), (247, 168, 98), (214, 160, 118)
JUNGLE, JUNGLE_DARK = (40, 92, 52), (22, 58, 36)
STONE_A, STONE_B, STONE_EDGE = (196, 170, 130), (181, 154, 115), (118, 96, 70)
WALL_C, WALL_DARK, MOSS = (168, 88, 66), (110, 54, 42), (92, 140, 70)
LOG_C, LOG_DARK, LOG_END = (128, 82, 44), (86, 52, 28), (204, 158, 104)
SKIN, SHIRT, PACK, HAT = (232, 188, 150), (52, 120, 150), (150, 105, 60), (205, 165, 95)
UI_BG, UI_CARD, UI_TEXT, UI_DIM = (24, 28, 38), (36, 41, 54), (236, 236, 240), (140, 146, 160)
GOOD, BAD, ACCENT = (96, 200, 120), (232, 86, 86), (250, 190, 80)
MODE_LABEL = {"agent": "AGENT Q-LEARNING", "random": "AGENT RANDOM", "human": "MANUSIA"}


# ---------------- helper ----------------
def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def depth(rc):
    """rc = posisi baris (boleh pecahan). Baris pemain (H-1) ada di jarak 2."""
    return H + 1 - rc


def project(rc):
    d = max(depth(rc), 0.3)
    return HY + C / d, 2 * W0 / d / LANES, d          # y layar, lebar lajur, jarak


def half_road(y):
    return W0 * (y - HY) / C


def fog_of(d):
    return min(max((d - 3) / (H - 1), 0), 1) * 0.8


def text(surf, s, font, color, pos, anchor="topleft"):
    img = font.render(str(s), True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


# ---------------- latar (digambar sekali) ----------------
def make_background():
    bg = pygame.Surface((GW, GH))
    for y in range(HY):
        pygame.draw.line(bg, mix(SKY_TOP, SKY_HORIZON, (y / HY) ** 1.6), (0, y), (GW, y))
    pygame.draw.circle(bg, (255, 226, 160), (GW // 2 + 130, HY - 48), 30)       # matahari
    for i, (col, amp, base) in enumerate([((176, 120, 112), 34, 62), ((126, 94, 98), 24, 30)]):
        pts = [(0, HY)]
        for x in range(0, GW + 8, 8):                                         # pegunungan
            h = base + amp * (0.6 * math.sin(x * 0.013 + i * 2) + 0.4 * math.sin(x * 0.031 + i))
            pts.append((x, HY - h))
        pygame.draw.polygon(bg, col, pts + [(GW, HY)])
    cx = GW // 2                                                              # kuil di ujung jalan
    for i in range(5):
        w = 120 - i * 20
        pygame.draw.rect(bg, (122, 92, 88), (cx - w // 2, HY - 12 * (i + 1), w, 12))
        pygame.draw.line(bg, (96, 72, 70), (cx - w // 2, HY - 12 * i - 1), (cx + w // 2, HY - 12 * i - 1))
    pygame.draw.rect(bg, (122, 92, 88), (cx - 14, HY - 78, 28, 18))
    pygame.draw.rect(bg, (60, 40, 40), (cx - 5, HY - 72, 10, 12))
    for y in range(HY, GH):                                                   # tanah hutan
        t = (y - HY) / (GH - HY)
        pygame.draw.line(bg, mix(mix(JUNGLE_DARK, FOG, 0.55), JUNGLE, t ** 0.6), (0, y), (GW, y))
    return bg


# ---------------- jalan & hutan ----------------
def draw_road(surf, p, T):
    y_top = HY + C / (H + 4)
    hw_top, hw_bot = half_road(y_top), half_road(GH)
    pygame.draw.polygon(surf, STONE_EDGE, [(GW / 2 - hw_top * 1.12, y_top), (GW / 2 + hw_top * 1.12, y_top),
                                           (GW / 2 + hw_bot * 1.12, GH), (GW / 2 - hw_bot * 1.12, GH)])
    for k in range(-4, H + 3):                       # ubin batu yang bergerak ke arah pemain
        y0, y1 = project(k - 0.5 + p)[0], project(k + 0.5 + p)[0]
        y0, y1 = max(y0, y_top), min(y1, GH)
        if y0 >= y1:
            continue
        f = fog_of(depth(k + p))
        band = (k - T) % 2                            # pakai id absolut supaya warna tidak berkedip
        col = mix(STONE_A if band == 0 else STONE_B, FOG, f)
        h0, h1 = half_road(y0), half_road(y1)
        pygame.draw.polygon(surf, col, [(GW / 2 - h0, y0), (GW / 2 + h0, y0), (GW / 2 + h1, y1), (GW / 2 - h1, y1)])
        seam = mix(STONE_EDGE, FOG, f)
        pygame.draw.line(surf, seam, (GW / 2 - h0, y0), (GW / 2 + h0, y0), max(1, int((y1 - y0) / 25)))
        for l in range(LANES):                        # sambungan ubin di tengah lajur (selang-seling)
            if band == 1:
                x0 = GW / 2 + (l - 1) * 2 * h0 / 3
                x1 = GW / 2 + (l - 1) * 2 * h1 / 3
                pygame.draw.line(surf, seam, (x0, y0), (x1, y1), 1)
    for side in (-1, 1):                              # garis pembatas lajur
        pygame.draw.line(surf, STONE_EDGE, (GW / 2 + side * hw_top / 3, y_top),
                         (GW / 2 + side * hw_bot / 3, GH), 2)


def draw_jungle(surf, p, T):
    items = []
    for k in range(-3, H + 3):
        a = k - T
        for side in (-1, 1):
            h = ((a * 73856093) ^ ((side + 2) * 19349663)) & 0xFFFF
            rc = k + p + (h % 7) / 10
            y, lw, d = project(rc)
            if d < 0.7:
                continue
            items.append((d, side, h, y, lw))
    for d, side, h, y, lw in sorted(items, reverse=True):     # jauh dulu, baru dekat
        f = fog_of(d)
        leaf = mix(((40, 110, 55), (52, 128, 62), (34, 96, 60))[h % 3], FOG, f)
        shade = mix(leaf, (0, 0, 0), 0.3)
        if h % 3 == 0:                                         # semak di pinggir jalan
            x = GW / 2 + side * half_road(y) * 1.18
            r = lw * 0.2
            pygame.draw.circle(surf, shade, (x, y - r * 0.4), r)
            pygame.draw.circle(surf, leaf, (x - side * r * 0.4, y - r * 0.8), r * 0.8)
            continue
        x = GW / 2 + side * (half_road(y) * 1.2 + lw * (0.55 + (h % 5) * 0.25))
        th = lw * (1.1 + (h % 4) * 0.2)
        pygame.draw.rect(surf, mix((86, 60, 40), FOG, f), (x - lw * 0.07, y - th, lw * 0.14, th))
        r = lw * (0.42 + (h % 3) * 0.08)
        pygame.draw.circle(surf, shade, (x, y - th + r * 0.35), r * 0.95)
        pygame.draw.circle(surf, leaf, (x - r * 0.55, y - th + r * 0.15), r * 0.7)
        pygame.draw.circle(surf, leaf, (x + r * 0.55, y - th + r * 0.15), r * 0.7)
        pygame.draw.circle(surf, leaf, (x, y - th - r * 0.15), r * 0.8)


# ---------------- rintangan ----------------
def draw_wall(surf, x, y, lw, f):
    w, h = lw * 0.86, lw * 0.62
    left, top = x - w / 2, y - h
    pygame.draw.ellipse(surf, mix((120, 98, 72), FOG, f), (left - w * 0.05, y - lw * 0.06, w * 1.1, lw * 0.14))
    pygame.draw.rect(surf, mix(WALL_C, FOG, f), (left, top, w, h))
    line, lw_px = mix(WALL_DARK, FOG, f), max(1, int(lw / 55))
    for i in range(3):                                         # pola bata
        yy = top + h * i / 3
        if i:
            pygame.draw.line(surf, line, (left, yy), (left + w, yy), lw_px)
        for fx in ((0.5,) if i % 2 == 0 else (0.25, 0.75)):
            pygame.draw.line(surf, line, (left + w * fx, yy), (left + w * fx, yy + h / 3), lw_px)
    pygame.draw.rect(surf, line, (left, top, w, h), lw_px)
    pygame.draw.rect(surf, mix(MOSS, FOG, f), (left - 1, top - h * 0.1, w + 2, h * 0.14),
                     border_radius=max(1, int(lw / 30)))


def draw_log(surf, x, y, lw, f):
    w, h = lw * 0.92, lw * 0.22
    pygame.draw.ellipse(surf, mix((120, 98, 72), FOG, f), (x - w * 0.55, y - lw * 0.05, w * 1.1, lw * 0.11))
    pygame.draw.rect(surf, mix(LOG_C, FOG, f), (x - w / 2, y - h, w, h), border_radius=max(1, int(h / 2)))
    dark = mix(LOG_DARK, FOG, f)
    for fy, a, b in ((0.35, -0.3, 0.25), (0.68, -0.2, 0.32)):
        pygame.draw.line(surf, dark, (x + w * a, y - h * fy), (x + w * b, y - h * fy), max(1, int(lw / 70)))
    for side in (-1, 1):                                       # ujung kayu (lingkaran tahun)
        ex = x + side * (w / 2 - h * 0.3)
        pygame.draw.ellipse(surf, mix(LOG_END, FOG, f), (ex - h * 0.3, y - h, h * 0.6, h))
        pygame.draw.ellipse(surf, dark, (ex - h * 0.13, y - h * 0.72, h * 0.26, h * 0.44), max(1, int(lw / 90)))


# ---------------- pemain (dilihat dari belakang) ----------------
def draw_player(surf, x, lift, t, airborne, dizzy):
    s = 1.0
    y = Y_PLAYER - lift
    sw = 48 * (1 - 0.45 * min(lift / 70, 1))
    pygame.draw.ellipse(surf, (120, 98, 72), (x - sw / 2, Y_PLAYER - 7, sw, 14))
    swing = 0 if (airborne or dizzy) else math.sin(t * 16)
    for side in (-1, 1):                                       # kaki
        hip = (x + side * 9, y - 32)
        foot = (x + side * 13, y - 16) if airborne else (x + side * 9, y - 3 - max(0, side * swing) * 12)
        pygame.draw.line(surf, (62, 54, 48), hip, foot, 10)
        pygame.draw.ellipse(surf, (50, 36, 30), (foot[0] - 7, foot[1] - 4, 14, 9))
    for side in (-1, 1):                                       # lengan
        sh = (x + side * 16, y - 64)
        hand = (sh[0] + side * 16, sh[1] - 16) if airborne else (sh[0] + side * 5, sh[1] + 22 + side * swing * 8)
        pygame.draw.line(surf, SKIN, sh, hand, 8)
        pygame.draw.circle(surf, SKIN, hand, 5)
    pygame.draw.rect(surf, SHIRT, (x - 17, y - 72, 34, 44), border_radius=10)        # badan
    pygame.draw.rect(surf, PACK, (x - 12, y - 66, 24, 28), border_radius=6)          # ransel
    pygame.draw.line(surf, mix(PACK, (0, 0, 0), 0.3), (x - 12, y - 56), (x + 12, y - 56), 2)
    pygame.draw.circle(surf, (70, 46, 34), (x, y - 84), 12)                          # kepala (rambut)
    pygame.draw.ellipse(surf, HAT, (x - 23, y - 93, 46, 12))                         # topi
    pygame.draw.rect(surf, HAT, (x - 12, y - 106, 24, 17), border_radius=6)
    pygame.draw.rect(surf, (110, 70, 40), (x - 12, y - 95, 24, 4))
    if dizzy:                                                  # bintang pusing saat nabrak
        for i in range(3):
            ang = t * 6 + i * 2 * math.pi / 3
            sx, sy = x + math.cos(ang) * 26, y - 112 + math.sin(ang) * 7
            pygame.draw.circle(surf, ACCENT, (sx, sy), 4)


# ---------------- panel info ----------------
def decode(code):
    if code == 0:
        return "kosong"
    kind = "tembok" if (code - 1) // VIEW == 0 else "kayu"
    return f"{kind}, jarak {(code - 1) % VIEW + 1}"


def draw_panel(surf, fonts, info):
    x0, pad = GW, GW + 20
    right = GW + PANEL - 20
    pygame.draw.rect(surf, UI_BG, (x0, 0, PANEL, GH))
    text(surf, "TEMPLE RUN RL", fonts["title"], ACCENT, (pad, 18))
    badge = text(surf, MODE_LABEL[info["mode"]], fonts["xs"], UI_BG, (pad + 10, 56))
    pygame.draw.rect(surf, UI_TEXT, badge.inflate(20, 8), border_radius=10)
    text(surf, MODE_LABEL[info["mode"]], fonts["xs"], UI_BG, (pad + 10, 56))

    text(surf, "SKOR (langkah bertahan)", fonts["xs"], UI_DIM, (pad, 94))
    text(surf, info["score"], fonts["huge"], UI_TEXT, (pad, 108))
    for i, (lab, val) in enumerate([("TERBAIK", info["best"]), ("EPISODE", info["episode"])]):
        text(surf, lab, fonts["xs"], UI_DIM, (pad + i * 110, 168))
        text(surf, val, fonts["m"], UI_TEXT, (pad + i * 110, 184))
    r = info["reward"]
    r_col = UI_DIM if r is None else (BAD if r < 0 else GOOD)
    for i, (lab, val, col) in enumerate([("REWARD", "-" if r is None else f"{r:+g}", r_col),
                                         ("TOTAL REWARD", f"{info['ret']:g}", UI_TEXT)]):
        text(surf, lab, fonts["xs"], UI_DIM, (pad + i * 110, 222))
        text(surf, val, fonts["m"], col, (pad + i * 110, 238))

    s = info["state"]
    text(surf, f"STATE  s = {tuple(int(v) for v in s)}", fonts["xs"], UI_DIM, (pad, 284))
    for l in range(LANES):
        me = l == s[0]
        pygame.draw.rect(surf, UI_CARD if not me else (58, 66, 86), (pad - 6, 302 + l * 22, PANEL - 28, 20),
                         border_radius=5)
        text(surf, f"lajur {l}{' (kamu)' if me else ''}: {decode(s[1 + l])}", fonts["s"],
             ACCENT if me else UI_TEXT, (pad, 304 + l * 22))

    src = {"agent": "dari policy", "random": "acak", "human": "dari keyboard"}[info["mode"]]
    text(surf, f"AKSI ({src})", fonts["xs"], UI_DIM, (pad, 380))
    for i, name in enumerate(ACTION_NAMES):
        bx, by = pad + (i % 2) * 106, 398 + (i // 2) * 34
        on = i == info["action"]
        pygame.draw.rect(surf, ACCENT if on else UI_CARD, (bx, by, 100, 28), border_radius=8)
        text(surf, name, fonts["s"], UI_BG if on else UI_DIM, (bx + 50, by + 14), "center")

    text(surf, "Q(s, a)  - value tiap aksi", fonts["xs"], UI_DIM, (pad, 478))
    q = info["q"]
    if q is None:
        text(surf, "tidak dipakai di mode ini", fonts["s"], UI_DIM, (pad, 500))
    else:
        cx, half = pad + 112, 48
        m = max(np.abs(q).max(), 1e-6)
        pygame.draw.line(surf, UI_DIM, (cx, 496), (cx, 496 + 4 * 22 - 4), 1)
        for i, name in enumerate(ACTION_NAMES):
            yy = 498 + i * 22
            on = i == int(q.argmax())
            text(surf, name, fonts["s"], ACCENT if on else UI_TEXT, (pad, yy))
            length = q[i] / m * half
            col = GOOD if q[i] >= 0 else BAD
            pygame.draw.rect(surf, col, (min(cx, cx + length), yy + 3, abs(length), 12), border_radius=3)
            text(surf, f"{q[i]:.1f}", fonts["s"], ACCENT if on else UI_DIM, (right, yy), "topright")

    y = GH - 112
    pygame.draw.line(surf, UI_CARD, (pad, y - 12), (right, y - 12), 1)
    draw_wall(surf, pad + 12, y + 18, 26, 0)
    text(surf, "tembok: harus pindah lajur", fonts["s"], UI_TEXT, (pad + 34, y + 2))
    draw_log(surf, pad + 12, y + 44, 26, 0)
    text(surf, "kayu: lompati / pindah", fonts["s"], UI_TEXT, (pad + 34, y + 28))
    keys = "panah kiri/kanan, atas = lompat" if info["mode"] == "human" else f"kecepatan: {info['fps']} langkah/detik"
    text(surf, keys, fonts["xs"], UI_DIM, (pad, y + 58))
    text(surf, "+/- kecepatan   P pause   Esc keluar", fonts["xs"], UI_DIM, (pad, y + 76))


def draw_banner(surf, fonts, title, color, sub):
    t_img, s_img = fonts["big"].render(title, True, color), fonts["s"].render(sub, True, UI_TEXT)
    w, h = max(t_img.get_width(), s_img.get_width()) + 60, 104
    box = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(box, (20, 22, 30, 220), box.get_rect(), border_radius=18)
    pygame.draw.rect(box, (*color, 255), box.get_rect(), 3, border_radius=18)
    box.blit(t_img, t_img.get_rect(center=(w / 2, 40)))
    box.blit(s_img, s_img.get_rect(center=(w / 2, 78)))
    surf.blit(box, box.get_rect(center=(GW / 2, GH * 0.36)))


# ---------------- main loop ----------------
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "agent"
    if mode not in MODE_LABEL:
        print(__doc__)
        return
    Q = None
    if mode == "agent":
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.npy")
        if not os.path.exists(path):
            print("q_table.npy belum ada. Jalankan dulu cell training di RL_TempleRun.ipynb.")
            return
        Q = np.load(path)

    pygame.init()
    screen = pygame.display.set_mode((GW + PANEL, GH))
    pygame.display.set_caption(f"Temple Run RL - mode: {mode}")
    font_name = "avenirnext,helveticaneue,arial"
    fonts = {k: pygame.font.SysFont(font_name, sz, bold=True) for k, sz in
             [("xs", 13), ("s", 15), ("m", 24), ("title", 24), ("big", 42), ("huge", 46)]}
    clock, bg, game = pygame.time.Clock(), make_background(), pygame.Surface((GW, GH))

    env, rng = RunnerEnv(seed=42), np.random.default_rng(42)
    s = env.reset()
    prev_lane, was_jump, leaving = env.lane, False, []
    T, acc, fps, anim = 0, 0.0, 6, 0.0
    best, episode, ep_ret = 0, 1, 0.0
    last_s, last_a, last_r, key_action = s, STAY, None, STAY
    banner, banner_timer, crash_time, particles, paused = None, 0, 0, [], False

    running = True
    while running:
        dt = clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps = min(20, fps + 1)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(1, fps - 1)
                elif e.key == pygame.K_p:
                    paused = not paused
                elif mode == "human":
                    key_action = {pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT, pygame.K_UP: JUMP,
                                  pygame.K_SPACE: JUMP}.get(e.key, key_action)
        if paused:
            dt = 0
        anim += dt / 1000
        tick = 1000 / fps

        if banner_timer > 0:                                   # jeda setelah nabrak / selesai
            banner_timer -= dt
            if banner_timer <= 0:
                s = env.reset()
                prev_lane, was_jump, leaving, acc = env.lane, False, [], 0.0
                episode, ep_ret, last_r, banner = episode + 1, 0.0, None, None
                last_s = s
        else:
            acc += dt
            if acc >= tick:                                    # satu langkah environment
                acc = min(acc - tick, tick)
                if mode == "agent":
                    a = int(Q[s].argmax())                     # policy = aksi dengan Q terbesar
                elif mode == "random":
                    a = int(rng.integers(N_ACTIONS))
                else:
                    a, key_action = key_action, STAY
                last_s, last_a = s, a
                leaving = [list(o) for o in env.obstacles if o[0] == H - 1]
                prev_lane, was_jump = env.lane, env.jumping
                s, r, done = env.step(a)
                T, last_r, ep_ret = T + 1, r, ep_ret + r
                best = max(best, env.t)
                if done:
                    acc, banner_timer = tick, 1600
                    if r < 0:
                        banner, crash_time = ("NABRAK!", BAD), anim
                        px = GW / 2 + (env.lane - 1) * LANE_PX
                        for _ in range(26):
                            ang, sp = rng.uniform(math.pi * 1.05, math.pi * 1.95), rng.uniform(120, 420)
                            particles.append([px, Y_PLAYER - 30, math.cos(ang) * sp, math.sin(ang) * sp,
                                              rng.uniform(500, 900), rng.uniform(3, 7),
                                              mix(WALL_C, STONE_A, rng.uniform())])
                    else:
                        banner = ("SELESAI!", GOOD)
        p = 1.0 if banner_timer > 0 else min(acc / tick, 1.0)
        dizzy = banner is not None and banner[0] == "NABRAK!"
        if dizzy:
            p = 0.7                                           # rintangan berhenti tepat di depan pemain

        # ---- gambar dunia ----
        game.blit(bg, (0, 0))
        draw_road(game, p, T)
        draw_jungle(game, p, T)
        e_lane = 1 - (1 - min(p / 0.45, 1)) ** 2             # gerak pindah lajur yang halus
        px = GW / 2 + (prev_lane + (env.lane - prev_lane) * e_lane - 1) * LANE_PX
        jump_h = LANE_PX * 0.55
        if dizzy:
            lift = 0
        elif env.jumping:
            lift = jump_h if was_jump else jump_h * math.sin(p * math.pi / 2)
        else:
            lift = jump_h * math.cos(p * math.pi / 2) if was_jump else 0

        objs = sorted([(r - 1 + p, l, k) for r, l, k in env.obstacles] + [(r + p, l, k) for r, l, k in leaving])
        player_done = False
        for rc, l, k in objs:
            if not player_done and rc > H - 1 + 1e-6:
                draw_player(game, px, lift, anim, lift > 1, dizzy)
                player_done = True
            y, lw, d = project(rc)
            if d < 0.5 or y - lw > GH:
                continue
            x = GW / 2 + (l - 1) * lw
            (draw_wall if k == WALL else draw_log)(game, x, y, lw, fog_of(d))
        if not player_done:
            draw_player(game, px, lift, anim, lift > 1, dizzy)

        for pt in particles:                                  # debu saat nabrak
            pt[0] += pt[2] * dt / 1000
            pt[1] += pt[3] * dt / 1000
            pt[3] += 900 * dt / 1000
            pt[4] -= dt
            pygame.draw.circle(game, pt[6], (pt[0], pt[1]), pt[5])
        particles = [pt for pt in particles if pt[4] > 0]

        since = anim - crash_time
        shake = (0, 0)
        if dizzy and since < 0.35:
            amp = 10 * (1 - since / 0.35)
            shake = (rng.uniform(-amp, amp), rng.uniform(-amp, amp))
            flash = pygame.Surface((GW, GH), pygame.SRCALPHA)
            flash.fill((230, 50, 50, int(110 * (1 - since / 0.35))))
            game.blit(flash, (0, 0))
        if banner:
            draw_banner(game, fonts, banner[0], banner[1], f"skor {env.t} langkah")
        elif paused:
            draw_banner(game, fonts, "PAUSE", ACCENT, "tekan P untuk lanjut")

        screen.fill((0, 0, 0))
        screen.blit(game, shake)
        draw_panel(screen, fonts, {
            "mode": mode, "score": env.t, "best": best, "episode": episode, "fps": fps,
            "state": last_s, "action": last_a, "reward": last_r, "ret": ep_ret,
            "q": Q[last_s] if Q is not None else None,
        })
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
