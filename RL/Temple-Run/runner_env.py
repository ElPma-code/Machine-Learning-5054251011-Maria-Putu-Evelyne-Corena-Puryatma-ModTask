import numpy as np

LANES, H, VIEW = 3, 8, 3          # 3 lajur, 8 baris, agent "melihat" 3 baris ke depan
WALL, LOG = 1, 2                  # tembok (harus pindah lajur), kayu (bisa dilompati)
LEFT, STAY, RIGHT, JUMP = 0, 1, 2, 3
ACTION_NAMES = ["kiri", "diam", "kanan", "lompat"]
N_ACTIONS = 4
N_CODES = 1 + 2 * VIEW            # per lajur: kosong, atau (jenis rintangan x jarak)
STATE_SHAPE = (LANES, N_CODES, N_CODES, N_CODES)


class RunnerEnv:
    """Endless runner 3 lajur ala Temple Run (tanpa grafis, supaya training cepat)."""

    def __init__(self, seed=None, max_steps=500, spawn_every=3):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.spawn_every = spawn_every

    def reset(self):
        self.lane, self.jumping, self.t = 1, False, 0
        self.obstacles = []               # list of [baris, lajur, jenis]
        return self.obs()

    def obs(self):
        """State = (lajur pemain, kode lajur 0, kode lajur 1, kode lajur 2)."""
        codes = []
        for l in range(LANES):
            near = [(H - 1 - r, k) for r, ln, k in self.obstacles
                    if ln == l and 1 <= H - 1 - r <= VIEW]
            if near:
                d, k = min(near)
                codes.append(1 + (k - 1) * VIEW + (d - 1))
            else:
                codes.append(0)
        return (self.lane, *codes)

    def _spawn(self):
        n = self.rng.integers(1, LANES)   # 1 atau 2 lajur terhalang, selalu ada celah
        for l in self.rng.choice(LANES, size=n, replace=False):
            self.obstacles.append([0, int(l), int(self.rng.choice([WALL, LOG]))])

    def step(self, action):
        if action == LEFT:
            self.lane = max(0, self.lane - 1)
        elif action == RIGHT:
            self.lane = min(LANES - 1, self.lane + 1)
        self.jumping = action == JUMP

        for o in self.obstacles:
            o[0] += 1                     # rintangan maju mendekati pemain
        crashed = any(r == H - 1 and ln == self.lane and (k == WALL or not self.jumping)
                      for r, ln, k in self.obstacles)
        self.obstacles = [o for o in self.obstacles if o[0] < H]

        self.t += 1
        if self.t % self.spawn_every == 0:
            self._spawn()

        if crashed:
            return self.obs(), -100, True
        reward = 1 - (0.5 if self.jumping else 0)   # +1 tiap langkah selamat, lompat ada biaya kecil
        return self.obs(), reward, self.t >= self.max_steps