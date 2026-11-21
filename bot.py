"""IronOval - a mechfight/1 battle royale bot (pure standard library, zero dependencies).

Strategy
--------
* Navigate the campus with A* over a grid built from the LiDAR terrain and building
  prisms (built once from hello.map.dir, within the 20 s hello budget).
* Zone survival is the top priority: always plan to stay inside the *next* circle,
  and run full speed (boosting) when outside the current one.
* Fight selectively: lead shots, strafe, face attackers (rear armor x1.5),
  disengage and kite when outmatched, use the weapon triangle for loot.
* Economy: farm drones/crates early, contest supply drops, collect cubes,
  take the highest-value upgrade when offered.
* Sound tracking: attribute bearings to known enemies, keep audio ghosts for
  blind gun shots (trees do not block bullets) and threat awareness.
"""
from __future__ import annotations

import array
import heapq
import json
import math
import os
import random
import sys

import mechfight as mf

PI = math.pi
TWO_PI = 2.0 * PI

# Base value of each upgrade id (used to pick between offer choices).
UPG = {
    "armor": 9.0, "plating": 8.5, "nanites": 7.5, "sensors": 6.0,
    "gyros": 5.5, "servos": 5.5, "jets": 4.5, "capacitor": 4.5, "reactor": 4.0,
    "gun_damage": 7.0, "gun_rate": 7.0, "gun_mag": 5.0, "gun_velocity": 5.0,
    "mortar_reload": 7.0, "mortar_splash": 5.0, "mortar_velocity": 4.5,
    "laser_damage": 7.0, "laser_charge": 6.0, "laser_pierce": 4.0,
    "lightning_efficiency": 7.0, "lightning_range": 5.0, "lightning_chain": 5.0,
}


# --------------------------------------------------------------------- helpers
def angdiff(a: float, b: float) -> float:
    """Signed shortest rotation taking b to a, in [-pi, pi]."""
    return (a - b + PI) % TWO_PI - PI


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def unit(dx, dy):
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return (0.0, 0.0)
    return (dx / d, dy / d)


# ---------------------------------------------------------------- map model
class MapModel:
    """Static campus map: terrain heightfield, prism solids, nav grid, A*, line of sight."""

    RES = 4.0  # nav cell size (m)

    def __init__(self, mdir: str, world: dict):
        tn = world["terrain"]
        self.t0x = float(tn["minX"])
        self.t0y = float(tn["minY"])
        self.step = float(tn.get("step", 1.0))
        self.nx = int(tn["nx"])
        self.ny = int(tn["ny"])
        self.h = array.array("h")
        tpath = os.path.join(mdir, world.get("_dir", ""), tn.get("file", "terrain.i16"))
        with open(tpath, "rb") as f:
            self.h.fromfile(f, self.nx * self.ny)
        self.bounds = world.get("bounds") or {}
        self.prisms = []
        self.hash = {}
        for p in world.get("prisms", []):
            ring = p.get("ring") or []
            if len(ring) < 6:
                continue
            xs = ring[0::2]
            ys = ring[1::2]
            pr = {
                "bb": (min(xs), min(ys), max(xs), max(ys)),
                "rx": ring[0::2],
                "ry": ring[1::2],
                "holes": [(h[0::2], h[1::2]) for h in (p.get("holes") or [])],
                "base": float(p.get("base", -1e9)),
                "top": float(p.get("top", 1e9)),
            }
            self.prisms.append(pr)
            x0, y0, x1, y1 = pr["bb"]
            for cx in range(int(x0 // 32), int(x1 // 32) + 1):
                for cy in range(int(y0 // 32), int(y1 // 32) + 1):
                    self.hash.setdefault((cx, cy), []).append(len(self.prisms) - 1)
        self._build_nav()

    # ------------------------------------------------------------ terrain
    def hgt(self, x: float, y: float) -> float:
        """Bilinear terrain height (m) at world (x, y)."""
        h = self.h
        n = self.nx
        fx = (x - self.t0x) / self.step
        fy = (y - self.t0y) / self.step
        if fx < 0.0:
            fx = 0.0
        elif fx > n - 1.001:
            fx = n - 1.001
        if fy < 0.0:
            fy = 0.0
        elif fy > self.ny - 1.001:
            fy = self.ny - 1.001
        i = int(fx)
        j = int(fy)
        tx = fx - i
        ty = fy - j
        r0 = j * n
        a = h[r0 + i]
        b = h[r0 + i + 1]
        c = h[r0 + n + i]
        d = h[r0 + n + i + 1]
        top = a + (b - a) * tx
        bot = c + (d - c) * tx
        return (top + (bot - top) * ty) * 0.01

    # ------------------------------------------------------------ solids
    @staticmethod
    def _pip(xs, ys, x, y):
        n = len(xs)
        inside = False
        j = n - 1
        for i in range(n):
            yi = ys[i]
            yj = ys[j]
            if (yi > y) != (yj > y):
                if x < (xs[j] - xs[i]) * (y - yi) / (yj - yi) + xs[i]:
                    inside = not inside
            j = i
        return inside

    def _in_prism(self, pi: int, x: float, y: float) -> bool:
        p = self.prisms[pi]
        if not self._pip(p["rx"], p["ry"], x, y):
            return False
        for hx, hy in p["holes"]:
            if self._pip(hx, hy, x, y):
                return False
        return True

    def _prisms_near(self, x: float, y: float):
        return self.hash.get((int(x // 32), int(y // 32)), ())

    def solid_at(self, x: float, y: float, z: float) -> bool:
        """True if (x, y, z) is inside a solid prism (blocks bullets and beams)."""
        for pi in self._prisms_near(x, y):
            p = self.prisms[pi]
            bb = p["bb"]
            if not (bb[0] <= x <= bb[1] and bb[2] <= y <= bb[3]):
                continue
            if not (p["base"] <= z <= p["top"]):
                continue
            if self._in_prism(pi, x, y):
                return True
        return False

    def wall_at(self, x: float, y: float, ground: float) -> bool:
        """True if a prism makes the ground cell at (x, y) unwalkable from the ground."""
        for pi in self._prisms_near(x, y):
            p = self.prisms[pi]
            bb = p["bb"]
            if not (bb[0] <= x <= bb[1] and bb[2] <= y <= bb[3]):
                continue
            if p["top"] <= ground + 0.3:
                continue  # sunk below ground
            if p["base"] > ground + 9.5:
                continue  # clears the 9 m head height: walk under
            if self._in_prism(pi, x, y):
                return True
        return False

    def los(self, ax, ay, az, bx, by, bz, crates=()) -> bool:
        """Bullet/beam line of sight: prisms and terrain block; crates too. Trees do not."""
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return True
        steps = max(1, int(d / 2.0))
        for k in range(1, steps):
            f = k / steps
            x = ax + dx * f
            y = ay + dy * f
            z = az + dz * f
            if self.solid_at(x, y, z):
                return False
            if self.hgt(x, y) > z + 0.25:
                return False
            for cx, cy in crates:
                if math.hypot(x - cx, y - cy) < 1.8 and z < self.hgt(cx, cy) + 2.4:
                    return False
        return True

    # ------------------------------------------------------------ nav grid
    def _build_nav(self):
        self.g0x = self.t0x
        self.g0y = self.t0y
        self.nx2 = int((self.nx - 1) // self.RES) + 1
        self.ny2 = int((self.ny - 1) // self.RES) + 1
        n = self.nx2 * self.ny2
        self.cost = bytearray(n)  # 0 = blocked, else cost*2 (>=2)
        self.ch = array.array("h", [0] * n)  # coarse height, cm
        b = self.bounds
        bx0 = b.get("minX", self.t0x) + 1.5
        bx1 = b.get("maxX", self.t0x + self.nx) - 1.5
        by0 = b.get("minY", self.t0y) + 1.5
        by1 = b.get("maxY", self.t0y + self.ny) - 1.5
        for j in range(self.ny2):
            y = self.g0y + (j + 0.5) * self.RES
            for i in range(self.nx2):
                x = self.g0x + (i + 0.5) * self.RES
                idx = j * self.nx2 + i
                h = self.hgt(x, y)
                self.ch[idx] = int(h * 100)
                if not (bx0 <= x <= bx1 and by0 <= y <= by1):
                    continue
                if self.wall_at(x, y, h):
                    continue
                worst = 0.0
                if i > 0:
                    worst = max(worst, abs(self.ch[idx - 1] - self.ch[idx]))
                if i < self.nx2 - 1:
                    worst = max(worst, abs(self.ch[idx + 1] - self.ch[idx]))
                if j > 0:
                    worst = max(worst, abs(self.ch[idx - self.nx2] - self.ch[idx]))
                if j < self.ny2 - 1:
                    worst = max(worst, abs(self.ch[idx + self.nx2] - self.ch[idx]))
                worst /= 100.0
                if worst > 4.6:  # steeper than ~50 deg: jet-only
                    self.cost[idx] = 12
                elif worst > 1.4:
                    self.cost[idx] = 4 + min(6, int(worst * 2))
                else:
                    self.cost[idx] = 2

    def cell_of(self, x: float, y: float) -> int:
        i = int((x - self.g0x) / self.RES)
        j = int((y - self.g0y) / self.RES)
        if i < 0:
            i = 0
        elif i >= self.nx2:
            i = self.nx2 - 1
        if j < 0:
            j = 0
        elif j >= self.ny2:
            j = self.ny2 - 1
        return j * self.nx2 + i

    def cell_xy(self, idx: int):
        i = idx % self.nx2
        j = idx // self.nx2
        return (self.g0x + (i + 0.5) * self.RES, self.g0y + (j + 0.5) * self.RES)

    def _near_passable(self, idx, radius):
        nx2, ny2 = self.nx2, self.ny2
        ci = idx % nx2
        cj = idx // nx2
        for r in range(1, radius + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nx2 and 0 <= nj < ny2:
                        k = nj * nx2 + ni
                        if self.cost[k]:
                            return k
        return None

    def clear(self, sx, sy, gx, gy) -> bool:
        """Can the mech walk a straight line from (sx, sy) to (gx, gy)?"""
        d = math.hypot(gx - sx, gy - sy)
        if d < 2.0:
            return True
        steps = max(1, int(d / 3.0))
        for k in range(1, steps + 1):
            f = k / steps
            x = sx + (gx - sx) * f
            y = sy + (gy - sy) * f
            h = self.hgt(x, y)
            if self.wall_at(x, y, h):
                return False
            if k < steps:
                h0 = self.hgt(sx + (gx - sx) * ((k - 1) / steps), sy + (gy - sy) * ((k - 1) / steps))
                if abs(h - h0) > 4.5:
                    return False
        return True

    def astar(self, sx, sy, gx, gy, budget=20000):
        """Grid A*. Returns a list of world (x, y) points from start to goal, or None."""
        if self.clear(sx, sy, gx, gy):
            return [(sx, sy), (gx, gy)]
        cost = self.cost
        s = self.cell_of(sx, sy)
        g = self.cell_of(gx, gy)
        if not cost[s]:
            s = self._near_passable(s, 6)
            if s is None:
                return None
        if not cost[g]:
            g = self._near_passable(g, 12)
            if g is None:
                return None
        if s == g:
            return [(sx, sy)]
        n = self.nx2 * self.ny2
        gscore = [-1e18] * n
        came = [-1] * n
        closed = bytearray(n)
        gscore[s] = 0.0
        ci0 = g % self.nx2
        cj0 = g // self.nx2
        openq = [((abs(s // self.nx2 - cj0) + abs(s % self.nx2 - ci0)) * 2.0, 0.0, s)]
        push = heapq.heappush
        pop = heapq.heappop
        pops = 0
        while openq:
            _, fc, cur = pop(openq)
            if closed[cur]:
                continue
            closed[cur] = 1
            pops += 1
            if cur == g:
                break
            if pops >= budget:
                return None
            ci = cur % self.nx2
            cj = cur // self.nx2
            base = gscore[cur]
            hcur = self.ch[cur]
            for di, dj, w in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                              (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)):
                ni = ci + di
                nj = cj + dj
                if ni < 0 or nj < 0 or ni >= self.nx2 or nj >= self.ny2:
                    continue
                nb = nj * self.nx2 + ni
                c = cost[nb]
                if not c:
                    continue
                if w > 1.0:  # no corner cutting
                    if not cost[nj * self.nx2 + ci] or not cost[cj * self.nx2 + ni]:
                        continue
                climb = (abs(self.ch[nb] - hcur) - 100) * 0.02
                if climb > 0.0:
                    climb += 2.0
                nc = base + w * (c + climb)
                if nc < gscore[nb]:
                    gscore[nb] = nc
                    came[nb] = cur
                    hh = (abs(ni - ci0) + abs(nj - cj0)) * 2.0
                    push(openq, (nc + hh, nc, nb))
        if not closed[g]:
            return None
        path = [g]
        cur = g
        while cur != s:
            cur = came[cur]
            path.append(cur)
        path.reverse()
        out = [self.cell_xy(i) for i in path]
        out[0] = (sx, sy)
        out[-1] = (gx, gy)
        return out


# ---------------------------------------------------------------- bot
class Bot:
    name = "IronOval"

    def __init__(self, hello):
        cfg = hello["config"]
        self.cfg = cfg
        mdir = hello["map"]["dir"]
        files = hello["map"].get("files") or []
        wpath = "v2/world.json"
        for f in files:
            if f.endswith("world.json"):
                wpath = f
                break
        with open(os.path.join(mdir, wpath)) as fh:
            world = json.load(fh)
        world["_dir"] = os.path.dirname(wpath)  # terrain file lives next to world.json
        self.map = MapModel(mdir, world)
        self.rng = random.Random(hello.get("seed", 0))
        self.t0 = 0.0
        self.enemies = {}      # id -> dict
        self.ghosts = []       # [{x, y, t, kind}]
        self.known_dead = set()
        self.path = None
        self.path_goal = None
        self.path_plan_t = -1e9
        self.path_idx = 0
        self.stuck_t = 0.0
        self.stuck_pos = (0.0, 0.0)
        self.dmg_from = None
        self.dmg_t = -1e9
        self.dmg_bearing = None
        self.dmg_bearing_t = -1e9
        self.target = None
        self.strafe_dir = 1
        self.strafe_flip_t = 0.0
        self.los_key = None
        self.los_t = -1e9
        self.los_ok = True
        self.my_kills = 0
        self.blind_cd = -1e9
        self.last_sound = None  # (bearing, t, kind)
        self.energy_floor = 25.0
        self.w0 = None
        self.w1 = None
        self.gun_ammo = 12
        self.survival_hp = 520.0
        self.danger_radius = 60.0
        self.cover_search_radius = 30.0

    # ------------------------------------------------------------ state
    def _new_enemy(self, m):
        return {
            "pos": m["pos"], "vel": m.get("vel", (0, 0, 0)), "aim": m.get("aim", (0, 0)),
            "hp": m.get("hp", 1400.0), "max_hp": m.get("max_hp", 1400.0),
            "weapons": tuple(m.get("weapons") or ("gun",)), "charging": bool(m.get("charging")),
            "last_seen": self.t0, "last_heard": -1e9, "shot_time": -1e9,
            "kind": m.get("kind", "mech"), "cubes": m.get("cubes", 0), "dead": False,
        }

    def _update_state(self, obs, me, t):
        self.t0 = t
        eid = me["id"]
        w = me["weapons"]
        self.w0 = w[0] if len(w) > 0 else None
        self.w1 = w[1] if len(w) > 1 else None
        if self.w0 and self.w0.get("kind") == "gun":
            self.gun_ammo = self.w0.get("ammo", 12)
        mx, my, _ = me["pos"]
        seen = set()
        for m in obs.get("mechs", []):
            mid = m["id"]
            if mid == eid:
                continue
            e = self.enemies.get(mid)
            if e is None:
                e = self._new_enemy(m)
                self.enemies[mid] = e
            e["pos"] = m["pos"]
            e["vel"] = m.get("vel", (0, 0, 0))
            e["aim"] = m.get("aim", (0, 0))
            e["last_seen"] = t
            if m.get("kind") == "npc":
                e["kind"] = "npc"
                e["hp"] = m.get("hp", 350.0)
            else:
                e["kind"] = "mech"
                e["hp"] = m.get("hp", e["hp"])
                e["max_hp"] = m.get("max_hp", e["max_hp"])
                e["cubes"] = m.get("cubes", 0)
                e["weapons"] = tuple(m.get("weapons") or ("gun",))
                e["charging"] = bool(m.get("charging"))
            seen.add(mid)
        for mid in list(self.enemies):
            e = self.enemies[mid]
            if t - e["last_seen"] > 9.0 and t - e["last_heard"] > 14.0:
                del self.enemies[mid]
        for ev in obs.get("events", []):
            k = ev.get("type")
            if k == "damaged":
                by = ev.get("by")
                br = ev.get("bearing")
                if by is not None:
                    self.dmg_from = by
                    self.dmg_t = t
                    e = self.enemies.get(by)
                    if e:
                        e["shot_time"] = t
                if br is not None and ev.get("weapon") != "zone":
                    self.dmg_bearing = br
                    self.dmg_bearing_t = t
            elif k == "kill":
                tg = ev.get("target")
                self.known_dead.add(tg)
                e = self.enemies.get(tg)
                if e:
                    e["dead"] = True
                self.my_kills += 1
        # sounds
        for s in obs.get("sounds", []):
            kind = s.get("kind")
            bearing = s.get("bearing")
            if bearing is None:
                continue
            if kind in ("gun", "mortar", "laser", "lightning", "explosion"):
                self.last_sound = (bearing, t, kind)
            band = s.get("dist", "far")
            lo, hi = {"near": (12.0, 55.0), "mid": (55.0, 150.0), "far": (150.0, 380.0)}[band]
            best = None
            best_err = 1e9
            for mid, e in self.enemies.items():
                if e["kind"] == "npc":
                    continue
                ex, ey, _ = e["pos"]
                b = math.atan2(ey - my, ex - mx)
                if abs(angdiff(b, bearing)) < 0.30:
                    d = math.hypot(ex - mx, ey - my)
                    if lo * 0.6 < d < hi * 1.4:
                        err = abs(d - (lo + hi) * 0.5)
                        if err < best_err:
                            best_err = err
                            best = mid
            if best is not None:
                e = self.enemies[best]
                e["last_heard"] = t
                if kind in ("gun", "mortar", "laser", "lightning", "explosion"):
                    e["shot_time"] = t
            elif kind not in ("step", "boost", "reload", "jet"):
                d = lo + self.rng.random() * (hi - lo)
                self.ghosts.append({"x": mx + d * math.cos(bearing),
                                    "y": my + d * math.sin(bearing), "t": t, "kind": kind})
        self.ghosts = [g for g in self.ghosts if t - g["t"] < 12.0]
        if len(self.ghosts) > 24:
            self.ghosts = self.ghosts[-24:]
        # beams: someone fired an instant weapon
        for bm in obs.get("beams", []):
            e = self.enemies.get(bm.get("by"))
            if e is not None:
                e["last_heard"] = t
                e["shot_time"] = t

    # ------------------------------------------------------------ offers
    def _pick_offer(self, offer, me):
        best_i = 0
        best_sc = -1e9
        for i, ch in enumerate(offer.get("choices", [{}])):
            sc = UPG.get(ch.get("id", ""), 3.0)
            if ch.get("id") in ("armor", "plating") and me["hp"] < me["max_hp"] * 0.6:
                sc += 2.0
            if ch.get("id") in ("armor", "plating") and me["hp"] < me["max_hp"] * 0.85:
                sc += 1.0
            if ch.get("id") == "sensors":
                sc += 1.0
            if ch.get("id") in ("gun_damage", "laser_damage", "mortar_damage", "gun_rate",
                                "lightning_efficiency", "laser_charge"):
                sc += 0.5
            sc += self.rng.random() * 0.6
            if sc > best_sc:
                best_sc = sc
                best_i = i
        return best_i

    # ------------------------------------------------------------ loot
    def _weapon_value(self, kind):
        nl = nm = nlt = 0
        for e in self.enemies.values():
            if e["kind"] == "npc" or e["dead"]:
                continue
            w = e["weapons"]
            if "laser" in w:
                nl += 1
            if "mortar" in w:
                nm += 1
            if "lightning" in w:
                nlt += 1
        if kind == "mortar":
            return 5.0 + (5.0 if nlt else 0.0) - (2.0 if nl else 0.0)
        if kind == "laser":
            return 5.0 + (5.0 if nm else 0.0) - (2.0 if nlt else 0.0)
        return 5.0 + (5.0 if nl else 0.0)  # lightning

    def _loot(self, act, obs, me, mx, my):
        if not obs.get("pickups"):
            return
        reach = self.cfg["loot"]["pickup_radius"]
        bykind = {}
        for p in obs["pickups"]:
            d = math.hypot(p["pos"][0] - mx, p["pos"][1] - my)
            if d < reach:
                bykind.setdefault(p["kind"], []).append((d, p))
        bykind.pop("gun", None)  # a second gun is useless
        my_w = [w["kind"] for w in me["weapons"] if w]
        laser_busy = self.w1 is not None and self.w1.get("kind") == "laser" and self.w1.get("charge", 0) > 0
        if len(my_w) < 2:
            best = None
            for kind, lst in bykind.items():
                if kind in my_w:
                    continue
                lst.sort()
                sc = self._weapon_value(kind) - lst[0][0] * 0.02
                if best is None or sc > best[0]:
                    best = (sc, kind, lst[0])
            if best:
                act["take"] = best[2][1]["id"]
        elif not laser_busy:
            cur_v = 0.0
            for k in my_w:
                if k != "gun":
                    cur_v = max(cur_v, self._weapon_value(k))
            for kind, lst in bykind.items():
                if kind in my_w:
                    continue
                if self._weapon_value(kind) > cur_v + 3.0:
                    drop_slot = 1 if (len(my_w) > 1 and my_w[1] != "gun") else 0
                    if my_w[drop_slot] == "gun":
                        drop_slot = 1
                    act["take"] = lst[0][1]["id"]
                    act["drop"] = drop_slot
                    break

    # ------------------------------------------------------------ zone
    def _zone_goal(self, obs, me):
        z = obs["zone"]
        t = obs["t"]
        cx, cy = z["center"]
        r = z["radius"]
        mx, my, _ = me["pos"]
        dx, dy = mx - cx, my - cy
        d = math.hypot(dx, dy)
        if not me["in_zone"]:
            if d > 1e-6:
                return (cx + dx / d * r * 0.8, cy + dy / d * r * 0.8), 100
            return (cx, cy), 100
        ncx, ncy = z["next_center"]
        nr = z["next_radius"]
        if nr < 2.0:
            return (ncx, ncy), 60
        nd = math.hypot(mx - ncx, my - ncy)
        safe = nr * 0.62
        if nd > safe:
            gx = ncx + (mx - ncx) / nd * safe
            gy = ncy + (my - ncy) / nd * safe
            t_left = z["shrink_end"] - t
            need = (nd - safe) / 13.0
            if t > z["shrink_start"] or t_left < need + 8:
                return (gx, gy), 40
            if t_left < need + 45:
                return (gx, gy), 12
            return (gx, gy), 2
        return (ncx, ncy), 1

    # ------------------------------------------------------------ dodge
    def _dodge(self, obs, me, mx, my, mz):
        cx, cy, cz = mx, my, mz + 4.5
        best = None
        best_sc = 0.0
        for p in obs.get("projectiles", []):
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            rvx, rvy, rvz = cx - px, cy - py, cz - pz
            vv = vx * vx + vy * vy + vz * vz
            if vv < 1e-9:
                continue
            tt = -(rvx * vx + rvy * vy + rvz * vz) / vv
            if tt <= 0.0 or tt > 0.75:
                continue
            dx = rvx + vx * tt
            dy = rvy + vy * tt
            dz = rvz + vz * tt
            d = math.hypot(dx, dy, dz)
            if d < 3.0:
                vh = math.hypot(vx, vy)
                if vh < 1e-6:
                    continue
                nx_, ny_ = -vy / vh, vx / vh
                sdist = nx_ * (cx - px) + ny_ * (cy - py)
                sgn = 1.0 if sdist >= 0.0 else -1.0
                sc = (1.0 - tt / 0.75) * (3.0 - d)
                if sc > best_sc:
                    best_sc = sc
                    best = (nx_ * sgn, ny_ * sgn, 1.0 - tt / 0.75)
        for e in self.enemies.values():
            if not e.get("charging"):
                continue
            ex, ey, _ = e["pos"]
            d = math.hypot(ex - mx, ey - my)
            if d < 4.0 or d > 220.0:
                continue
            ay = e["aim"][0]
            b = math.atan2(my - ey, mx - ex)
            if abs(angdiff(b, ay)) < 0.12:
                px_, py_ = -math.sin(ay), math.cos(ay)
                sgn = 1.0 if (px_ * (mx - ex) + py_ * (my - ey)) >= 0.0 else -1.0
                sc = 1.5 * (1.0 - d / 220.0)
                if sc > best_sc:
                    best = (px_ * sgn, py_ * sgn, 0.7)
                    best_sc = sc
        return best

    # ------------------------------------------------------------ targeting
    def _select_target(self, obs, me, t, zone_urg):
        mx, my, _ = me["pos"]
        best = None
        best_sc = 280.0
        for mid, e in self.enemies.items():
            if e["kind"] == "npc" or e["dead"] or mid in self.known_dead:
                continue
            if t - e["last_seen"] > 2.8:
                continue
            ex, ey, _ = e["pos"]
            d = math.hypot(ex - mx, ey - my)
            if d > 235.0:
                continue
            sc = 1400.0 / (1.0 + d / 25.0)
            if e["hp"] < 600.0:
                sc += 550.0
            if e["hp"] < 300.0:
                sc += 700.0
            if t - e["shot_time"] < 2.5:
                sc += 320.0
            sc += 10.0 * e.get("cubes", 0)
            if e["hp"] > me["hp"] * 1.7 and d < 100.0:
                sc -= 280.0
            if zone_urg >= 40:
                sc -= 220.0
            if sc > best_sc:
                best_sc = sc
                best = mid
        if best is not None:
            return best
        if t < 240.0 and me["hp"] > 380.0:
            bd = 1e9
            for mid, e in self.enemies.items():
                if e["kind"] != "npc" or e["dead"]:
                    continue
                if t - e["last_seen"] > 2.8:
                    continue
                ex, ey, _ = e["pos"]
                d = math.hypot(ex - mx, ey - my)
                if d < 75.0 and d < bd:
                    bd = d
                    best = mid
            return best
        return None

    def _engage(self, me, e, t):
        if e["kind"] == "npc":
            return True
        mx, my, _ = me["pos"]
        ex, ey, _ = e["pos"]
        d = math.hypot(ex - mx, ey - my)
        hp = me["hp"]
        if hp < 700.0:
            return False
        if hp < self.survival_hp or d < 18.0:
            return False
        if d < 75.0 and self.gun_ammo <= 1:
            w1 = self.w1
            if not (w1 and w1.get("kind") in ("laser", "mortar", "lightning")):
                return False
        return True

    # ------------------------------------------------------------ combat math
    def _predict(self, muzzle, tpos, tvel, v, iters=3):
        mx, my, _ = muzzle
        px, py, pz = tpos
        t = math.hypot(px - mx, py - my) / v
        qx, qy, qz = px, py, pz
        for _ in range(iters):
            qx = px + tvel[0] * t
            qy = py + tvel[1] * t
            qz = pz + tvel[2] * t
            t = math.hypot(qx - mx, qy - my) / v
        return (qx, qy, qz)

    @staticmethod
    def _aim_at(muzzle, point):
        mx, my, mz = muzzle
        dx = point[0] - mx
        dy = point[1] - my
        dz = point[2] - mz
        return (math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy)))

    def _mortar_solve(self, muzzle, tpos, tvel, v, grav):
        """(pitch, flight_time) for a shell from muzzle to predicted target, or (None, 0)."""
        mx, my, mz = muzzle
        px, py, pz = tpos
        t = 2.5
        pitch = None
        for _ in range(3):
            qx = px + tvel[0] * t
            qy = py + tvel[1] * t
            d = math.hypot(qx - mx, qy - my)
            if d < 3.0:
                return (None, 0.0)
            h = pz - mz
            A = grav * d * d / (2.0 * v * v)
            disc = d * d - 4.0 * A * (A + h)
            if disc < 0.0:
                return (None, 0.0)
            u = (d - math.sqrt(disc)) / (2.0 * A)
            if u < 0.03:
                u = 0.03
            pitch = math.atan(u)
            s = math.sin(pitch)
            t = (v * s + math.sqrt(max(0.0, v * v * s * s + 2.0 * grav * (mz - pz)))) / grav
        return (pitch, t)

    def _los_check(self, me, obs, muzzle, point):
        key = (id(self.target), int(self.t0 * 4))
        if key == self.los_key:
            return self.los_ok
        crates = [(c["pos"][0], c["pos"][1]) for c in obs.get("crates", [])]
        ok = self.map.los(muzzle[0], muzzle[1], muzzle[2],
                          point[0], point[1], point[2], crates)
        self.los_key = key
        self.los_t = self.t0
        self.los_ok = ok
        return ok

    def _combat(self, act, obs, me, t, tgt_id, engaged, zone_goal):
        mx, my, mz = me["pos"]
        stats = me["stats"]
        mz_h = mz + stats.get("muzzle_height", 7.0)
        aim = None
        fire = [False, False]
        w0, w1 = self.w0, self.w1
        if tgt_id is not None and engaged:
            e = self.enemies[tgt_id]
            ex, ey, ez = e["pos"]
            vx, vy, vz = e["vel"]
            d = math.hypot(ex - mx, ey - my)
            tgt_z = ez + 4.5 * (1.0 + 0.05 * e.get("cubes", 0))
            used = False
            # lightning: close-range burst
            if (w1 and w1["kind"] == "lightning" and d < 42.0
                    and me["energy"] >= w1["stats"].get("energy", 3.0) + 2.0
                    and w1["cooldown"] <= 0.02):
                fire[1] = True
                used = True
                aim = self._aim_at((mx, my, mz_h), (ex, ey, tgt_z))
            # gun: primary
            if not used and w0 and w0.get("kind") == "gun" and w0.get("ammo", 0) > 0 and w0["cooldown"] <= 0.02:
                speed = w0["stats"].get("speed", 252.0)
                pred = self._predict((mx, my, mz_h), (ex, ey, tgt_z), (vx, vy, vz), speed)
                yaw, pit = self._aim_at((mx, my, mz_h), pred)
                ay, ap = me["aim"]
                if self._los_check(me, obs, (mx, my, mz_h), pred) \
                        and abs(angdiff(yaw, ay)) < 0.14 and abs(pit - ap) < 0.11:
                    fire[0] = True
                    used = True
                aim = (yaw, pit)
            # mortar: lob (no line of sight needed, it flies over cover)
            if not used and w1 and w1["kind"] == "mortar" and 15.0 < d < 205.0 and w1["cooldown"] <= 0.02:
                st = w1["stats"]
                muzzle_m = (mx, my, mz + 9.5)
                pitch, tfl = self._mortar_solve(muzzle_m, (ex, ey, tgt_z), (vx, vy, vz),
                                                st.get("speed", 72.0), st.get("gravity", 22.0))
                if pitch is not None:
                    pred = (ex + vx * tfl, ey + vy * tfl, tgt_z)
                    yaw = math.atan2(pred[1] - my, pred[0] - mx)
                    ay, ap = me["aim"]
                    if abs(angdiff(yaw, ay)) < 0.16 and abs(pitch - ap) < 0.13:
                        fire[1] = True
                        used = True
                    aim = (yaw, pitch)
            # laser: charge and release (only when it looks safe from lightning)
            if not used and w1 and w1["kind"] == "laser" and d < 230.0:
                st = w1["stats"]
                charge = w1.get("charge", 0.0)
                if charge > 0.0:
                    pred = self._predict((mx, my, mz_h), (ex, ey, tgt_z), (vx, vy, vz), 252.0)
                    aim = self._aim_at((mx, my, mz_h), pred)
                    ay, ap = me["aim"]
                    full = st.get("charge", 0.75)
                    min_charge = st.get("min_charge", 0.15)
                    aim_error = abs(angdiff(aim[0], ay)) + abs(aim[1] - ap)
                    if charge >= full - 0.01 or (charge >= min_charge and aim_error < 0.18):
                        fire[1] = False  # release -> fires the beam
                        used = True
                    else:
                        fire[1] = True   # keep holding
                        used = True
                elif w1["cooldown"] <= 0.02 and d > 48.0:
                    safe = True
                    for m2 in obs.get("mechs", []):
                        if m2.get("kind") == "mech" and "lightning" in (m2.get("weapons") or ()):
                            if math.hypot(m2["pos"][0] - mx, m2["pos"][1] - my) < 60.0:
                                safe = False
                                break
                    if safe and self._los_check(me, obs, (mx, my, mz_h), (ex, ey, tgt_z)):
                        fire[1] = True  # start charging
                        aim = self._aim_at((mx, my, mz_h), (ex, ey, tgt_z))
            # gun reload
            if w0 and w0.get("kind") == "gun" and w0.get("ammo", 0) <= 0 and w0.get("reload", 0) == 0:
                act["reload"] = True
            if aim is None:
                aim = self._aim_at((mx, my, mz_h), (ex, ey, tgt_z))
        else:
            # no target: look toward the movement goal (or hold aim)
            if zone_goal:
                yaw = math.atan2(zone_goal[1] - my, zone_goal[0] - mx)
                aim = (yaw, 0.0)
            else:
                aim = (me["aim"][0], me["aim"][1])
            # face a recent attacker (rear armor x1.5) when not already aiming at someone
            if self.dmg_from is not None and t - self.dmg_t < 1.2:
                e = self.enemies.get(self.dmg_from)
                if e is not None and t - e["last_seen"] < 3.0:
                    aim = self._aim_at((mx, my, mz_h), e["pos"])
                elif self.dmg_bearing is not None and t - self.dmg_bearing_t < 1.2:
                    aim = (self.dmg_bearing, me["aim"][1])
            # otherwise face the freshest weapon sound (they are firing somewhere)
            elif self.last_sound is not None and t - self.last_sound[1] < 2.0:
                aim = (self.last_sound[0], me["aim"][1])
        # blind shot at a fresh audio contact (trees do not block bullets)
        if fire[0] is False and fire[1] is False and t - self.blind_cd > 2.0:
            if w0 and w0.get("kind") == "gun" and w0.get("ammo", 0) >= 5 and w0["cooldown"] <= 0.05:
                muzzle = (mx, my, mz_h)
                for g in self.ghosts:
                    if t - g["t"] > 2.0:
                        continue
                    if g["kind"] not in ("gun", "mortar", "lightning", "explosion", "laser"):
                        continue
                    d = math.hypot(g["x"] - mx, g["y"] - my)
                    if d < 15.0 or d > 170.0:
                        continue
                    gz = self.map.hgt(g["x"], g["y"]) + 4.5
                    if not self.map.los(muzzle[0], muzzle[1], muzzle[2], g["x"], g["y"], gz):
                        continue
                    yaw, pit = self._aim_at(muzzle, (g["x"], g["y"], gz))
                    ay, ap = me["aim"]
                    if abs(angdiff(yaw, ay)) < 0.25 and abs(pit - ap) < 0.25:
                        fire[0] = True
                        aim = (yaw, pit)
                        self.blind_cd = t
                        break
        return (aim, fire)

    # ------------------------------------------------------------ objectives
    def _objective(self, obs, me, t):
        mx, my, _ = me["pos"]
        best = None
        best_sc = 0.0
        eid = me["id"]
        # supply drops
        for dr in obs.get("drops", []):
            if dr["lands_in"] > 0.5:
                continue
            cx, cy, _ = dr["pos"]
            d = math.hypot(cx - mx, cy - my)
            holder = dr.get("holder")
            if holder == eid and d < dr["radius"]:
                return (cx, cy), 3.0  # stand still and capture
            if holder is None and d < 260.0:
                contested = False
                for m2 in obs.get("mechs", []):
                    if m2.get("kind") == "mech" and math.hypot(m2["pos"][0] - cx, m2["pos"][1] - cy) < dr["radius"] + 6.0:
                        contested = True
                        break
                if not contested:
                    sc = 200.0 - d * 0.3
                    if sc > best_sc:
                        best_sc = sc
                        best = (cx, cy)
        # early cube sources: drones and crates
        if t < 220.0 and me["hp"] > 300.0:
            for m2 in obs.get("mechs", []):
                if m2.get("kind") == "npc":
                    d = math.hypot(m2["pos"][0] - mx, m2["pos"][1] - my)
                    if d < 100.0:
                        sc = 120.0 - d * 0.5
                        if sc > best_sc:
                            best_sc = sc
                            best = (m2["pos"][0], m2["pos"][1])
            for c in obs.get("crates", []):
                d = math.hypot(c["pos"][0] - mx, c["pos"][1] - my)
                if d < 140.0:
                    sc = 110.0 - d * 0.4
                    if sc > best_sc:
                        best_sc = sc
                        best = (c["pos"][0], c["pos"][1])
        # visible cubes
        for c in obs.get("cubes", []):
            d = math.hypot(c["pos"][0] - mx, c["pos"][1] - my)
            if d < 70.0:
                sc = 60.0 - d * 0.3
                if sc > best_sc:
                    best_sc = sc
                    best = (c["pos"][0], c["pos"][1])
        # weapon pickups when the second slot is empty
        if self.w1 is None:
            for p in obs.get("pickups", []):
                if p["kind"] == "gun":
                    continue
                d = math.hypot(p["pos"][0] - mx, p["pos"][1] - my)
                if d < 120.0:
                    sc = 90.0 - d * 0.35
                    if sc > best_sc:
                        best_sc = sc
                        best = (p["pos"][0], p["pos"][1])
        if best is None:
            return (None, 0.0)
        return (best, 1.0)

    # ------------------------------------------------------------ path following
    @staticmethod
    def _seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 < 1e-9:
            return math.hypot(px - ax, py - ay)
        f = clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
        return math.hypot(px - (ax + f * dx), py - (ay + f * dy))

    def _off_path(self, mx, my):
        if not self.path:
            return 999.0
        best = 999.0
        prev = None
        start = max(0, self.path_idx - 2)
        for i in range(start, len(self.path)):
            x, y = self.path[i]
            if prev is not None:
                d = self._seg_dist(mx, my, prev[0], prev[1], x, y)
                if d < best:
                    best = d
            else:
                d = math.hypot(x - mx, y - my)
                if d < best:
                    best = d
            prev = (x, y)
        return best

    def _replan(self, me, goal, force=False):
        if goal is None:
            self.path = None
            return
        mx, my, _ = me["pos"]
        gx, gy = goal
        t = self.t0
        if math.hypot(gx - mx, gy - my) < 3.0:
            self.path = None
            self.path_goal = goal
            return
        if not force:
            if self.path and self.path_goal is not None:
                if math.hypot(gx - self.path_goal[0], gy - self.path_goal[1]) < 14.0:
                    if self._off_path(mx, my) < 8.0:
                        return
            elif t - self.path_plan_t < 1.5:
                return
        self.path_plan_t = t
        self.path_goal = goal
        self.path = self.map.astar(mx, my, gx, gy)
        self.path_idx = 0

    def _follow_path(self, mx, my):
        if not self.path:
            return None
        n = len(self.path)
        while self.path_idx < n - 1:
            x, y = self.path[self.path_idx + 1]
            if math.hypot(x - mx, y - my) > 5.5:
                break
            self.path_idx += 1
        for i in range(self.path_idx, n):
            x, y = self.path[i]
            if math.hypot(x - mx, y - my) > 8.0:
                return (x, y)
        return self.path[-1]

    def _stuck(self, me, t):
        mx, my, _ = me["pos"]
        if t - self.stuck_t >= 1.0:
            moved = math.hypot(mx - self.stuck_pos[0], my - self.stuck_pos[1])
            self.stuck_t = t
            self.stuck_pos = (mx, my)
            if moved < 0.7:
                self.path = None
                self.path_plan_t = -1e9
                return True
        return False

    def _wall_ahead(self, me, mv):
        if abs(mv[0]) + abs(mv[1]) < 0.5:
            return False
        mx, my, mz = me["pos"]
        ax, ay = mx + mv[0] * 4.5, my + mv[1] * 4.5
        h = self.map.hgt(ax, ay)
        if h - mz > 2.0:
            return True
        return self.map.wall_at(ax, ay, h)

    def _break_los_goal(self, me, obs, tgt_id):
        mx, my, mz = me["pos"]
        if tgt_id is None:
            return None
        e = self.enemies.get(tgt_id)
        if e is None:
            return None
        ex, ey, ez = e["pos"]
        d = math.hypot(ex - mx, ey - my)
        if d > self.danger_radius:
            return None

        enemy_height = ez + 4.5
        best = None
        best_score = -1e9
        for r in (12.0, 18.0, 24.0, 30.0):
            for k in range(16):
                a = k * TWO_PI / 16.0
                cx = mx + r * math.cos(a)
                cy = my + r * math.sin(a)
                cz = self.map.hgt(cx, cy)
                if not self.map.clear(mx, my, cx, cy):
                    continue
                if self.map.los(ex, ey, enemy_height, cx, cy, cz + 4.5,
                                [(c["pos"][0], c["pos"][1]) for c in obs.get("crates", [])]):
                    continue
                moved = math.hypot(cx - mx, cy - my)
                away = ((mx - ex) * (cx - mx) + (my - ey) * (cy - my)) / max(d, 1.0)
                score = 100.0 - moved - max(0.0, away) * 8.0
                if score > best_score:
                    best_score = score
                    best = (cx, cy)
        return best

    # ------------------------------------------------------------ movement
    def _move(self, obs, me, t, tgt_id, engaged, dodge, zone_goal, zone_urg):
        mx, my, _ = me["pos"]
        energy = me["energy"]
        weights = []
        boost = False
        jump = False
        # 1. primary positional goal
        goal = None
        goal_w = 0.0
        cover_goal = None
        if zone_urg < 40:
            cover_goal = self._break_los_goal(me, obs, tgt_id)
        if cover_goal is not None:
            goal, goal_w = cover_goal, 5.0
            if energy > self.energy_floor:
                boost = True
        elif zone_urg >= 40:
            goal, goal_w = zone_goal, 3.0
        elif zone_urg >= 12:
            goal, goal_w = zone_goal, 1.6
        elif tgt_id is None:
            og, ow = self._objective(obs, me, t)
            if og is not None:
                goal, goal_w = og, max(ow, 1.0)
            elif zone_goal is not None and zone_urg >= 1:
                d = math.hypot(zone_goal[0] - mx, zone_goal[1] - my)
                if d > 70.0:
                    goal, goal_w = zone_goal, 0.5
        if goal is not None:
            self._replan(me, goal, force=(zone_urg >= 40))
            la = self._follow_path(mx, my)
            if la is not None:
                d = math.hypot(la[0] - mx, la[1] - my)
                if d > 3.0:
                    weights.append((unit(la[0] - mx, la[1] - my), goal_w))
                    if zone_urg >= 40 and energy > self.energy_floor:
                        boost = True
            else:
                d = math.hypot(goal[0] - mx, goal[1] - my)
                if d > 6.0:
                    weights.append((unit(goal[0] - mx, goal[1] - my), goal_w))
                    if zone_urg >= 40 and energy > self.energy_floor and d > 50.0:
                        boost = True
        # 2. combat movement
        if tgt_id is not None and cover_goal is None:
            e = self.enemies.get(tgt_id)
            if e is not None:
                ex, ey, _ = e["pos"]
                d = math.hypot(ex - mx, ey - my)
                ux, uy = unit(ex - mx, ey - my)
                if engaged:
                    if t - self.strafe_flip_t > 1.6:
                        self.strafe_flip_t = t
                        self.strafe_dir = -self.strafe_dir
                    px_, py_ = -uy * self.strafe_dir, ux * self.strafe_dir
                    if e["kind"] == "npc":
                        lo, hi = 25.0, 60.0
                    else:
                        lo, hi = 35.0, 130.0
                    if d < lo:
                        weights.append(((-ux, -uy), 1.4))
                    elif d > hi:
                        weights.append(((ux, uy), 1.2))
                    weights.append(((px_, py_), 1.0))
                else:
                    weights.append(((-ux, -uy), 2.4))
                    if energy > self.energy_floor and d < 120.0:
                        boost = True
        # 3. dodge (dominant)
        if dodge is not None:
            dx, dy, urg = dodge
            weights.append(((dx, dy), 3.5 + 2.0 * urg))
            if urg > 0.6 and energy > 15.0:
                boost = True
        mv = (0.0, 0.0)
        if weights:
            sx = sum(w[0][0] * w[1] for w in weights)
            sy = sum(w[0][1] * w[1] for w in weights)
            mv = unit(sx, sy)
        # 4. stuck / climb: jet over walls and ledges
        if me["grounded"] and energy > self.energy_floor:
            if self._stuck(me, t) or self._wall_ahead(me, mv):
                boost = True
                jump = True
        return (mv, boost, jump)

    # ------------------------------------------------------------ main
    def act(self, obs):
        t = obs["t"]
        me = obs["self"]
        mx, my, _ = me["pos"]
        act = {}
        self._update_state(obs, me, t)
        offer = obs.get("offer")
        if offer:
            act["pick"] = self._pick_offer(offer, me)
        self._loot(act, obs, me, mx, my)
        zone_goal, zone_urg = self._zone_goal(obs, me)
        dodge = self._dodge(obs, me, mx, my, me["pos"][2])
        tgt = self._select_target(obs, me, t, zone_urg)
        self.target = tgt
        engaged = tgt is not None and self._engage(me, self.enemies[tgt], t)
        aim, fire = self._combat(act, obs, me, t, tgt, engaged, zone_goal)
        mv, boost, jump = self._move(obs, me, t, tgt, engaged, dodge, zone_goal, zone_urg)
        act["move"] = [round(mv[0], 4), round(mv[1], 4)]
        act["boost"] = bool(boost)
        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        if fire[0] or fire[1]:
            act["fire"] = [bool(fire[0]), bool(fire[1])]
        if jump:
            act["jump"] = True
        return act


# ---------------------------------------------------------------- selftest
def _selftest(stream_path):
    import gzip
    import time

    with gzip.open(stream_path, "rt") as f:
        msgs = [json.loads(l) for l in f if l.strip()]
    hello = msgs[0]
    mdir = hello["map"]["dir"]
    if not os.path.isabs(mdir):  # the raw stream uses a path relative to the check folder
        hello["map"]["dir"] = os.path.abspath(os.path.join(os.path.dirname(stream_path), mdir))
    t0 = time.perf_counter()
    bot = Bot(hello)
    print(f"map built in {(time.perf_counter() - t0) * 1000:.0f} ms")
    times = []
    for m in msgs:
        if m.get("type") != "obs":
            continue
        t1 = time.perf_counter()
        bot.act(m)
        times.append((time.perf_counter() - t1) * 1000.0)
    times.sort()
    n = len(times)
    print(f"{n} obs: p50 {times[n // 2]:.2f}  p90 {times[int(n * 0.9)]:.2f}  "
          f"p99 {times[int(n * 0.99)]:.2f}  max {times[-1]:.2f} ms")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2])
    else:
        mf.run(Bot)
