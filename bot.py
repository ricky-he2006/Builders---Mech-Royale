#!/usr/bin/env python3
import sys
import json
import math
import heapq
import os


def send(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def readline():
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def pip(x, y, ring):
    inside = False
    count = len(ring) // 2
    j = count - 1
    for i in range(count):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


class Nav:
    """Coarse 5 m A* over prism footprints; direct steering is the fallback."""

    CELL = 5.0

    def __init__(self, map_dir):
        self.ready = False
        self.blocked = None
        try:
            with open(os.path.join(map_dir, "v2", "world.json")) as handle:
                world = json.load(handle)
        except Exception:
            return

        terrain = world.get("terrain", {})
        bounds = world.get("bounds", {})
        if isinstance(bounds, dict):
            x0 = bounds.get("minX", -600)
            y0 = bounds.get("minY", -600)
            x1 = bounds.get("maxX", 600)
            y1 = bounds.get("maxY", 600)
        else:
            x0, y0, x1, y1 = bounds

        x0 = terrain.get("minX", x0)
        y0 = terrain.get("minY", y0)
        if "nx" in terrain and "step" in terrain:
            x1 = max(x1, x0 + terrain["nx"] * terrain["step"])
            y1 = max(y1, y0 + terrain["ny"] * terrain["step"])

        self.x0, self.y0 = x0, y0
        self.nx = int((x1 - x0) / self.CELL) + 1
        self.ny = int((y1 - y0) / self.CELL) + 1
        if self.nx <= 0 or self.ny <= 0 or self.nx * self.ny > 300000:
            return

        self.blocked = bytearray(self.nx * self.ny)
        for prism in world.get("prisms", []):
            base = prism.get("base", 0.0)
            top = prism.get("top", 0.0)
            if top < 1.2 or base > 8.5:
                continue
            ring = prism.get("ring", [])
            if len(ring) < 6:
                continue
            xs = ring[0::2]
            ys = ring[1::2]
            i0 = max(0, int((min(xs) - x0) / self.CELL) - 1)
            i1 = min(self.nx - 1, int((max(xs) - x0) / self.CELL) + 1)
            j0 = max(0, int((min(ys) - y0) / self.CELL) - 1)
            j1 = min(self.ny - 1, int((max(ys) - y0) / self.CELL) + 1)
            holes = [hole for hole in prism.get("holes", []) if len(hole) >= 6]
            for j in range(j0, j1 + 1):
                cy = y0 + (j + 0.5) * self.CELL
                for i in range(i0, i1 + 1):
                    cx = x0 + (i + 0.5) * self.CELL
                    if pip(cx, cy, ring) and not any(pip(cx, cy, hole) for hole in holes):
                        self.blocked[j * self.nx + i] = 1
        self.ready = True

    def cell(self, x, y):
        i = int((x - self.x0) / self.CELL)
        j = int((y - self.y0) / self.CELL)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def blocked_xy(self, x, y):
        cell = self.cell(x, y)
        return bool(cell) and self.blocked[cell[1] * self.nx + cell[0]] == 1

    def los(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        distance = math.hypot(dx, dy)
        steps = max(2, int(distance / 2.5))
        for k in range(1, steps):
            f = k / steps
            if self.blocked_xy(a[0] + dx * f, a[1] + dy * f):
                return False
        return True

    def astar(self, start, goal, max_expand=14000):
        if not self.ready:
            return None
        start_cell = self.cell(*start)
        goal_cell = self.cell(*goal)
        if not start_cell or not goal_cell:
            return None
        if self.blocked_xy(*goal):
            best = None
            best_distance = 1e9
            for dj in range(-4, 5):
                for di in range(-4, 5):
                    ni = goal_cell[0] + di
                    nj = goal_cell[1] + dj
                    if (
                        0 <= ni < self.nx
                        and 0 <= nj < self.ny
                        and not self.blocked[nj * self.nx + ni]
                    ):
                        distance = di * di + dj * dj
                        if distance < best_distance:
                            best_distance = distance
                            best = ni, nj
            if not best:
                return None
            goal_cell = best

        open_heap = [(0, start_cell)]
        came = {}
        cost = {start_cell: 0}
        gx, gy = goal_cell
        while open_heap and len(came) < max_expand:
            _, current = heapq.heappop(open_heap)
            if current == goal_cell:
                path = [current]
                while current in came:
                    current = came[current]
                    path.append(current)
                path.reverse()
                waypoints = [
                    (
                        self.x0 + (i + 0.5) * self.CELL,
                        self.y0 + (j + 0.5) * self.CELL,
                    )
                    for i, j in path
                ]
                waypoints[-1] = goal
                output = [waypoints[0]]
                for k in range(1, len(waypoints) - 1):
                    ax, ay = output[-1]
                    bx, by = waypoints[k]
                    cx, cy = waypoints[k + 1]
                    if abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > 1e-3:
                        output.append(waypoints[k])
                output.append(waypoints[-1])
                return output

            for di, dj in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                ni, nj = current[0] + di, current[1] + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                    continue
                if self.blocked[nj * self.nx + ni]:
                    continue
                if di and dj:
                    if self.blocked[current[1] * self.nx + ni]:
                        continue
                    if self.blocked[nj * self.nx + current[0]]:
                        continue
                new_cost = cost[current] + (1.414 if di and dj else 1.0)
                neighbor = ni, nj
                if new_cost < cost.get(neighbor, 1e18):
                    cost[neighbor] = new_cost
                    came[neighbor] = current
                    heapq.heappush(
                        open_heap,
                        (
                            new_cost + math.hypot(ni - gx, nj - gy),
                            neighbor,
                        ),
                    )
        return None


class Bot:
    def __init__(self, hello):
        self.slot = hello["slot"]
        self.name = "farm-then-fight"
        self.nav = Nav(hello["map"]["dir"])
        self.path = []
        self.goal = None
        self.pos = [0, 0, 0]
        self.last_zone = None
        self.last_aim = [0.0, 0.0]

    def zone_goal(self):
        zone = self.last_zone
        next_x, next_y = zone["next_center"]
        next_radius = max(zone["next_radius"], 1.0)
        dx = self.pos[0] - next_x
        dy = self.pos[1] - next_y
        distance = math.hypot(dx, dy)
        if distance < 1.0:
            dx, dy, distance = 1.0, 0.0, 1.0
        scale = next_radius * (0.65 if distance > next_radius * 0.82 else 0.80)
        return (
            next_x + dx / distance * scale,
            next_y + dy / distance * scale,
        )

    def steer_to(self, goal):
        if self.goal is None or math.dist(self.goal, goal) > 8:
            self.goal = goal
            self.path = (
                self.nav.astar(self.pos[:2], goal) or []
            ) if self.nav.ready else []
        target = goal
        if self.path:
            while self.path and math.dist(self.pos[:2], self.path[0]) < 5:
                self.path.pop(0)
            if self.path:
                target = self.path[0]
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        distance = math.hypot(dx, dy)
        if distance < 1:
            return [0.0, 0.0]
        return [dx / distance, dy / distance]

    def aim_at(self, mech, target, lead=True):
        muzzle_height = mech["pos"][2] + mech.get("stats", {}).get(
            "muzzle_height", 7.0
        )
        speed = 252.0
        for weapon in mech.get("weapons", []) or []:
            if weapon and weapon.get("kind") == "gun":
                speed = weapon["stats"].get("speed", 252.0)
                break
        target_velocity = (
            list(target.get("vel", [0, 0, 0])) if lead else [0, 0, 0]
        )
        predicted = list(target["pos"])
        for _ in range(3):
            horizontal = math.hypot(
                predicted[0] - mech["pos"][0],
                predicted[1] - mech["pos"][1],
            )
            flight_time = horizontal / max(speed, 1.0)
            predicted = [
                target["pos"][0] + target_velocity[0] * flight_time,
                target["pos"][1] + target_velocity[1] * flight_time,
                target["pos"][2] + target_velocity[2] * flight_time,
            ]
        target_height = 9.0 if target.get("kind") == "mech" else 3.5
        dz = predicted[2] + target_height * 0.5 - muzzle_height
        horizontal = math.hypot(
            predicted[0] - mech["pos"][0],
            predicted[1] - mech["pos"][1],
        )
        return (
            [
                math.atan2(
                    predicted[1] - mech["pos"][1],
                    predicted[0] - mech["pos"][0],
                ),
                math.atan2(dz, max(horizontal, 0.1)),
            ],
            horizontal,
        )

    def fire_for(self, mech, target, distance):
        fire = [False, False]
        weapons = (mech.get("weapons") or [None, None])[:2]
        for i, weapon in enumerate(weapons):
            if not weapon:
                continue
            kind = weapon.get("kind")
            stats = weapon.get("stats", {})
            if kind == "gun":
                if (
                    distance < stats.get("range", 230)
                    and weapon.get("cooldown", 0) <= 0
                    and weapon.get("ammo", 0) > 0
                ):
                    fire[i] = True
            elif kind == "lightning":
                if distance < 45 and mech.get("energy", 0) >= 3:
                    fire[i] = True
            elif kind == "laser":
                if weapon.get("charge", 0) < 0.60:
                    fire[i] = True
        return fire

    def should_engage(self, mech, enemy, hp_fraction, distance):
        if hp_fraction < 0.50:
            return False

        held = [
            weapon["kind"]
            for weapon in (mech.get("weapons") or [])
            if weapon
        ]
        enemy_weapons = enemy.get("weapons") or []
        enemy_hp = enemy.get("hp", 1400)
        enemy_max_hp = max(enemy.get("max_hp", 1400), 1)
        if enemy_hp / enemy_max_hp < 0.35 and distance < 230:
            return True
        if "lightning" in held and "laser" in enemy_weapons and distance < 45:
            return True
        if "laser" in held and "mortar" in enemy_weapons and distance < 280:
            return True
        angle = math.atan2(
            self.pos[1] - enemy["pos"][1],
            self.pos[0] - enemy["pos"][0],
        )
        difference = abs(
            (angle - enemy.get("aim", [0, 0])[0] + math.pi) % (2 * math.pi)
            - math.pi
        )
        if difference > math.pi * 0.60 and distance < 180:
            return True
        if (
            mech.get("cubes", 0) >= 6
            and mech.get("cubes", 0) > enemy.get("cubes", 0)
            and distance < 120
        ):
            return True
        return distance < 70 and hp_fraction > 0.75

    def choose_upgrade(self, offer, mech):
        best = 0
        best_score = -1
        held = [
            weapon["kind"]
            for weapon in (mech.get("weapons") or [])
            if weapon
        ]
        for i, choice in enumerate(offer.get("choices", [])):
            name = (
                choice.get("id", "") + " " + choice.get("name", "")
            ).lower()
            score = 0
            if any(keyword in name for keyword in ("gun", "damage", "rate", "feed")):
                score = 5
            elif any(keyword in name for keyword in ("armor", "plating")):
                score = 4
            elif any(keyword in name for keyword in ("sensor", "vision")):
                score = 3
            elif any(keyword in name for keyword in ("speed", "boost", "jet", "energy")):
                score = 3
            elif any(keyword in name for keyword in ("laser", "lightning", "mortar", "cannon", "arc")):
                score = 4 if any(kind in name for kind in held) else 1
            if score > best_score:
                best_score = score
                best = i
        return best

    def act(self, observation):
        mech = observation["self"]
        self.pos = list(mech["pos"])
        self.last_zone = observation["zone"]
        zone = observation["zone"]
        hp = mech["hp"]
        max_hp = max(mech.get("max_hp", 1400), 1)
        hp_fraction = hp / max_hp

        mechs = [
            item
            for item in observation.get("mechs", [])
            if item.get("kind") == "mech"
        ]
        npcs = [
            item
            for item in observation.get("mechs", [])
            if item.get("kind") == "npc"
        ]

        enemy = None
        enemy_distance = 1e18
        for item in mechs:
            distance = math.dist(self.pos[:2], item["pos"][:2])
            if distance < enemy_distance:
                enemy_distance = distance
                enemy = item

        npc = None
        npc_distance = 1e18
        for item in npcs:
            distance = math.dist(self.pos[:2], item["pos"][:2])
            if distance < npc_distance:
                npc_distance = distance
                npc = item

        next_x, next_y = zone["next_center"]
        next_radius = max(zone["next_radius"], 1.0)
        next_distance = math.hypot(
            self.pos[0] - next_x,
            self.pos[1] - next_y,
        )

        action = {"tick": observation["tick"]}
        move = [0.0, 0.0]
        aim = self.last_aim[:]
        fire = [False, False]
        boost = False
        reload = False
        take = None
        pick = None

        if observation.get("offer"):
            pick = self.choose_upgrade(observation["offer"], mech)

        if not mech.get("in_zone", True) or next_distance > next_radius * 0.82:
            goal = self.zone_goal()
            move = self.steer_to(goal)
            boost = (
                math.dist(self.pos[:2], goal) > 30
                and mech.get("energy", 0) > 40
            )
        else:
            drop = None
            for candidate in observation.get("drops", []) or []:
                if (
                    candidate.get("lands_in", 1) <= 0
                    and candidate.get("holder") in (None, mech["id"])
                ):
                    distance = math.dist(
                        self.pos[:2],
                        candidate["pos"][:2],
                    )
                    if distance < 70 and (enemy is None or enemy_distance > 25):
                        drop = candidate
                        break

            if drop:
                move = self.steer_to(drop["pos"])
            else:
                target = None
                if (
                    enemy
                    and enemy_distance < 230
                    and (self.nav.los(self.pos, enemy["pos"]) or not self.nav.ready)
                    and self.should_engage(
                        mech,
                        enemy,
                        hp_fraction,
                        enemy_distance,
                    )
                ):
                    target = enemy
                elif npc and npc_distance < 55 and (
                    enemy is None or enemy_distance > 40
                ):
                    target = npc

                if target:
                    aim, horizontal_distance = self.aim_at(mech, target)
                    fire = self.fire_for(mech, target, horizontal_distance)

                    dx = self.pos[0] - target["pos"][0]
                    dy = self.pos[1] - target["pos"][1]
                    distance = math.hypot(dx, dy) or 1.0
                    if distance < 55:
                        gx = next_x - self.pos[0]
                        gy = next_y - self.pos[1]
                        gd = math.hypot(gx, gy) or 1.0
                        move = [gx / gd, gy / gd]
                    elif distance > 160:
                        move = [-dx / distance, -dy / distance]
                    else:
                        move = [-dy / distance * 0.7, dx / distance * 0.7]

                    gun = (mech.get("weapons") or [None])[0]
                    if gun and gun.get("ammo", 1) == 0:
                        reload = True
                else:
                    got = False
                    if enemy is None or enemy_distance > 35:
                        for cube in observation.get("cubes", []) or []:
                            if (
                                math.dist(
                                    self.pos[:2],
                                    cube["pos"][:2],
                                )
                                < 25
                            ):
                                move = self.steer_to(cube["pos"])
                                got = True
                                break

                    if not got:
                        nearest_crate = None
                        nearest_crate_distance = 1e18
                        if enemy is None or enemy_distance > 45:
                            for crate in observation.get("crates", []) or []:
                                distance = math.dist(
                                    self.pos[:2],
                                    crate["pos"][:2],
                                )
                                if distance < nearest_crate_distance:
                                    nearest_crate_distance = distance
                                    nearest_crate = crate
                        if nearest_crate and nearest_crate_distance < 55:
                            move = self.steer_to(nearest_crate["pos"])
                            got = True

                    if not got:
                        weapons = mech.get("weapons") or [None, None]
                        has_empty_slot = len(weapons) < 2 or weapons[1] is None
                        if has_empty_slot:
                            for pickup in observation.get("pickups", []) or []:
                                if pickup.get("kind") in ("laser", "lightning"):
                                    distance = math.dist(
                                        self.pos[:2],
                                        pickup["pos"][:2],
                                    )
                                    if distance < 20 and (
                                        enemy is None or enemy_distance > 30
                                    ):
                                        move = self.steer_to(pickup["pos"])
                                        got = True
                                        if distance < 4:
                                            take = pickup["id"]
                                        break

                    if not got:
                        goal = self.zone_goal()
                        move = self.steer_to(goal)
                        boost = (
                            math.dist(self.pos[:2], goal) > 35
                            and mech.get("energy", 0) > 45
                        )

        if enemy and enemy_distance < 45 and hp_fraction < 0.50:
            aim, _ = self.aim_at(mech, enemy, lead=False)
            fire = [False, False]
            dx = self.pos[0] - enemy["pos"][0]
            dy = self.pos[1] - enemy["pos"][1]
            distance = math.hypot(dx, dy) or 1.0
            move = [dx / distance, dy / distance]

        move_length = math.hypot(move[0], move[1])
        if move_length > 1.0:
            move = [move[0] / move_length, move[1] / move_length]

        action.update(
            move=move,
            aim=aim,
            fire=fire,
            boost=bool(boost),
            jump=False,
            reload=bool(reload),
        )
        if take is not None:
            action["take"] = take
        if pick is not None:
            action["pick"] = pick
        self.last_aim = aim
        return action


def main():
    bot = None
    while True:
        message = readline()
        if message is None:
            break
        message_type = message.get("type")
        if message_type == "hello":
            bot = Bot(message)
            send({"type": "ready", "name": bot.name})
        elif message_type == "obs" and bot is not None:
            try:
                action = bot.act(message)
            except Exception as error:
                print("bot error:", error, file=sys.stderr)
                action = {"tick": message.get("tick", 0)}
            send(action)
        elif message_type == "end":
            break


if __name__ == "__main__":
    main()
