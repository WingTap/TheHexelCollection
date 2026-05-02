import pygame
import pymunk
import math
import random
import pickle

# --- Initialize ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Triangle Ship Fight")

# --- Colors ---
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# --- Physics ---
space = pymunk.Space()
space.gravity = (0, 0)

# --- Helpers ---
def clamp_pos(x, y, margin=20):
    x = max(margin, min(WIDTH - margin, x))
    y = max(margin, min(HEIGHT - margin, y))
    return x, y

def angle_diff_signed(a, b):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return d

# --- Ship class ---
class Ship:
    def __init__(self, x, y, color):
        self.color = color
        moment = pymunk.moment_for_circle(1, 0, 20)
        self.body = pymunk.Body(1, moment)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, 20)
        self.shape.sensor = True
        space.add(self.body, self.shape)
        self.angle = 0.0
        self.bullets = []
        self.fire_timer = 0.0  # cooldown timer for firing

    def draw(self):
        x, y = self.body.position
        points = [
            (x + 20 * math.cos(self.angle), y + 20 * math.sin(self.angle)),
            (x + 15 * math.cos(self.angle + 2.5), y + 15 * math.sin(self.angle + 2.5)),
            (x + 15 * math.cos(self.angle - 2.5), y + 15 * math.sin(self.angle - 2.5))
        ]
        pygame.draw.polygon(screen, self.color, points)

    def move_forward(self, dt, thrust=200):
        fx = math.cos(self.angle) * thrust
        fy = math.sin(self.angle) * thrust
        self.body.velocity = (fx, fy)

    def rotate(self, direction, speed=10.0, dt=1/60):
        self.angle += direction * speed * dt
        self.angle %= 2 * math.pi

    def shoot(self):
        bx = self.body.position.x + math.cos(self.angle) * 30
        by = self.body.position.y + math.sin(self.angle) * 30
        vel = (math.cos(self.angle) * 500, math.sin(self.angle) * 500)
        self.bullets.append([bx, by, vel[0], vel[1]])

    def update_bullets(self, dt, target_ship=None):
        new_bullets = []
        hit = False
        for b in self.bullets:
            b[0] += b[2] * dt
            b[1] += b[3] * dt
            if 0 < b[0] < WIDTH and 0 < b[1] < HEIGHT:
                if target_ship is not None:
                    tx, ty = target_ship.body.position
                    if math.hypot(b[0]-tx, b[1]-ty) < 20:
                        hit = True
                        continue
                pygame.draw.circle(screen, WHITE, (int(b[0]), int(b[1])), 4)
                new_bullets.append(b)
        self.bullets = new_bullets
        return hit

    def clamp_inside(self):
        x, y = self.body.position
        clamped_x, clamped_y = clamp_pos(x, y)
        hit_wall = (clamped_x != x) or (clamped_y != y)
        if hit_wall:
            cx, cy = WIDTH/2, HEIGHT/2
            dirx = cx - clamped_x
            diry = cy - clamped_y
            norm = math.hypot(dirx, diry) or 1.0
            nx, ny = dirx / norm, diry / norm
            self.body.position = (clamped_x + nx*5, clamped_y + ny*5)
            self.body.velocity = (nx*120, ny*120)
        else:
            self.body.velocity = (self.body.velocity[0]*0.95, self.body.velocity[1]*0.95)

# --- Q-learning AI ---
class AI:
    def __init__(self, ship, q_file='q_table.pkl'):
        self.ship = ship
        self.q_table = {}
        self.q_file = q_file
        self.epsilon = 0.4
        self.alpha = 0.2
        self.gamma = 0.9
        self.actions = ["forward", "rotate_left", "rotate_right", "shoot", "idle"]
        # load if exists
        try:
            with open(self.q_file, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            pass

    def save_q(self):
        with open(self.q_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def discretize_state(self, player):
        dx = player.body.position.x - self.ship.body.position.x
        dy = player.body.position.y - self.ship.body.position.y
        dist = math.hypot(dx, dy)
        angle_to_player = math.atan2(dy, dx)
        ang_diff = angle_diff_signed(self.ship.angle, angle_to_player)
        dist_bucket = min(7, int(dist // 80))
        angle_bucket = int((ang_diff + math.pi) // (math.pi / 4))
        min_edge = min(self.ship.body.position.x, WIDTH - self.ship.body.position.x,
                       self.ship.body.position.y, HEIGHT - self.ship.body.position.y)
        wall_near = 1 if min_edge < 80 else 0
        return (dist_bucket, angle_bucket, wall_near)

    def ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

    def choose_action(self, state):
        self.ensure_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        best_val = max(self.q_table[state].values())
        bests = [a for a, v in self.q_table[state].items() if v == best_val]
        return random.choice(bests)

    def update_q(self, state, action, reward, next_state):
        self.ensure_state(state)
        self.ensure_state(next_state)
        old = self.q_table[state][action]
        best_next = max(self.q_table[next_state].values())
        self.q_table[state][action] = old + self.alpha * (reward + self.gamma * best_next - old)

# --- Create ships and agent ---
player = Ship(100, HEIGHT//2, BLUE)
ai_ship = Ship(WIDTH-100, HEIGHT//2, RED)
ai_agent = AI(ai_ship)

# --- Episode / reset ---
def reset_round():
    player.body.position = (100, HEIGHT//2)
    player.body.velocity = (0, 0)
    player.angle = 0.0
    player.bullets = []
    player.fire_timer = 0.0
    ai_ship.body.position = (WIDTH-100, HEIGHT//2)
    ai_ship.body.velocity = (0, 0)
    ai_ship.angle = math.pi
    ai_ship.bullets = []

reset_round()
episode_count = 0
last_print = 0

# --- Main loop ---
running = True
while running:
    dt = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Player input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player.move_forward(dt)
    if keys[pygame.K_LEFT]:
        player.rotate(-1, dt=dt)
    if keys[pygame.K_RIGHT]:
        player.rotate(1, dt=dt)
    player.fire_timer = max(0.0, player.fire_timer - dt)
    if keys[pygame.K_SPACE]:
        if player.fire_timer <= 0.0:
            player.shoot()
            player.fire_timer = 0.5

    # AI decision
    state = ai_agent.discretize_state(player)
    action = ai_agent.choose_action(state)

    # Apply AI action (existing logic remains)
    dx = player.body.position.x - ai_ship.body.position.x
    dy = player.body.position.y - ai_ship.body.position.y
    angle_to_player = math.atan2(dy, dx)
    ang_diff = angle_diff_signed(ai_ship.angle, angle_to_player)
    min_edge = min(ai_ship.body.position.x, WIDTH - ai_ship.body.position.x,
                   ai_ship.body.position.y, HEIGHT - ai_ship.body.position.y)

    if action == "forward":
        if min_edge < 70:
            ang_to_player = angle_to_player
            angc = angle_diff_signed(ai_ship.angle, ang_to_player)
            if abs(angc) > 0.2:
                if angc > 0:
                    ai_ship.rotate(1, dt=dt)
                else:
                    ai_ship.rotate(-1, dt=dt)
            else:
                vx = math.cos(ang_to_player) * 220
                vy = math.sin(ang_to_player) * 220
                ai_ship.body.velocity = (vx, vy)
        else:
            ai_ship.move_forward(dt)
    elif action == "rotate_left":
        ai_ship.rotate(-1, dt=dt)
    elif action == "rotate_right":
        ai_ship.rotate(1, dt=dt)
    elif action == "shoot":
        if abs(ang_diff) < 0.35:
            ai_ship.shoot()
        else:
            if ang_diff > 0:
                ai_ship.rotate(1, dt=dt)
            else:
                ai_ship.rotate(-1, dt=dt)

    dx = player.body.position.x - ai_ship.body.position.x
    dy = player.body.position.y - ai_ship.body.position.y
    old_dist = math.hypot(dx, dy)

    space.step(dt)
    player.clamp_inside()
    ai_ship.clamp_inside()

    ai_hit_player = ai_ship.update_bullets(dt, target_ship=player)
    player_hit_ai = player.update_bullets(dt, target_ship=ai_ship)

    reward = 0.0
    dx2 = player.body.position.x - ai_ship.body.position.x
    dy2 = player.body.position.y - ai_ship.body.position.y
    new_dist = math.hypot(dx2, dy2)
    shaping = (old_dist - new_dist) * 0.2
    reward += shaping
    min_edge2 = min(ai_ship.body.position.x, WIDTH - ai_ship.body.position.x,
                    ai_ship.body.position.y, HEIGHT - ai_ship.body.position.y)
    if min_edge2 < 60:
        reward -= (60 - min_edge2) * 0.5

    next_state = ai_agent.discretize_state(player)

    if ai_hit_player:
        reward += 100.0
        ai_agent.update_q(state, action, reward, next_state)
        ai_agent.save_q()
        episode_count += 1
        ai_agent.epsilon = max(0.02, ai_agent.epsilon * 0.995)
        reset_round()
        continue
    if player_hit_ai:
        reward -= 100.0
        ai_agent.update_q(state, action, reward, next_state)
        ai_agent.save_q()
        episode_count += 1
        ai_agent.epsilon = max(0.02, ai_agent.epsilon * 0.995)
        reset_round()
        continue

    ai_agent.update_q(state, action, reward, next_state)

    screen.fill(BLACK)
    player.draw()
    ai_ship.draw()
    player.update_bullets(dt)
    ai_ship.update_bullets(dt)

    last_print += 1
    if last_print > 120:
        last_print = 0
        print(f"Episode: {episode_count}, Epsilon: {ai_agent.epsilon:.3f}, Q states: {len(ai_agent.q_table)}")

    pygame.display.flip()

pygame.quit()

