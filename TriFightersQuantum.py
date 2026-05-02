import pygame, pymunk, math, random
import pennylane as qml
import pennylane.numpy as np

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Tri Fighters QRL")

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

space = pymunk.Space()
space.gravity = (0, 0)

def clamp_pos(x, y, margin=20):
    return max(margin, min(WIDTH-margin, x)), max(margin, min(HEIGHT-margin, y))

def angle_diff(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi

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
        self.fire_timer = 0.0

    def draw(self):
        x, y = self.body.position
        pts = [
            (x + 20*math.cos(self.angle), y + 20*math.sin(self.angle)),
            (x + 15*math.cos(self.angle+2.5), y + 15*math.sin(self.angle+2.5)),
            (x + 15*math.cos(self.angle-2.5), y + 15*math.sin(self.angle-2.5))
        ]
        pygame.draw.polygon(screen, self.color, pts)

    def move_forward(self, dt, thrust=200):
        self.body.velocity = (math.cos(self.angle)*thrust,math.sin(self.angle)*thrust)

    def rotate(self, d, dt):
        self.angle = (self.angle + d*10*dt) % (2*math.pi)

    def shoot(self):
        bx = self.body.position.x + math.cos(self.angle)*30
        by = self.body.position.y + math.sin(self.angle)*30
        vx = math.cos(self.angle)*500
        vy = math.sin(self.angle)*500
        self.bullets.append([bx, by, vx, vy])

    def update_bullets(self, dt, target=None):
        new = []
        hit = False
        for b in self.bullets:
            b[0]+=b[2]*dt; b[1]+=b[3]*dt
            if 0<b[0]<WIDTH and 0<b[1]<HEIGHT:
                if target:
                    tx, ty = target.body.position
                    if math.hypot(b[0]-tx,b[1]-ty)<20:
                        hit = True
                        continue
                pygame.draw.circle(screen, WHITE,(int(b[0]),int(b[1])),4)
                new.append(b)
        self.bullets=new
        return hit

    def clamp_inside(self):
        x,y=self.body.position
        cx,cy=clamp_pos(x,y)
        if (cx!=x or cy!=y):
            dx,dy = WIDTH/2-cx, HEIGHT/2-cy
            n = math.hypot(dx,dy) or 1
            self.body.position=(cx+dx/n*5,cy+dy/n*5)
            self.body.velocity=(dx/n*120,dy/n*120)
        else:
            self.body.velocity=(self.body.velocity[0]*0.95,self.body.velocity[1]*0.95)

# ---------- QUANTUM POLICY ----------
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def policy_circuit(params, state):
    qml.RY(state[0], wires=0)
    qml.RY(state[1], wires=1)
    qml.RY(state[2], wires=2)
    for i in range(3):
        qml.RY(params[i], wires=i)
    for i in range(3):
        qml.CNOT(wires=[i, (i+1)%3])
    return qml.probs(wires=[0,1,2])

# Maps 8 quantum outputs → 5 action probs
def get_action_probs(params, state):
    probs = policy_circuit(params, state)
    p = probs[:5]
    p = p / np.sum(p)
    return p

ACTIONS = ["forward","rotate_left","rotate_right","shoot","idle"]

# Params for 3 qubit circuit
params = np.ones(3, requires_grad=True)*0.1
optimizer = qml.AdamOptimizer(0.05)

def pick_action(params, state):
    probs = get_action_probs(params, state)
    return np.random.choice(ACTIONS, p=probs)

# ---------- SETUP ----------
player = Ship(100, HEIGHT//2, BLUE)
ai = Ship(WIDTH-100, HEIGHT//2, RED)

def reset():
    player.body.position=(100,HEIGHT//2); player.body.velocity=(0,0)
    player.angle=0; player.bullets=[]
    ai.body.position=(WIDTH-100,HEIGHT//2); ai.body.velocity=(0,0)
    ai.angle=math.pi; ai.bullets=[]

reset()

# ---------- TRAIN LOOP ----------
running=True
episode=0
states=[]
actions=[]
rewards=[]

def get_state():
    dx = player.body.position.x - ai.body.position.x
    dy = player.body.position.y - ai.body.position.y
    dist = math.hypot(dx, dy)/400.0
    ang = angle_diff(ai.angle, math.atan2(dy,dx))/math.pi
    min_edge=min(ai.body.position.x,WIDTH-ai.body.position.x,
                 ai.body.position.y,HEIGHT-ai.body.position.y)
    wall=1.0 if min_edge<80 else 0.0
    return np.array([dist, ang, wall])

while running:
    dt = clock.tick(60)/1000.0
    for e in pygame.event.get():
        if e.type==pygame.QUIT:
            running=False

    # Player control
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]: player.move_forward(dt)
    if keys[pygame.K_LEFT]: player.rotate(-1,dt)
    if keys[pygame.K_RIGHT]: player.rotate(1,dt)
    player.fire_timer=max(0,player.fire_timer-dt)
    if keys[pygame.K_SPACE] and player.fire_timer<=0:
        player.shoot(); player.fire_timer=0.5

    # ------- AI STEP -------
    state = get_state()
    act = pick_action(params, state)
    states.append(state)
    actions.append(act)

    if act=="forward": ai.move_forward(dt)
    elif act=="rotate_left": ai.rotate(-1,dt)
    elif act=="rotate_right": ai.rotate(1,dt)
    elif act=="shoot":
        if abs(angle_diff(ai.angle, math.atan2(
            player.body.position.y-ai.body.position.y,
            player.body.position.x-ai.body.position.x)))<0.3:
            ai.shoot()

    old_dist = math.hypot(player.body.position.x-ai.body.position.x,
                          player.body.position.y-ai.body.position.y)

    space.step(dt)
    player.clamp_inside()
    ai.clamp_inside()

    ai_hit = ai.update_bullets(dt, target=player)
    player_hit = player.update_bullets(dt, target=ai)

    new_dist = math.hypot(player.body.position.x-ai.body.position.x,
                          player.body.position.y-ai.body.position.y)

    r = (old_dist-new_dist)*0.2
    min_edge2=min(ai.body.position.x,WIDTH-ai.body.position.x,
                  ai.body.position.y,HEIGHT-ai.body.position.y)
    if min_edge2<60: r -= (60-min_edge2)*0.5

    if ai_hit: r += 100
    if player_hit: r -= 100

    rewards.append(r)

    # Episode end
    if ai_hit or player_hit:
        episode+=1

        def loss(p):
            L=0
            G=0
            for i in reversed(range(len(rewards))):
                G = rewards[i] + 0.99*G
                probs = get_action_probs(p, states[i])
                a_idx = ACTIONS.index(actions[i])
                L -= np.log(probs[a_idx]+1e-9)*G
            return L

        params = optimizer.step(loss, params)
        states.clear(); actions.clear(); rewards.clear()
        reset()

    # Draw
    screen.fill(BLACK)
    player.draw(); ai.draw()
    player.update_bullets(dt); ai.update_bullets(dt)
    pygame.display.flip()

pygame.quit()
