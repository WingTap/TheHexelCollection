import pygame, pymunk, math, random, pickle
import pennylane as qml
import pennylane.numpy as np

# =============================
# GAME SETUP
# =============================
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("QRL vs QL Self-Play")

BLACK=(0,0,0); RED=(255,0,0); BLUE=(0,0,255); WHITE=(255,255,255)

space = pymunk.Space()
space.gravity=(0,0)

def angle_diff(a,b):
    return (b-a+math.pi)%(2*math.pi)-math.pi

def clamp_pos(x,y,m=20):
    return max(m,min(WIDTH-m,x)), max(m,min(HEIGHT-m,y))

# =============================
# SHIP CLASS
# =============================
class Ship:
    def __init__(self, x, y, color):
        moment = pymunk.moment_for_circle(1,0,20)
        self.body = pymunk.Body(1, moment)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body,20)
        self.shape.sensor=True
        space.add(self.body,self.shape)
        self.angle=0.0
        self.color=color
        self.bullets=[]
        self.fire_timer=0.0

    def draw(self):
        x,y=self.body.position
        pts=[
            (x+20*math.cos(self.angle), y+20*math.sin(self.angle)),
            (x+15*math.cos(self.angle+2.5), y+15*math.sin(self.angle+2.5)),
            (x+15*math.cos(self.angle-2.5), y+15*math.sin(self.angle-2.5)),
        ]
        pygame.draw.polygon(screen,self.color,pts)

    def move_forward(self,dt,thrust=220):
        self.body.velocity=(math.cos(self.angle)*thrust,math.sin(self.angle)*thrust)

    def rotate(self,dir,dt):
        self.angle=(self.angle+dir*10*dt)%(2*math.pi)

    def shoot(self):
        bx=self.body.position.x+math.cos(self.angle)*30
        by=self.body.position.y+math.sin(self.angle)*30
        vx=math.cos(self.angle)*500; vy=math.sin(self.angle)*500
        self.bullets.append([bx,by,vx,vy])

    def update_bullets(self,dt,target=None):
        new=[]; hit=False
        for b in self.bullets:
            b[0]+=b[2]*dt; b[1]+=b[3]*dt
            if 0<b[0]<WIDTH and 0<b[1]<HEIGHT:
                if target:
                    tx,ty=target.body.position
                    if math.hypot(b[0]-tx,b[1]-ty)<20:
                        hit=True; continue
                pygame.draw.circle(screen,WHITE,(int(b[0]),int(b[1])),4)
                new.append(b)
        self.bullets=new
        return hit

    def clamp_inside(self):
        x,y=self.body.position
        cx,cy=clamp_pos(x,y)
        if cx!=x or cy!=y:
            dx,dy = WIDTH/2-cx, HEIGHT/2-cy
            n=math.hypot(dx,dy) or 1
            self.body.position=(cx+dx/n*5,cy+dy/n*5)
            self.body.velocity=(dx/n*120,dy/n*120)
        else:
            self.body.velocity=(self.body.velocity[0]*0.95,self.body.velocity[1]*0.95)

# =============================
# RED: Q-LEARNING
# =============================
ACTIONS=["forward","rotate_left","rotate_right","shoot","idle"]
Q = {}
alpha=0.2
gamma=0.9
epsilon=0.3

def discretize_state(ai,op):
    dx=op.body.position.x-ai.body.position.x
    dy=op.body.position.y-ai.body.position.y
    dist=math.hypot(dx,dy)
    angle_to=math.atan2(dy,dx)
    ang=angle_diff(ai.angle,angle_to)
    dist_b=min(7,int(dist//80))
    ang_b=int((ang+math.pi)//(math.pi/4))
    min_edge=min(ai.body.position.x,WIDTH-ai.body.position.x,
                 ai.body.position.y,HEIGHT-ai.body.position.y)
    wall=1 if min_edge<80 else 0
    return (dist_b,ang_b,wall)

def ensure(state):
    if state not in Q:
        Q[state]={a:0.0 for a in ACTIONS}

def choose_action(state):
    ensure(state)
    if random.random()<epsilon:
        return random.choice(ACTIONS)
    best=max(Q[state].values())
    bests=[a for a,v in Q[state].items() if v==best]
    return random.choice(bests)

def update_q(s,a,r,s2):
    ensure(s); ensure(s2)
    old=Q[s][a]
    best=max(Q[s2].values())
    Q[s][a]=old+alpha*(r+gamma*best-old)

# =============================
# BLUE: QUANTUM RL (POLICY GRADIENT)
# =============================
dev=qml.device("default.qubit",wires=3)

@qml.qnode(dev)
def circuit(p, st):
    qml.RY(st[0],wires=0)
    qml.RY(st[1],wires=1)
    qml.RY(st[2],wires=2)
    for i in range(3):
        qml.RY(p[i],wires=i)
    for i in range(3):
        qml.CNOT(wires=[i,(i+1)%3])
    return qml.probs(wires=[0,1,2])

params=np.random.uniform(0,0.1,3)
optimizer=qml.GradientDescentOptimizer(0.05)

def state_vec(ai,op):
    dx=op.body.position.x-ai.body.position.x
    dy=op.body.position.y-ai.body.position.y
    dist=(math.hypot(dx,dy)/400.0)
    ang=angle_diff(ai.angle,math.atan2(dy,dx))/math.pi
    min_edge=min(ai.body.position.x,WIDTH-ai.body.position.x,
                 ai.body.position.y,HEIGHT-ai.body.position.y)
    wall=1.0 if min_edge<80 else 0.0
    return np.array([dist,ang,wall])

def quantum_policy(p,st):
    probs=circuit(p,st)[:5]
    probs=probs/np.sum(probs)
    return probs

# For policy gradient, store episode transitions
episode_states=[]
episode_actions=[]
episode_rewards=[]

def finish_episode():
    global params
    G=0
    returns=[]
    for r in reversed(episode_rewards):
        G=r+0.9*G
        returns.insert(0,G)
    returns=np.array(returns)
    returns=(returns-returns.mean())/(returns.std()+1e-6)

    def loss_fn(p):
        L=0
        for st,ac,Gt in zip(episode_states,episode_actions,returns):
            probs=quantum_policy(p,st)
            L-=np.log(probs[ac]+1e-9)*Gt
        return L

    params = optimizer.step(loss_fn, params)

    episode_states.clear()
    episode_actions.clear()
    episode_rewards.clear()

# =============================
# CREATE SHIPS
# =============================
QLship = Ship(100,HEIGHT//2,RED)
QRLship = Ship(WIDTH-100,HEIGHT//2,BLUE)
QRLship.angle=math.pi

def reset():
    QLship.body.position=(100,HEIGHT//2)
    QLship.body.velocity=(0,0)
    QLship.angle=0
    QLship.bullets=[]

    QRLship.body.position=(WIDTH-100,HEIGHT//2)
    QRLship.body.velocity=(0,0)
    QRLship.angle=math.pi
    QRLship.bullets=[]

# =============================
# MAIN LOOP
# =============================
running=True
episode_len=0
while running:
    dt=clock.tick(60)/1000.0
    for e in pygame.event.get():
        if e.type==pygame.QUIT:
            running=False

    # ---- RED Q-Learning ----
    s = discretize_state(QLship,QRLship)
    a = choose_action(s)

    dx=QRLship.body.position.x-QLship.body.position.x
    dy=QRLship.body.position.y-QLship.body.position.y
    ang_to=math.atan2(dy,dx)
    angd=angle_diff(QLship.angle,ang_to)

    if a=="forward": QLship.move_forward(dt)
    elif a=="rotate_left": QLship.rotate(-1,dt)
    elif a=="rotate_right": QLship.rotate(1,dt)
    elif a=="shoot" and abs(angd)<0.3: QLship.shoot()

    # ---- BLUE QRL ----
    st = state_vec(QRLship,QLship)
    probs=quantum_policy(params,st)
    act=np.random.choice(len(probs),p=probs)
    action=ACTIONS[act]

    dx2=QLship.body.position.x-QRLship.body.position.x
    dy2=QLship.body.position.y-QRLship.body.position.y
    ang_to2=math.atan2(dy2,dx2)
    angd2=angle_diff(QRLship.angle,ang_to2)

    if action=="forward": QRLship.move_forward(dt)
    elif action=="rotate_left": QRLship.rotate(-1,dt)
    elif action=="rotate_right": QRLship.rotate(1,dt)
    elif action=="shoot" and abs(angd2)<0.3: QRLship.shoot()

    # Physics
    space.step(dt)
    QLship.clamp_inside()
    QRLship.clamp_inside()

    hit_red = QRLship.update_bullets(dt,target=QLship)
    hit_blue = QLship.update_bullets(dt,target=QRLship)

    # Rewards
    reward_red = 0
    reward_blue = 0

    if hit_blue:
        reward_red += 100
        reward_blue -= 100
    if hit_red:
        reward_red -= 100
        reward_blue += 100

    # QL UPDATE
    s2 = discretize_state(QLship,QRLship)
    update_q(s,a,reward_red,s2)

    # QRL STORE STEP
    episode_states.append(st)
    episode_actions.append(ACTIONS.index(action))
    episode_rewards.append(reward_blue)

    # Episode end
    if hit_red or hit_blue:
        finish_episode()
        reset()
        episode_len=0

    # Draw
    screen.fill(BLACK)
    QLship.draw()
    QRLship.draw()
    QLship.update_bullets(dt)
    QRLship.update_bullets(dt)
    pygame.display.flip()

pygame.quit()

