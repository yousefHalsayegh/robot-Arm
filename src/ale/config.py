#brain stuff
LEARNING_RATE = 0.00005
BATCH = 64
EPS_END = 0.05
EPS_START = 1.0
EPS_DECAY = 500000
WARMUP = 1000
GAMMA = 0.99
TAU = 0.005

#lerobot bit
ACTION_PER_CHUNK = 50
CHUNK_SIZE = 0.5
AGGREGATE = "weighted_average"
POLICY = "/home/yousef/Documents/lerobot/outputs/train/smolvla_fighter_V1.64/checkpoints/050000/pretrained_model/"

#game stuff
EPISODES = 5000
GOAL_REWARD = 5 
DISTANCE_REWARD = 0.01
PENALTIY_MOVE = 0.5
PENALTIY_CENTER = 0.001
MID_SAVE = 100
FULL_SAVE = 1000
CROP = 68
THRESHOLD = 0.15
CENTER_Y = CROP / 2