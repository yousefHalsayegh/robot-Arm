#brain stuff
LEARNING_RATE = 0.00003
BATCH = 32
EPS_END = 0.05
EPS_START = 1.0
EPS_DECAY = 15000000
WARMUP = 1000
GAMMA = 0.99
TAU = 0.005
SLOW = 0.067
UPDATES = 1
CAPACITY = 250000
CHECKPOINT = ""
ITERATION = 10

#lerobot bit
ACTION_PER_CHUNK = 25
CHUNK_SIZE = 0.25
AGGREGATE = "weighted_average"
POLICY = "/home/yousef/Documents/robot-Arm/outputs/train/smolvla_fighter_V1.66/checkpoints/040000/pretrained_model"
LENGTH = 25
STEPS = 10
TIMEOUT = 25
EXECUTION = 0
SWITCH = False

#game stuff
EPISODES = 5000
DISTANCE_REWARD = 0.8
PENALTIY_MOVE = 0.2
MID_SAVE = 100
FULL_SAVE = 5000
CROP = 68
THRESHOLD = 0.15
CENTER_Y = CROP / 2
ENV = 4
CONTROLLER = {
    # Keep in mind this is on the X/Y trgger not DP and the values are for Pong
    ('ABS_Y', 0): 2,   # up 
    ('ABS_Y',  255): 3,   # down
    ('ABS_Y',  128): 0,   # Neutral

    # D-pad horizontal (not used in Pong but mapped anyway)
    ('ABS_X', 0): 4,   # left
    ('ABS_X',  255): 5,   # right
    ('ABS_X',  128): 0, #Neutral

    # Face buttons
    ('BTN_SOUTH', 1): 0,    # A Touch
    ('BTN_EAST',  1): 0,    # B Touch
    ('BTN_NORTH', 1): 0,    # X Touch
    ('BTN_WEST',  1): 0,    # Y Touch

    ('BTN_SOUTH', 0): 0,    # A Release
    ('BTN_EAST',  0): 0,    # B Release
    ('BTN_NORTH', 0): 0,    # X Release
    ('BTN_WEST',  0): 0,    # Y Release

    # Other "Bumpers"
    ('BTN_TR',  1): 0,      # RB Touch
    ('BTN_TL',  1): 0,      # LB Touch
    ('BTN_TR2', 1): 0,      # RT Touch
    ('BTN_TL2', 1): 0,      # LT Touch
    ('BTN_TR',  0): 0,      # RB Release
    ('BTN_TL',  0): 0,      # LB Release
    ('BTN_TR2', 0): 0,      # RT Release
    ('BTN_TL2', 0): 0,      # LT Release
}