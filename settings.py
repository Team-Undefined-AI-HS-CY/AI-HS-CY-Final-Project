from string import ascii_uppercase, digits
import os
from dotenv import dotenv_values

env = dotenv_values(".env")
# make them all appear in the global scope
globals().update(env)

DATASETS_DIR = "Datasets"
OCR_DATASETS_DIR = f"{DATASETS_DIR}/OCR"
YOLO_DATASETS_DIR = f"{DATASETS_DIR}/YOLO"
RESULTS_DIR = "Results"
OCR_RESULTS_DIR = f"{RESULTS_DIR}/OCR"
YOLO_RESULTS_DIR = f"{RESULTS_DIR}/YOLO"
MODELS_DIR = "Models"
OCR_MODELS_DIR = f"{MODELS_DIR}/OCR"
YOLO_MODELS_DIR = f"{MODELS_DIR}/YOLO"

# Ensure that all the above directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(OCR_DATASETS_DIR, exist_ok=True)
os.makedirs(YOLO_DATASETS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OCR_RESULTS_DIR, exist_ok=True)
os.makedirs(YOLO_RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OCR_MODELS_DIR, exist_ok=True)
os.makedirs(YOLO_MODELS_DIR, exist_ok=True)

COMBINED_CHARS = digits + ascii_uppercase
# DROP THE O character as it is similar to 0
COMBINED_CHARS = COMBINED_CHARS.replace("O", "")

TRAIN_PERCENT = 0.9
TEST_PERCENT = 1 - TRAIN_PERCENT
VALIDATION_PERCENT = 0.2

RANDOM_STATE = 1

BG_COLOR = 255 # Background color is white
FG_COLOR = 0 # Foreground color is black (LETTERS SHOULD BE BLACK)

X_SIZE = 50#75
Y_SIZE = 100

INPUT_SHAPE = (Y_SIZE, X_SIZE, 1)  # (Y X C) as it is just black and white we can just use a 2D tensor

ENGLISH_FNT_FONTS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    73,
    75,
    77,
    78,
    79,
    80,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    145,
    146,
    147,
    148,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    201,
    202,
    203,
    204,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    233,
    234,
    235,
    236,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    265,
    266,
    269,
    270,
    277,
    278,
    279,
    280,
    281,
    282,
    283,
    284,
    285,
    286,
    287,
    288,
    289,
    290,
    291,
    292,
    305,
    306,
    307,
    308,
    309,
    310,
    311,
    312,
    313,
    314,
    315,
    316,
    317,
    318,
    319,
    320,
    321,
    322,
    323,
    324,
    329,
    330,
    331,
    332,
    333,
    334,
    335,
    336,
    337,
    338,
    339,
    340,
    349,
    350,
    351,
    352,
    353,
    354,
    355,
    356,
    357,
    358,
    359,
    360,
    361,
    362,
    363,
    364,
    365,
    366,
    367,
    368,
    369,
    370,
    371,
    372,
    373,
    374,
    375,
    376,
    377,
    378,
    379,
    380,
    393,
    394,
    395,
    396,
    397,
    399,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    419,
    420,
    421,
    422,
    423,
    424,
    425,
    426,
    427,
    428,
    429,
    430,
    431,
    437,
    438,
    439,
    440,
    441,
    442,
    443,
    444,
    453,
    457,
    458,
    459,
    460,
    465,
    466,
    467,
    468,
    469,
    470,
    471,
    472,
    473,
    474,
    475,
    476,
    493,
    494,
    495,
    496,
    497,
    498,
    499,
    500,
    509,
    510,
    511,
    512,
    513,
    514,
    515,
    516,
    517,
    518,
    519,
    520,
    521,
    522,
    523,
    524,
    533,
    534,
    535,
    536,
    537,
    538,
    539,
    540,
    541,
    542,
    543,
    544,
    545,
    546,
    547,
    548,
    549,
    550,
    551,
    552,
    553,
    554,
    555,
    556,
    557,
    558,
    559,
    560,
    561,
    562,
    563,
    564,
    565,
    566,
    567,
    568,
    569,
    570,
    571,
    572,
    573,
    574,
    575,
    576,
    577,
    578,
    579,
    580,
    581,
    582,
    583,
    584,
    585,
    586,
    587,
    588,
    589,
    590,
    591,
    592,
    593,
    594,
    595,
    596,
    597,
    598,
    599,
    600,
    601,
    602,
    603,
    604,
    605,
    606,
    607,
    608,
    609,
    610,
    611,
    612,
    617,
    618,
    619,
    620,
    621,
    622,
    623,
    624,
    625,
    626,
    627,
    628,
    633,
    634,
    635,
    636,
    637,
    638,
    639,
    640,
    641,
    642,
    643,
    644,
    645,
    646,
    647,
    648,
    649,
    650,
    651,
    652,
    653,
    654,
    655,
    656,
    657,
    658,
    659,
    660,
    661,
    662,
    663,
    664,
    665,
    666,
    667,
    668,
    669,
    670,
    671,
    672,
    673,
    674,
    675,
    676,
    693,
    694,
    695,
    696,
    697,
    698,
    699,
    700,
    701,
    702,
    703,
    704,
    705,
    706,
    707,
    708,
    709,
    710,
    711,
    712,
    713,
    714,
    715,
    716,
    717,
    718,
    719,
    720,
    721,
    722,
    723,
    724,
    725,
    726,
    727,
    728,
    729,
    730,
    731,
    732,
    733,
    734,
    735,
    736,
    737,
    738,
    739,
    740,
    741,
    742,
    743,
    744,
    745,
    746,
    747,
    748,
    749,
    750,
    751,
    752,
    761,
    762,
    763,
    764,
    765,
    766,
    767,
    768,
    769,
    770,
    771,
    772,
    785,
    786,
    787,
    788,
    793,
    794,
    795,
    796,
    797,
    798,
    799,
    800,
    809,
    810,
    811,
    812,
    813,
    814,
    815,
    816,
    817,
    818,
    819,
    820,
    829,
    830,
    831,
    832,
    833,
    834,
    835,
    836,
    845,
    846,
    847,
    848,
    849,
    850,
    851,
    852,
    857,
    858,
    859,
    860,
    861,
    862,
    863,
    864,
    865,
    866,
    867,
    868,
    869,
    870,
    871,
    872,
    881,
    882,
    883,
    893,
    894,
    895,
    896,
    901,
    902,
    903,
    904,
    905,
    906,
    907,
    908,
    909,
    910,
    911,
    912,
    913,
    914,
    915,
    916,
    917,
    918,
    919,
    920,
    921,
    922,
    923,
    924,
    925,
    926,
    927,
    928,
    937,
    938,
    939,
    940,
    941,
    942,
    943,
    944,
    945,
    946,
    947,
    948,
    953,
    954,
    955,
    956,
    957,
    958,
    959,
    960,
    961,
    962,
    963,
    964,
    965,
    966,
    967,
    968,
    969,
    970,
    971,
    972,
    973,
    974,
    975,
    976,
    977,
    978,
    979,
    980,
    981,
    982,
    983,
    984,
    985,
    986,
    987,
    988,
    1001,
    1002,
    1003,
    1004,
    1009,
    1010,
    1011,
    1012,
    1013,
    1014,
    1015,
    1016,
]

BOUNDING_BOX_SCALE = 1.15
