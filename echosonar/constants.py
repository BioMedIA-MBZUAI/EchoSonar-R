import torch

COARSE_VIEWS = [
    "A2C",
    "A3C",
    "A4C",
    "A5C",
    "Apical_Doppler",
    "Doppler_Parasternal_Long",
    "Doppler_Parasternal_Short",
    "Parasternal_Long",
    "Parasternal_Short",
    "SSN",
    "Subcostal",
]
NUM_VIEWS = len(COARSE_VIEWS)

CLIP_DIM = 512
NUM_FRAMES = 16
VIDEO_SIZE = 224

ENCODED_STUDY_DIM = CLIP_DIM + NUM_VIEWS  # 523

NUM_MIL_SECTIONS = 15

TOKEN_DIM = 768
NUM_TOKENS = 393

ENCODER_MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
ENCODER_STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

ENCODER_WEIGHTS = "echo_prime_encoder.pt"
VIEW_CLASSIFIER_WEIGHTS = "view_classifier.pt"

RT_DETR_CLASSES = [
    "Left Ventricle",
    "Left Atrium",
    "Right Atrium",
    "Right Ventricle",
    "Mitral Valve",
    "Tricuspid Valve",
    "LVOT Area",
]
DETR_DIM = 256 
DETR_IMGSZ = 640
DETR_CONF_THRESH = 0.25
NUM_QUERIES = 300


REPORT_PLACEHOLDER_TOKEN_ID = 151643

FINE_VIEW_NAMES = [
    "A2C",
    "A2C Color on MV",
    "A3C",
    "A3C Color on AV",
    "A4C",
    "A4C Zoomed LV",
    "PLAX Mitral Cusps",
    "PLAX Standard",
    "PSAX Color on Pulmonary",
    "PSAX Zoomed Out",
]
NUM_FINE_VIEWS = len(FINE_VIEW_NAMES)
FINE_VIEW_CLASSIFIER_WEIGHTS = "view_classifier.pt"
