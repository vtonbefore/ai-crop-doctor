class Config:
    DATA_DIR = "./data/train"    # Your training data folder, organized by class subfolders
    MODEL_DIR = "./model"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    MODEL_NAME = "crop_disease_model.pth"
    CLASS_NAMES_FILE = "class_names.json"
