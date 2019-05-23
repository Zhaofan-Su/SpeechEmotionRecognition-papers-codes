import os


class Config:
    def __init__(self, model_type):

        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.join(self.ROOT_DIR, model_type)
        self.MODEL_PATH = os.path.join(self.BASE_DIR, 'models')
        self.DATA_PATH = os.path.join(self.BASE_DIR, 'database')
        self.PREDICT_PATH = os.path.join(self.BASE_DIR, 'predicts')
        self.SCALER_PATH = os.path.join(self.BASE_DIR, 'scalers')
        self.EMOTION_LABELS_6 = [
            'neutral', 'fear', 'sad', 'surprise', 'angry', 'happy'
        ]
        self.EMOTION_LABELS_3 = ['neutral', 'positive', 'negative']
