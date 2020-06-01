from ajctr.helpers import timing
from ajctr.models.fm import train_fm_model
from ajctr.models.logit import train_logistic_model
from ajctr.models.boosting import train_gradientboosting_model


@timing
def train():
    train_logistic_model()
    train_gradientboosting_model()
    train_fm_model()
