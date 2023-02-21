from stardist.models import StarDist2D
from csbdeep.utils import Path, normalize
from keras.models import load_model


def get_stardist_model(name='stardist_model',directory= './model_fiber_segmentation/'):
    model = StarDist2D(None, name='stardist', basedir='models')
    return model
def stardist_predict(img):
    model = get_stardist_model()
    axis_norm = (0,1)
    img_norm = normalize(img, 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img_norm)
    return labels, details
def stardist_predict_with_class(img):
    model = get_stardist_model()
    axis_norm = (0,1)
    img_norm = normalize(img, 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img_norm)
    return labels, details


def get_unet_model(name='FR_model',directory= './model_FR/'):
    model = load_model(directory+name)
    return model
def unet_predict(img):
    model = def get_unet_model()
    img_norm= np.array(img)/255
    img_ready= np.expand_dims(img_norm,0)
    prediction = (model.predict(img_ready)[0,:,:,0] > 0.5).astype(np.uint8)
    return prediction

def get_classifier_model(name='classification_model',directory= './model_fiber_classification/'):
    model = load_model(directory+name)
    return model
def classifier_predict(ind_fiber_img):
    model = get_classifier_model()
    prediction = model.predict(ind_fiber_img)
    prediction_NN = np.argmax(prediction, axis=-1)
    prediction_NN = le.inverse_transform(prediction_NN)
    return prediction_NN
    