import keras_applications
import keras


keras_applications.set_keras_submodules(
    backend=keras.backend,
    models=keras.models,
    layers=keras.layers,
    utils=keras.utils
)


large_networks = [
    (keras_applications.resnet_v2.ResNet101V2, keras_applications.resnet_v2.preprocess_input),
    (keras_applications.resnet_v2.ResNet152V2, keras_applications.resnet_v2.preprocess_input),
    (keras_applications.resnext.ResNeXt50, keras_applications.resnext.preprocess_input),
    (keras_applications.resnext.ResNeXt101, keras_applications.resnext.preprocess_input),
    (keras_applications.inception_resnet_v2.InceptionResNetV2, keras_applications.inception_resnet_v2.preprocess_input),
    (keras_applications.nasnet.NASNetLarge, keras_applications.nasnet.preprocess_input),
]


small_networks = [
    (keras_applications.vgg16.VGG16, keras_applications.vgg16.preprocess_input),
    (keras_applications.vgg19.VGG19, keras_applications.vgg19.preprocess_input),
    (keras_applications.resnet_v2.ResNet50V2, keras_applications.resnet_v2.preprocess_input),
    (keras_applications.inception_v3.InceptionV3, keras_applications.inception_v3.preprocess_input),
    (keras_applications.densenet.DenseNet121, keras_applications.densenet.preprocess_input),
    (keras_applications.densenet.DenseNet169, keras_applications.densenet.preprocess_input),
    (keras_applications.densenet.DenseNet201, keras_applications.densenet.preprocess_input),
    (keras_applications.xception.Xception, keras_applications.xception.preprocess_input),
    (keras_applications.mobilenet_v2.MobileNetV2, keras_applications.mobilenet_v2.preprocess_input),
    (keras_applications.nasnet.NASNetMobile, keras_applications.nasnet.preprocess_input)
]

networks = small_networks # + large_networks

model_folder = './keras_code/scripts/pretrained_models/'
info_folder = './keras_code/scripts/info/'