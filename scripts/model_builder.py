# scripts/model_builder.py
# @author: Y.S

# Model
import tensorflow as tf
from tensorflow.keras.applications import (MobileNetV2, InceptionV3, DenseNet121) # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

IMG_HEIGHT = 224
IMG_WIDTH = 224

def build_model(model_name):
    """Build and return the appropriate model architecture."""
    
    if model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    # Freeze the base model layers
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)  # Binary classification

    # Create the final model
    model = Model(inputs=base_model.input, outputs=x)

    return model

def fine_tune_model(model, model_name):
    """Unfreeze the base model layers for fine-tuning."""
    
    # Unfreeze all layers in the base model to allow fine-tuning
    base_model = model.layers[0]  # Access the base model
    base_model.trainable = True
    
    # Re-compile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model
