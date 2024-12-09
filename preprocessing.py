import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    """Preprocesa una imagen para la entrada del modelo."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Aplicar transformaciones
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Añadir dimensión del batch

def load_images_from_folder(folder_path):
    """Carga y preprocesa imágenes desde carpetas para entrenamiento."""
    images = []
    labels = []
    class_names = []
    
    # Obtener todas las subcarpetas (clases)
    class_folders = sorted([d for d in os.listdir(folder_path) 
                          if os.path.isdir(os.path.join(folder_path, d))])
    
    for idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(folder_path, class_folder)
        class_names.append(class_folder)
        
        # Obtener todas las imágenes de la carpeta
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            try:
                # Cargar y preprocesar imagen
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Aplicar transformaciones y mantener como tensor
                image_tensor = transform(image)
                images.append(image_tensor)
                labels.append(idx)
            except Exception as e:
                print(f"Error procesando {image_path}: {str(e)}")
    
    # Apilar todos los tensores en un solo tensor
    X_train = torch.stack(images)
    # Convertir a numpy manteniendo el orden correcto de los canales
    X_train = X_train.numpy()
    y_train = np.array(labels)
    
    return X_train, y_train, class_names
