"""
Classes de prédiction pour le système de classification des lésions cutanées
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from .config import (
    THREE_PARTITION_CLASSES,
    MALIGNANT_CLASSES,
    KERATINOCYTE_CLASSES,
    IMG_SIZE
)


class ViTFeatureExtractor:
    """Extracteur de features ViT pour les deux modèles"""

    def __init__(self):
        from transformers import ViTImageProcessor, ViTModel

        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # Freeze le modèle
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        print("✅ ViT Feature Extractor initialisé")

    def extract_features(self, image):
        """Extrait les features pour une seule image"""
        import torch

        # Conversion en PIL Image si nécessaire
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Prétraitement avec ViT processor
        inputs = self.processor(images=pil_image, return_tensors="pt")

        # Extraction des features
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        return features.cpu().numpy()[0]  # Retourne un array 1D


class Model1Predictor:
    """Prédicteur pour le modèle 1 (classification générale)"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.feature_extractor = ViTFeatureExtractor()
        self.model = None
        self.load_model()

    def load_model(self):
        """Charge le modèle 1"""
        print(f"📥 Chargement du modèle 1 depuis {self.model_path}...")
        self.model = keras.models.load_model(self.model_path, compile=False)
        print("✅ Modèle 1 chargé avec succès")

    def preprocess_image(self, image_path):
        """Prétraite une image pour la prédiction"""
        try:
            # Chargement de l'image
            if isinstance(image_path, str):
                image = tf.keras.preprocessing.image.load_img(
                    image_path, target_size=IMG_SIZE
                )
            elif isinstance(image_path, Image.Image):
                image = image_path.resize(IMG_SIZE)
            else:
                # Si c'est déjà un array numpy
                if isinstance(image_path, np.ndarray):
                    if image_path.max() <= 1.0:
                        image_path = (image_path * 255).astype(np.uint8)
                    image = Image.fromarray(image_path).resize(IMG_SIZE)
                else:
                    raise ValueError("Format d'image non supporté")

            # Conversion en array et normalisation
            image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
            return image_array, image

        except Exception as e:
            print(f"❌ Erreur lors du prétraitement de l'image: {e}")
            return None, None

    def predict(self, image_path):
        """
        Fait une prédiction avec le modèle 1

        Returns:
            dict: Résultats de la prédiction du modèle 1
        """
        # Prétraitement de l'image
        image_array, original_image = self.preprocess_image(image_path)
        if image_array is None:
            return None

        # Extraction des features ViT
        vit_features = self.feature_extractor.extract_features(image_array)
        vit_features = np.expand_dims(vit_features, axis=0)  # Ajouter dimension batch

        # Prédiction
        predictions = self.model.predict(vit_features, verbose=0)

        # Gestion flexible des sorties du modèle
        three_part_probs = None

        if isinstance(predictions, list):
            # Si c'est une liste, prendre le deuxième élément (index 1) pour three_part_output
            if len(predictions) >= 2:
                three_part_probs = predictions[1][0]  # [1] pour three_part_output, [0] pour premier batch
            else:
                # Si seulement un élément, l'utiliser
                three_part_probs = predictions[0][0]
        elif isinstance(predictions, dict):
            # Si c'est un dictionnaire, chercher la clé three_part_output
            if 'three_part_output' in predictions:
                three_part_probs = predictions['three_part_output'][0]
            elif 'three_part_output' not in predictions and len(predictions) > 0:
                # Prendre la première valeur du dictionnaire
                three_part_probs = list(predictions.values())[0][0]
        else:
            # Si c'est un array simple
            three_part_probs = predictions[0]

        if three_part_probs is None:
            print("❌ Impossible de trouver les probabilités three_part_output")
            return None

        # Obtenir la classe prédite et la probabilité
        three_part_pred = np.argmax(three_part_probs)
        predicted_class = THREE_PARTITION_CLASSES[three_part_pred]
        confidence = float(three_part_probs[three_part_pred])

        results = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                THREE_PARTITION_CLASSES[i]: float(prob) for i, prob in enumerate(three_part_probs)
            },
            'image_array': image_array,
            'original_image': original_image
        }

        return results


class Model2Predictor:
    """Prédicteur pour le modèle 2 (classification des lésions malignes)"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.feature_extractor = ViTFeatureExtractor()
        self.model = None
        self.load_model()

    def load_model(self):
        """Charge le modèle 2"""
        print(f"📥 Chargement du modèle 2 depuis {self.model_path}...")
        self.model = keras.models.load_model(self.model_path, compile=False)
        print("✅ Modèle 2 chargé avec succès")

    def predict(self, image_array):
        """
        Fait une prédiction avec le modèle 2 sur une image déjà prétraitée

        Args:
            image_array: Array de l'image prétraitée (de Model1)

        Returns:
            dict: Résultats de la prédiction du modèle 2
        """
        # Extraction des features ViT
        vit_features = self.feature_extractor.extract_features(image_array)
        vit_features = np.expand_dims(vit_features, axis=0)  # Ajouter dimension batch

        # Prédiction
        predictions = self.model.predict(vit_features, verbose=0)

        # Gestion flexible des sorties
        malignant_probs = None

        if isinstance(predictions, list):
            # Si c'est une liste, prendre le premier élément
            malignant_probs = predictions[0][0]  # [0] pour premier élément de liste, [0] pour premier batch
        else:
            # Si c'est un array simple
            malignant_probs = predictions[0]

        if malignant_probs is None:
            print("❌ Impossible de trouver les probabilités du modèle 2")
            return None

        # Obtenir la classe prédite et la probabilité
        malignant_pred = np.argmax(malignant_probs)

        # Vérifier si nous avons le bon nombre de classes
        if len(malignant_probs) == len(MALIGNANT_CLASSES):
            original_class = MALIGNANT_CLASSES[malignant_pred]
        else:
            # Si le nombre ne correspond pas, utiliser des indices génériques
            print(f"⚠️  Nombre de classes inattendu: {len(malignant_probs)} au lieu de {len(MALIGNANT_CLASSES)}")
            original_class = f"class_{malignant_pred}"

        confidence = float(malignant_probs[malignant_pred])

        # Appliquer la logique de regroupement
        if original_class in KERATINOCYTE_CLASSES:
            final_class = 'keratinocytes'
        else:
            final_class = original_class

        # Créer le dictionnaire des probabilités
        all_probs = {}
        if len(malignant_probs) == len(MALIGNANT_CLASSES):
            for i, cls in enumerate(MALIGNANT_CLASSES):
                all_probs[cls] = float(malignant_probs[i])
        else:
            for i in range(len(malignant_probs)):
                all_probs[f"class_{i}"] = float(malignant_probs[i])

        results = {
            'original_class': original_class,
            'final_class': final_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }

        return results


class EnsembleSkinLesionPredictor:
    """Prédicteur ensemble qui combine les deux modèles"""

    def __init__(self, model1_path, model2_path):
        self.model1_predictor = Model1Predictor(model1_path)
        self.model2_predictor = Model2Predictor(model2_path)
        print("✅ Ensemble Predictor initialisé avec succès")

    def predict(self, image_path, show_results=True):
        """
        Fait une prédiction complète en utilisant les deux modèles

        Args:
            image_path: Chemin vers l'image ou objet PIL Image
            show_results: Si True, affiche les résultats détaillés

        Returns:
            dict: Résultats complets de la prédiction
        """
        # Étape 1: Prédiction avec le modèle 1
        model1_results = self.model1_predictor.predict(image_path)
        if model1_results is None:
            print("❌ Échec de la prédiction avec le modèle 1")
            return None

        predicted_class = model1_results['predicted_class']
        confidence = model1_results['confidence']

        if show_results:
            print(f"📊 Résultat modèle 1: {predicted_class} (confiance: {confidence:.4f})")

        # Application de la logique de décision
        if predicted_class in ['benign', 'non-neoplastic']:
            final_prediction = 'non_cancerous_lesion'
            final_confidence = confidence
            model2_results = None
            detailed_class = predicted_class
            if show_results:
                print("🔀 Logique: benign/non-neoplastic → non_cancerous_lesion")

        elif predicted_class == 'malignant':
            if show_results:
                print("🔀 Logique: malignant → application du modèle 2")
            # Étape 2: Prédiction avec le modèle 2
            model2_results = self.model2_predictor.predict(model1_results['image_array'])
            if model2_results is None:
                print("❌ Échec de la prédiction avec le modèle 2")
                return None

            final_prediction = model2_results['final_class']
            final_confidence = model2_results['confidence']
            detailed_class = model2_results['original_class']
            if show_results:
                print(f"📊 Résultat modèle 2: {detailed_class} → {final_prediction} (confiance: {final_confidence:.4f})")

        else:  # 'healthy'
            final_prediction = 'healthy'
            final_confidence = confidence
            model2_results = None
            detailed_class = predicted_class
            if show_results:
                print("🔀 Logique: healthy → conservé tel quel")

        # Formater la confiance en pourcentage avec 2 décimales
        confidence_percentage = f"{final_confidence * 100:.2f}%"

        # Préparer les résultats finaux
        final_results = {
            'final_prediction': final_prediction,
            'final_confidence': final_confidence,
            'confidence_percentage': confidence_percentage,
            'model1_prediction': predicted_class,
            'model1_confidence': confidence,
            'model2_results': model2_results,
            'detailed_class': detailed_class,
            'image_array': model1_results['image_array'],
            'original_image': model1_results['original_image']
        }

        return final_results