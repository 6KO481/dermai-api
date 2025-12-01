"""
Script de test pour l'API DermAI
Testons l'API localement et en production
"""
import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"  # Local
# API_URL = "https://votre-app.up.railway.app"  # Production Railway

def test_health():
    """Test du endpoint health"""
    print("🧪 Test du endpoint /health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_info():
    """Test du endpoint info"""
    print("🧪 Test du endpoint /info...")
    response = requests.get(f"{API_URL}/info")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_classes():
    """Test du endpoint classes"""
    print("🧪 Test du endpoint /classes...")
    response = requests.get(f"{API_URL}/classes")
    print(f"Status: {response.status_code}")
    print(f"Nombre de classes: {response.json()['total_classes']}")
    print()

def test_prediction(image_path):
    """Test du endpoint predict"""
    print(f"🧪 Test de prédiction avec {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prédiction: {result['prediction']}")
        print(f"   Confiance: {result['confidence_percentage']}")
        print(f"   Sévérité: {result['severity']}")
        print(f"   Recommandation: {result['recommendation']}")
    else:
        print(f"❌ Erreur: {response.text}")
    print()

def test_invalid_file():
    """Test avec un fichier non-image"""
    print("🧪 Test avec fichier invalide...")
    
    # Créer un faux fichier texte
    files = {"file": ("test.txt", b"This is not an image", "text/plain")}
    response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Réponse: {response.json()}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Tests de l'API DermAI")
    print("=" * 60)
    print()
    
    # Tests de base
    test_health()
    test_info()
    test_classes()
    
    # Test de prédiction (remplacez par votre image)
    # test_prediction("examples/test_image.jpg")
    
    # Test de validation
    test_invalid_file()
    
    print("=" * 60)
    print("✅ Tests terminés")
    print("=" * 60)