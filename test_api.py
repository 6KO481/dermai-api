"""
Script de test pour l'API DermAI
Testons l'API localement et en production
"""
import requests
import json
from pathlib import Path

# Configuration - Changez selon votre environnement
# API_URL = "http://localhost:8000"  # Local
API_URL = "https://votre-app.onrender.com"  # Production Render

def test_health():
    """Test du endpoint health"""
    print("🧪 Test du endpoint /health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_info():
    """Test du endpoint info"""
    print("🧪 Test du endpoint /info...")
    try:
        response = requests.get(f"{API_URL}/info", timeout=10)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_classes():
    """Test du endpoint classes"""
    print("🧪 Test du endpoint /classes...")
    try:
        response = requests.get(f"{API_URL}/classes", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Nombre de classes: {data['total_classes']}")
            print()
            return True
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_prediction(image_path):
    """Test du endpoint predict"""
    print(f"🧪 Test de prédiction avec {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image non trouvée: {image_path}")
        return False
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prédiction: {result['prediction']}")
            print(f"   Confiance: {result['confidence_percentage']}")
            print(f"   Sévérité: {result['severity']}")
            print(f"   Recommandation: {result['recommendation']}")
            print()
            return True
        else:
            print(f"❌ Erreur: {response.text}")
            print()
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_invalid_file():
    """Test avec un fichier non-image"""
    print("🧪 Test avec fichier invalide...")
    
    try:
        # Créer un faux fichier texte
        files = {"file": ("test.txt", b"This is not an image", "text/plain")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        
        print(f"Status: {response.status_code}")
        print(f"Réponse: {response.json()}")
        print()
        
        # On s'attend à une erreur 400
        return response.status_code == 400
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Tests de l'API DermAI")
    print(f"📍 URL: {API_URL}")
    print("=" * 60)
    print()
    
    results = []
    
    # Tests de base
    results.append(("Health Check", test_health()))
    results.append(("Info", test_info()))
    results.append(("Classes", test_classes()))
    
    # Test de prédiction (décommenter et remplacer par votre image)
    # results.append(("Prediction", test_prediction("examples/test_image.jpg")))
    
    # Test de validation
    results.append(("Invalid File", test_invalid_file()))
    
    print("=" * 60)
    print("📊 Résumé des tests")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print()
    print(f"Total: {passed}/{total} tests passés")
    print("=" * 60)