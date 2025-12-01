# 🚀 Déploiement DermAI API sur Railway

Guide complet pour déployer votre API de classification de lésions cutanées.

## 📋 Prérequis

- Compte GitHub
- Compte Railway (gratuit)
- Vos 2 modèles ML (<50 Mo)
- Python 3.9+

## 🗂️ Structure du projet

```
dermai-api/
├── main.py                 # API FastAPI
├── predictor.py            # Vos classes de prédiction
├── config.py               # Configuration
├── requirements.txt        # Dépendances Python
├── railway.toml            # Config Railway
├── Procfile               # Commande de démarrage
├── .env.example           # Variables d'environnement
├── .gitignore             # Fichiers à ignorer
├── test_api.py            # Tests
└── models/
    ├── model1_general.h5
    └── model2_malignant.h5
```

## 🔧 Installation locale

### 1. Cloner le projet
```bash
git clone https://github.com/votre-username/dermai-api.git
cd dermai-api
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ajouter vos modèles
Placez vos fichiers `.h5` dans le dossier `models/`

### 5. Lancer l'API localement
```bash
uvicorn main:app --reload
```

Accédez à : http://localhost:8000/docs

## 🚂 Déploiement sur Railway

### Méthode 1 : Via GitHub (Recommandée)

#### 1. Créer un repository GitHub
```bash
git init
git add .
git commit -m "Initial commit - DermAI API"
git branch -M main
git remote add origin https://github.com/votre-username/dermai-api.git
git push -u origin main
```

#### 2. Connecter Railway
1. Allez sur [railway.app](https://railway.app)
2. Cliquez sur "Start a New Project"
3. Sélectionnez "Deploy from GitHub repo"
4. Choisissez votre repository `dermai-api`

#### 3. Configuration Railway
Railway détecte automatiquement votre `railway.toml` et `Procfile`

Variables d'environnement à ajouter (optionnel) :
- `PORT` : 8000 (Railway le gère automatiquement)
- `PYTHON_VERSION` : 3.9

#### 4. Déployer
Railway build et déploie automatiquement ! 🎉

Votre API sera disponible à : `https://votre-app.up.railway.app`

### Méthode 2 : Railway CLI

```bash
# Installer Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialiser
railway init

# Déployer
railway up
```

## 🧪 Tester l'API en production

```bash
# Modifier test_api.py avec votre URL Railway
API_URL = "https://votre-app.up.railway.app"

# Lancer les tests
python test_api.py
```

## 📡 Endpoints disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/health` | GET | Status de l'API |
| `/info` | GET | Infos sur l'app |
| `/classes` | GET | Classes disponibles |
| `/predict` | POST | Prédiction sur image |
| `/docs` | GET | Documentation Swagger |

## 💻 Intégration dans vos sites/apps

### JavaScript/TypeScript
```javascript
async function predictLesion(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('https://votre-app.up.railway.app/predict', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log('Prédiction:', result.prediction);
  console.log('Confiance:', result.confidence_percentage);
  return result;
}
```

### React Example
```jsx
import { useState } from 'react';

function SkinLesionClassifier() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('https://votre-app.up.railway.app/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Erreur:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {loading && <p>Analyse en cours...</p>}
      {result && (
        <div>
          <h3>Résultat: {result.prediction}</h3>
          <p>Confiance: {result.confidence_percentage}</p>
          <p>{result.recommendation}</p>
        </div>
      )}
    </div>
  );
}
```

### Python
```python
import requests

def predict_lesion(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'https://votre-app.up.railway.app/predict',
            files=files
        )
    return response.json()

result = predict_lesion('lesion.jpg')
print(f"Prédiction: {result['prediction']}")
```

### cURL
```bash
curl -X POST "https://votre-app.up.railway.app/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@lesion.jpg"
```

## 🔒 Sécurité (Optionnel)

Pour ajouter une authentification API key :

1. Modifier `main.py` :
```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_lesion(...):
    ...
```

2. Ajouter `API_KEY` dans Railway variables d'environnement

## 📊 Monitoring

Railway fournit :
- Logs en temps réel
- Métriques CPU/RAM
- Uptime monitoring
- Alertes email

## 💰 Coûts

- **Plan gratuit Railway** : 
  - $5 de crédit gratuit/mois
  - Suffisant pour ~500 requêtes/jour
  - Pas de cold start

- **Plan payant** : $5/mois pour usage illimité

## 🐛 Troubleshooting

### L'API ne démarre pas
- Vérifiez les logs Railway
- Assurez-vous que les modèles sont bien présents
- Vérifiez que `requirements.txt` est complet

### Out of Memory
- Réduire la taille d'image acceptée
- Utiliser `torch.quantization` pour ViT
- Passer au plan Railway Pro (8GB → 32GB RAM)

### Build trop long
- Railway timeout après 15 min
- Mettre les modèles sur GitHub LFS si >100 Mo
- Ou héberger les modèles sur Hugging Face

## 🆘 Support

- Documentation Railway : https://docs.railway.app
- Issues GitHub : Créez une issue sur votre repo

## 📝 Licence

MIT

---

Fait avec ❤️ pour améliorer le diagnostic dermatologique sur peaux noires