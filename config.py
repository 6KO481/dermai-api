"""
Configuration et constantes pour le système de classification des lésions cutanées
"""
import os

# 📁 Chemins des modèles
MODEL1_PATH = "models/model1_general.h5"
MODEL2_PATH = "models/model2_malignant.h5"

# 🏷️ Configuration des classes
THREE_PARTITION_CLASSES = ['healthy', 'malignant', 'benign', 'non-neoplastic']

MALIGNANT_CLASSES = [
    'actinic_keratosis', 
    'basal_cell_carcinoma', 
    'kaposi_sarcoma',
    'melanoma', 
    'mycosis_fungoides', 
    'squamous_cell_carcinoma'
]

KERATINOCYTE_CLASSES = [
    'actinic_keratosis', 
    'basal_cell_carcinoma', 
    'squamous_cell_carcinoma'
]

# 🖼️ Configuration des images
IMG_SIZE = (224, 224)

# 🎨 Couleurs pour l'UI
COLORS = {
    'healthy': '#10b981',  # Vert
    'non_cancerous_lesion': '#3b82f6',  # Bleu
    'benign': '#3b82f6',  # Bleu
    'non-neoplastic': '#3b82f6',  # Bleu
    'malignant': '#ef4444',  # Rouge
    'melanoma': '#dc2626',  # Rouge foncé
    'kaposi_sarcoma': '#dc2626',
    'mycosis_fungoides': '#dc2626',
    'keratinocytes': '#f59e0b',  # Orange
    'actinic_keratosis': '#f59e0b',
    'basal_cell_carcinoma': '#f59e0b',
    'squamous_cell_carcinoma': '#f59e0b',
}

# 📝 Descriptions des classes
CLASSES_INFO = {
    'healthy': {
        'name': 'Peau Saine',
        'description': 'Aucune lésion ou anomalie détectée. La peau présente un aspect normal.',
        'color': COLORS['healthy'],
        'severity': 'none',
        'recommendation': 'Continuez votre routine de soins habituelle et une surveillance régulière.'
    },
    'non_cancerous_lesion': {
        'name': 'Lésion Non-Cancéreuse',
        'description': 'Lésion bénigne ou non-néoplasique détectée. Ces lésions sont généralement inoffensives mais nécessitent une surveillance.',
        'color': COLORS['non_cancerous_lesion'],
        'severity': 'low',
        'recommendation': 'Consultez un dermatologue pour confirmation et surveillance.'
    },
    'benign': {
        'name': 'Lésion Bénigne',
        'description': 'Lésion bénigne détectée. Non cancéreuse mais nécessite un suivi médical.',
        'color': COLORS['benign'],
        'severity': 'low',
        'recommendation': 'Surveillance recommandée par un professionnel de santé.'
    },
    'non-neoplastic': {
        'name': 'Lésion Non-Néoplasique',
        'description': 'Lésion inflammatoire ou autre condition non tumorale.',
        'color': COLORS['non-neoplastic'],
        'severity': 'low',
        'recommendation': 'Consultation dermatologique recommandée pour évaluation.'
    },
    'malignant': {
        'name': 'Lésion Maligne',
        'description': 'Lésion potentiellement cancéreuse détectée. Nécessite une évaluation médicale urgente.',
        'color': COLORS['malignant'],
        'severity': 'high',
        'recommendation': '⚠️ CONSULTATION URGENTE RECOMMANDÉE avec un dermatologue.'
    },
    'melanoma': {
        'name': 'Mélanome',
        'description': 'Forme la plus grave de cancer de la peau. Le mélanome se développe à partir des mélanocytes.',
        'color': COLORS['melanoma'],
        'severity': 'critical',
        'recommendation': '🚨 CONSULTATION MÉDICALE URGENTE NÉCESSAIRE. Le mélanome nécessite un traitement rapide.'
    },
    'kaposi_sarcoma': {
        'name': 'Sarcome de Kaposi',
        'description': 'Type de cancer qui forme des lésions sur la peau, les muqueuses ou les organes internes.',
        'color': COLORS['kaposi_sarcoma'],
        'severity': 'critical',
        'recommendation': '🚨 CONSULTATION MÉDICALE URGENTE NÉCESSAIRE.'
    },
    'mycosis_fungoides': {
        'name': 'Mycosis Fongoïde',
        'description': 'Type de lymphome cutané à cellules T. Forme la plus commune de lymphome cutané.',
        'color': COLORS['mycosis_fungoides'],
        'severity': 'critical',
        'recommendation': '🚨 CONSULTATION MÉDICALE URGENTE NÉCESSAIRE.'
    },
    'keratinocytes': {
        'name': 'Carcinome à Cellules Kératinocytes',
        'description': 'Cancer de la peau affectant les cellules kératinocytes. Généralement traitable s\'il est détecté tôt.',
        'color': COLORS['keratinocytes'],
        'severity': 'high',
        'recommendation': '⚠️ CONSULTATION DERMATOLOGIQUE URGENTE RECOMMANDÉE.'
    },
    'actinic_keratosis': {
        'name': 'Kératose Actinique',
        'description': 'Lésion précancéreuse causée par une exposition excessive au soleil. Peut évoluer en carcinome.',
        'color': COLORS['actinic_keratosis'],
        'severity': 'medium',
        'recommendation': 'Consultation dermatologique recommandée pour traitement préventif.'
    },
    'basal_cell_carcinoma': {
        'name': 'Carcinome Basocellulaire',
        'description': 'Forme la plus courante de cancer de la peau. Croissance lente, rarement métastatique.',
        'color': COLORS['basal_cell_carcinoma'],
        'severity': 'high',
        'recommendation': '⚠️ CONSULTATION DERMATOLOGIQUE NÉCESSAIRE pour traitement.'
    },
    'squamous_cell_carcinoma': {
        'name': 'Carcinome Spinocellulaire',
        'description': 'Deuxième cancer de la peau le plus fréquent. Peut métastaser s\'il n\'est pas traité.',
        'color': COLORS['squamous_cell_carcinoma'],
        'severity': 'high',
        'recommendation': '⚠️ CONSULTATION DERMATOLOGIQUE URGENTE NÉCESSAIRE.'
    }
}

def get_class_color(class_name):
    """Retourne la couleur associée à une classe"""
    return COLORS.get(class_name, '#6b7280')

def get_class_description(class_name):
    """Retourne la description d'une classe"""
    info = CLASSES_INFO.get(class_name, {})
    return info.get('description', 'Description non disponible.')

def get_class_recommendation(class_name):
    """Retourne la recommandation pour une classe"""
    info = CLASSES_INFO.get(class_name, {})
    return info.get('recommendation', 'Consultez un professionnel de santé.')

def format_confidence_bar(confidence, color):
    """Génère une barre de progression HTML pour la confiance"""
    return f"""
    <div style='background: #e2e8f0; border-radius: 12px; height: 24px; overflow: hidden;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);'>
        <div style='background: linear-gradient(90deg, {color} 0%, {color}dd 100%);
                    height: 100%; width: {confidence*100}%; 
                    transition: width 1s ease;
                    display: flex; align-items: center; justify-content: flex-end;
                    padding-right: 8px; color: white; font-weight: 600; font-size: 0.85rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);'>
        </div>
    </div>
    """

# 🌍 Configuration de l'environnement
def setup_environment():
    """Configure l'environnement d'exécution"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Réduit les logs TensorFlow
    
    # Créer les dossiers nécessaires s'ils n'existent pas
    os.makedirs('models', exist_ok=True)
    os.makedirs('examples', exist_ok=True)
    
    print("✅ Environnement configuré")

# 📊 Métadonnées de l'application
APP_METADATA = {
    'name': 'DermAI - Skin Lesion Classifier',
    'version': '1.0.0',
    'description': 'Système d\'IA pour la classification des lésions cutanées sur peau noire',
    'author': 'DermAI Team',
    'license': 'MIT',
    'models': {
        'model1': 'Classification générale (4 classes)',
        'model2': 'Classification maligne détaillée (6 classes)'
    },
    'features': [
        'Vision Transformer (ViT) pour l\'extraction de features',
        'Architecture en cascade à deux étapes',
        'Interface utilisateur moderne et intuitive',
        'Spécialement optimisé pour les peaux noires et métissées'
    ]
}

def print_app_info():
    """Affiche les informations de l'application"""
    print("\n" + "="*60)
    print(f"🏥 {APP_METADATA['name']}")
    print(f"📌 Version: {APP_METADATA['version']}")
    print(f"📝 {APP_METADATA['description']}")
    print("="*60)
    print("\n🔧 Fonctionnalités:")
    for feature in APP_METADATA['features']:
        print(f"  • {feature}")
    print("\n" + "="*60 + "\n")