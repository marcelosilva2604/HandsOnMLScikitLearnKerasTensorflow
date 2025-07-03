"""
Funções utilitárias para o projeto Hands-on Machine Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_plotting(style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
    """
    Configura o estilo de plots para o projeto
    
    Args:
        style: Estilo do matplotlib
        figsize: Tamanho padrão das figuras
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    sns.set_palette("husl")
    logger.info(f"Plotting configurado com estilo: {style}")

def load_and_explore_data(filepath: str) -> pd.DataFrame:
    """
    Carrega e faz uma exploração básica dos dados
    
    Args:
        filepath: Caminho para o arquivo de dados
        
    Returns:
        DataFrame com os dados carregados
    """
    try:
        # Detectar tipo de arquivo e carregar
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {filepath}")
            
        logger.info(f"Dados carregados com sucesso: {df.shape}")
        
        # Exploração básica
        print("=" * 60)
        print(f"📊 EXPLORAÇÃO DOS DADOS: {Path(filepath).name}")
        print("=" * 60)
        print(f"Forma dos dados: {df.shape}")
        print(f"Colunas: {list(df.columns)}")
        print(f"Tipos de dados:\n{df.dtypes}")
        print(f"Valores nulos:\n{df.isnull().sum()}")
        print(f"Estatísticas descritivas:\n{df.describe()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         classes: Optional[List[str]] = None,
                         title: str = "Matriz de Confusão",
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plota matriz de confusão
    
    Args:
        y_true: Valores verdadeiros
        y_pred: Predições
        classes: Nomes das classes
        title: Título do plot
        figsize: Tamanho da figura
    """
    plt.figure(figsize=figsize)
    cm = confusion_matrix(y_true, y_pred)
    
    if classes is None:
        classes = [f'Classe {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predições')
    plt.ylabel('Valores Reais')
    plt.tight_layout()
    plt.show()

def plot_learning_curves(estimator, X: np.ndarray, y: np.ndarray,
                        title: str = "Curvas de Aprendizado",
                        cv: int = 5, scoring: str = 'accuracy',
                        figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plota curvas de aprendizado
    
    Args:
        estimator: Modelo a ser avaliado
        X: Features
        y: Target
        title: Título do plot
        cv: Número de folds para cross-validation
        scoring: Métrica de avaliação
        figsize: Tamanho da figura
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', label='Treinamento', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', label='Validação', color='red')
    plt.fill_between(train_sizes, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Tamanho do Conjunto de Treinamento')
    plt.ylabel(f'Score ({scoring})')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_model(model, filepath: str, metadata: Optional[Dict] = None) -> None:
    """
    Salva modelo com metadados
    
    Args:
        model: Modelo treinado
        filepath: Caminho para salvar
        metadata: Metadados adicionais
    """
    model_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_data, filepath)
    logger.info(f"Modelo salvo em: {filepath}")

def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Carrega modelo com metadados
    
    Args:
        filepath: Caminho do modelo
        
    Returns:
        Tupla com (modelo, metadados)
    """
    model_data = joblib.load(filepath)
    logger.info(f"Modelo carregado de: {filepath}")
    return model_data['model'], model_data.get('metadata', {})

def evaluate_classification_model(model, X_test: np.ndarray, y_test: np.ndarray,
                                class_names: Optional[List[str]] = None) -> Dict:
    """
    Avalia modelo de classificação
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Targets de teste
        class_names: Nomes das classes
        
    Returns:
        Dicionário com métricas
    """
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    results = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print("🎯 AVALIAÇÃO DO MODELO")
    print("=" * 50)
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(results['classification_report'])
    
    # Plot matriz de confusão
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    return results

def create_feature_importance_plot(model, feature_names: List[str],
                                 top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Cria gráfico de importância das features
    
    Args:
        model: Modelo com feature_importances_
        feature_names: Nomes das features
        top_n: Número de features mais importantes para mostrar
        figsize: Tamanho da figura
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Modelo não possui feature_importances_")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.title(f'Top {top_n} Features Mais Importantes')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importância')
    plt.tight_layout()
    plt.show()

def print_system_info():
    """
    Imprime informações do sistema e bibliotecas
    """
    import sys
    import sklearn
    import numpy as np
    import pandas as pd
    
    print("🔧 INFORMAÇÕES DO SISTEMA")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
    except ImportError:
        print("TensorFlow: Não instalado")
    
    print("=" * 50) 