"""
Script para verificar overfitting nos modelos de spam
Executa análise completa comparando treino vs teste
"""
import os
import email
import email.policy
import numpy as np
import pandas as pd
import re
from html import unescape
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def check_train_vs_test_performance(model, X_train, y_train, X_test, y_test):
    """
    Compara performance no treino vs teste para detectar overfitting
    """
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Predições
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Acurácias
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Diferença
    overfitting_gap = train_acc - test_acc
    
    print(f"Acurácia Treino: {train_acc:.4f}")
    print(f"Acurácia Teste:  {test_acc:.4f}")
    print(f"Gap (overfitting): {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.05:
        print("⚠️ ALERTA: Possível overfitting! Gap > 5%")
    elif overfitting_gap > 0.02:
        print("⚡ ATENÇÃO: Leve overfitting. Gap > 2%")
    else:
        print("✅ OK: Sem sinais significativos de overfitting")
    
    return train_acc, test_acc, overfitting_gap

def plot_learning_curves(model, X, y, cv=5):
    """
    Plota learning curves para visualizar overfitting
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    # Médias e desvios
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Score de Treino')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Score de Validação')
    
    # Áreas de incerteza
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Tamanho do conjunto de treino')
    plt.ylabel('Acurácia')
    plt.title('Learning Curves - Detecção de Overfitting')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Análise do gap final
    final_gap = train_mean[-1] - val_mean[-1]
    plt.text(train_sizes[-1], val_mean[-1]-0.02, 
             f'Gap final: {final_gap:.3f}', 
             fontsize=10, color='darkred')
    
    plt.tight_layout()
    plt.show()
    
    return train_mean, val_mean, final_gap

def analyze_model_complexity(model, X_train, y_train, X_test, y_test, param_name, param_range):
    """
    Analisa como a complexidade do modelo afeta overfitting
    """
    train_scores = []
    test_scores = []
    
    for param_value in param_range:
        # Configurar parâmetro
        model.set_params(**{param_name: param_value})
        
        # Treinar
        model.fit(X_train, y_train)
        
        # Avaliar
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, 'o-', label='Treino', color='blue')
    plt.plot(param_range, test_scores, 'o-', label='Teste', color='red')
    plt.xlabel(param_name)
    plt.ylabel('Acurácia')
    plt.title(f'Validation Curve - {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Marcar ponto ótimo
    best_idx = np.argmax(test_scores)
    plt.axvline(x=param_range[best_idx], color='green', linestyle='--', alpha=0.5)
    plt.text(param_range[best_idx], min(test_scores), 
             f'Melhor: {param_range[best_idx]}', 
             rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return train_scores, test_scores

def cross_dataset_validation(model, X_train, y_train, X_external, y_external):
    """
    Valida modelo em dataset completamente diferente
    """
    # Treinar no dataset original
    model.fit(X_train, y_train)
    
    # Testar no dataset externo
    predictions = model.predict(X_external)
    accuracy = accuracy_score(y_external, predictions)
    
    print(f"Acurácia em dados externos: {accuracy:.4f}")
    
    if accuracy < 0.85:
        print("⚠️ ALERTA: Performance muito baixa em dados externos!")
        print("Provável overfitting ao dataset de treino")
    elif accuracy < 0.90:
        print("⚡ ATENÇÃO: Performance reduzida em dados externos")
    else:
        print("✅ Boa generalização para dados externos")
    
    return accuracy

# Técnicas para reduzir overfitting
def apply_regularization_techniques():
    """
    Sugestões de técnicas para reduzir overfitting
    """
    techniques = """
    🛡️ TÉCNICAS PARA REDUZIR OVERFITTING:
    
    1. REGULARIZAÇÃO:
       - Aumentar alpha no Naive Bayes
       - Aumentar C no SVM e Logistic Regression
       - Diminuir max_depth no Random Forest
    
    2. SIMPLIFICAR FEATURES:
       - Reduzir max_features no TF-IDF
       - Remover features muito específicas
       - Usar apenas unigramas (não bi/trigramas)
    
    3. MAIS DADOS:
       - Coletar mais emails reais
       - Reduzir data augmentation artificial
       - Usar validação cruzada mais rigorosa
    
    4. DROPOUT/ENSEMBLE:
       - Usar dropout em redes neurais
       - Bagging com subsampling
       - Limitar complexidade dos ensembles
    
    5. EARLY STOPPING:
       - Parar treinamento quando validação piora
       - Monitorar métricas durante treino
    """
    print(techniques)

def load_and_prepare_data():
    """
    Carrega e prepara os dados de spam/ham
    """
    print("📧 Carregando dados...")
    
    def load_emails_from_folder(folder_path):
        emails = []
        if not os.path.exists(folder_path):
            return emails
        
        files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
        for filename in files[:100]:  # Limitar para teste rápido
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, 'rb') as f:
                    msg = email.message_from_binary_file(f, policy=email.policy.default)
                    if msg.is_multipart():
                        body = ""
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body += part.get_content()
                                except:
                                    body += str(part.get_payload())
                    else:
                        try:
                            body = msg.get_content()
                        except:
                            body = str(msg.get_payload())
                    
                    # Limpeza básica
                    body = re.sub(r'<[^>]+>', ' ', body)
                    body = re.sub(r'\s+', ' ', body)
                    emails.append(body[:5000])  # Limitar tamanho
            except:
                continue
        return emails
    
    # Carregar dados
    data_path = "spam_model_data"
    ham_emails = []
    spam_emails = []
    
    ham_emails.extend(load_emails_from_folder(os.path.join(data_path, "easy_ham")))
    ham_emails.extend(load_emails_from_folder(os.path.join(data_path, "hard_ham")))
    spam_emails.extend(load_emails_from_folder(os.path.join(data_path, "spam")))
    spam_emails.extend(load_emails_from_folder(os.path.join(data_path, "spam_2")))
    
    X = ham_emails + spam_emails
    y = ['ham'] * len(ham_emails) + ['spam'] * len(spam_emails)
    
    print(f"✅ Dados carregados: {len(X)} emails ({len(ham_emails)} ham, {len(spam_emails)} spam)")
    return X, y

def test_overfitting_all_models(X, y):
    """
    Testa overfitting em múltiplos modelos
    """
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Modelos para testar
    models = {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', MultinomialNB(alpha=0.1))
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', LogisticRegression(max_iter=100, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
        ]),
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', SVC(kernel='linear', random_state=42))
        ])
    }
    
    print("\n" + "="*70)
    print("🔍 ANÁLISE TRAIN vs TEST - DETECÇÃO DE OVERFITTING")
    print("="*70)
    
    overfitting_analysis = {}
    
    for name, model in models.items():
        print(f"\n📊 {name}:")
        print("-"*50)
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Predições no TREINO
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, pos_label='spam')
        
        # Predições no TESTE
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, pos_label='spam')
        
        # Calcular gap
        acc_gap = train_acc - test_acc
        f1_gap = train_f1 - test_f1
        
        # Mostrar resultados
        print(f"  TREINO - Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  TESTE  - Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
        print(f"  GAP    - Accuracy: {acc_gap:.4f} | F1: {f1_gap:.4f}")
        
        # Diagnóstico
        if acc_gap > 0.10:
            status = "🔴 OVERFITTING SEVERO!"
            diagnosis = "Modelo memorizou o treino"
        elif acc_gap > 0.05:
            status = "🟡 OVERFITTING MODERADO"
            diagnosis = "Precisa mais regularização"
        elif acc_gap > 0.02:
            status = "🟠 OVERFITTING LEVE"
            diagnosis = "Aceitável mas pode melhorar"
        else:
            status = "🟢 SEM OVERFITTING"
            diagnosis = "Boa generalização"
        
        print(f"  STATUS: {status}")
        print(f"  → {diagnosis}")
        
        # Guardar resultados
        overfitting_analysis[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': acc_gap,
            'status': status
        }
    
    return overfitting_analysis

def plot_overfitting_comparison(analysis):
    """
    Visualiza comparação treino vs teste
    """
    models = list(analysis.keys())
    train_scores = [analysis[m]['train_acc'] for m in models]
    test_scores = [analysis[m]['test_acc'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_scores, width, label='Treino', color='lightblue')
    bars2 = ax.bar(x + width/2, test_scores, width, label='Teste', color='lightcoral')
    
    ax.set_xlabel('Modelos')
    ax.set_ylabel('Acurácia')
    ax.set_title('Comparação Treino vs Teste - Análise de Overfitting')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Linha de referência 95%
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("🔍 ANÁLISE COMPLETA DE OVERFITTING - SPAM CLASSIFIER")
    print("="*70)
    
    # Carregar dados
    X, y = load_and_prepare_data()
    
    # Testar overfitting
    analysis = test_overfitting_all_models(X, y)
    
    # Resumo final
    print("\n" + "="*70)
    print("📊 RESUMO FINAL")
    print("="*70)
    
    # Contar status
    severe = sum(1 for a in analysis.values() if 'SEVERO' in a['status'])
    moderate = sum(1 for a in analysis.values() if 'MODERADO' in a['status'])
    light = sum(1 for a in analysis.values() if 'LEVE' in a['status'])
    none = sum(1 for a in analysis.values() if 'SEM' in a['status'])
    
    print(f"\n🔴 Overfitting Severo: {severe} modelos")
    print(f"🟡 Overfitting Moderado: {moderate} modelos")
    print(f"🟠 Overfitting Leve: {light} modelos")
    print(f"🟢 Sem Overfitting: {none} modelos")
    
    # Modelo com maior gap
    worst_model = max(analysis.items(), key=lambda x: x[1]['gap'])
    print(f"\n⚠️ Pior caso: {worst_model[0]} com gap de {worst_model[1]['gap']:.4f}")
    
    # Recomendações
    print("\n" + "="*70)
    apply_regularization_techniques()