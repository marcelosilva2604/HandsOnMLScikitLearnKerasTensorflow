"""
Script para verificar overfitting no spam1.ipynb
Reproduz exatamente o que foi feito no spam1 e calcula o gap treino vs teste
"""
import os
import email
import email.policy
import numpy as np
import re
from html import unescape
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_emails_from_folder(folder_path):
    """Carrega emails de uma pasta específica"""
    emails = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Pasta não encontrada: {folder_path}")
        return emails
    
    files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
    print(f"📁 Processando {len(files)} arquivos em {os.path.basename(folder_path)}")
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        try:
            with open(file_path, 'rb') as f:
                msg = email.message_from_binary_file(f, policy=email.policy.default)
                
                # Extrair conteúdo do email
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
                
                emails.append(str(body))
                
        except Exception as e:
            continue  # Pular emails com erro
    
    print(f"✅ {len(emails)} emails carregados com sucesso!")
    return emails

def clean_email_text(text):
    """
    Limpa o texto do email - EXATAMENTE como no spam1.ipynb
    """
    if not text or len(text.strip()) == 0:
        return "empty email"
    
    # Remover HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Decodificar HTML entities
    text = unescape(text)
    
    # Remover URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    # Remover emails
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    
    # Remover números longos
    text = re.sub(r'\b\d{3,}\b', 'NUMBER', text)
    
    # Remover caracteres especiais excessivos
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remover espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def main():
    print("=" * 70)
    print("🔍 ANÁLISE DE OVERFITTING - SPAM1.IPYNB")
    print("Reproduzindo exatamente o modelo do spam1 para verificar overfitting")
    print("=" * 70)
    
    # 1. CARREGAR DADOS (igual ao spam1)
    print("\n🔄 Carregando emails...")
    data_path = "spam_model_data"
    
    ham_emails = []
    spam_emails = []
    
    # HAM
    ham_emails.extend(load_emails_from_folder(os.path.join(data_path, "easy_ham")))
    ham_emails.extend(load_emails_from_folder(os.path.join(data_path, "hard_ham")))
    
    # SPAM
    spam_emails.extend(load_emails_from_folder(os.path.join(data_path, "spam")))
    spam_emails.extend(load_emails_from_folder(os.path.join(data_path, "spam_2")))
    
    print(f"\n📊 RESUMO:")
    print(f"HAM emails: {len(ham_emails)}")
    print(f"SPAM emails: {len(spam_emails)}")
    print(f"Total: {len(ham_emails) + len(spam_emails)}")
    
    # 2. PREPARAR DADOS (igual ao spam1)
    X = ham_emails + spam_emails
    y = ['ham'] * len(ham_emails) + ['spam'] * len(spam_emails)
    
    print(f"\n📊 Dataset preparado:")
    print(f"Total de emails: {len(X)}")
    print(f"HAM: {y.count('ham')}")
    print(f"SPAM: {y.count('spam')}")
    
    # 3. DIVIDIR EM TREINO E TESTE (igual ao spam1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🔄 Divisão treino/teste:")
    print(f"Treino: {len(X_train)} emails")
    print(f"Teste: {len(X_test)} emails")
    
    # 4. APLICAR LIMPEZA NOS DADOS (igual ao spam1)
    print("\n🧹 Limpando dados...")
    X_train_clean = [clean_email_text(email) for email in X_train]
    X_test_clean = [clean_email_text(email) for email in X_test]
    
    # 5. CRIAR MODELO EXATAMENTE IGUAL AO SPAM1
    print("\n🔄 Criando modelo TF-IDF + dados limpos (igual spam1)...")
    
    spam_classifier_clean = Pipeline([
        ('vectorizer', TfidfVectorizer(
            stop_words='english', 
            lowercase=True, 
            max_features=10000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1,2)
        )),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    # 6. TREINAR MODELO
    print("🔄 Treinando modelo...")
    spam_classifier_clean.fit(X_train_clean, y_train)
    
    # 7. FAZER PREDIÇÕES NO TREINO E TESTE
    print("📊 Fazendo predições no treino e teste...")
    
    # Predições no TREINO
    y_train_pred = spam_classifier_clean.predict(X_train_clean)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, pos_label='spam')
    
    # Predições no TESTE
    y_test_pred = spam_classifier_clean.predict(X_test_clean)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, pos_label='spam')
    
    # 8. CALCULAR GAP (O QUE NÃO FOI FEITO NO SPAM1!)
    accuracy_gap = train_accuracy - test_accuracy
    f1_gap = train_f1 - test_f1
    
    # 9. RESULTADOS
    print("\n" + "=" * 70)
    print("📊 RESULTADOS - SPAM1.IPYNB COM ANÁLISE DE OVERFITTING")
    print("=" * 70)
    
    print(f"\nTREINO:")
    print(f"  Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  F1-Score: {train_f1:.4f}")
    
    print(f"\nTESTE:")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  F1-Score: {test_f1:.4f}")
    
    print(f"\nGAP (Treino - Teste):")
    print(f"  Accuracy Gap: {accuracy_gap:.4f} ({accuracy_gap*100:.2f} pontos percentuais)")
    print(f"  F1 Gap: {f1_gap:.4f}")
    
    # 10. DIAGNÓSTICO DE OVERFITTING
    print(f"\n🔍 DIAGNÓSTICO DE OVERFITTING:")
    print("-" * 50)
    
    if accuracy_gap > 0.10:
        status = "🔴 OVERFITTING SEVERO"
        diagnosis = "Modelo memorizou o conjunto de treino!"
        recommendation = "Precisa de regularização forte"
    elif accuracy_gap > 0.05:
        status = "🟡 OVERFITTING MODERADO"
        diagnosis = "Modelo tem dificuldade de generalizar"
        recommendation = "Aplicar regularização"
    elif accuracy_gap > 0.02:
        status = "🟠 OVERFITTING LEVE"
        diagnosis = "Leve preferência pelos dados de treino"
        recommendation = "Aceitável, mas pode melhorar"
    else:
        status = "🟢 SEM OVERFITTING"
        diagnosis = "Boa capacidade de generalização"
        recommendation = "Modelo está bem balanceado"
    
    print(f"STATUS: {status}")
    print(f"DIAGNÓSTICO: {diagnosis}")
    print(f"RECOMENDAÇÃO: {recommendation}")
    
    # 11. MATRIZ DE CONFUSÃO
    cm = confusion_matrix(y_test, y_test_pred, labels=['ham', 'spam'])
    
    print(f"\n📊 MATRIZ DE CONFUSÃO:")
    print(f"HAM→HAM: {cm[0][0]}, HAM→SPAM: {cm[0][1]}")
    print(f"SPAM→HAM: {cm[1][0]}, SPAM→SPAM: {cm[1][1]}")
    
    # 12. COMPARAÇÃO COM RESULTADOS ORIGINAIS
    original_reported = 0.9721  # Resultado reportado no spam1.ipynb
    
    print(f"\n📈 COMPARAÇÃO COM RESULTADO ORIGINAL:")
    print("-" * 50)
    print(f"Reportado no spam1.ipynb: {original_reported:.4f} ({original_reported*100:.2f}%)")
    print(f"Nosso resultado (teste):   {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Diferença: {abs(test_accuracy - original_reported):.4f}")
    
    if abs(test_accuracy - original_reported) < 0.01:
        print("✅ Resultados consistentes!")
    else:
        print("⚠️ Diferença significativa - pode haver variação nos dados/split")
    
    # 13. CONCLUSÃO
    print(f"\n" + "=" * 70)
    print("💡 CONCLUSÃO SOBRE O SPAM1.IPYNB:")
    print("=" * 70)
    
    if accuracy_gap <= 0.02:
        verdict = "✅ O spam1.ipynb NÃO tem problema significativo de overfitting!"
    elif accuracy_gap <= 0.05:
        verdict = "⚠️ O spam1.ipynb tem overfitting leve/moderado"
    else:
        verdict = "🚨 O spam1.ipynb tem overfitting significativo!"
    
    print(verdict)
    print(f"\nAcurácia real no teste: {test_accuracy*100:.2f}%")
    print(f"Gap de overfitting: {accuracy_gap*100:.2f} pontos percentuais")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'gap': accuracy_gap,
        'status': status,
        'model': spam_classifier_clean
    }

if __name__ == "__main__":
    result = main()