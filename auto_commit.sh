#!/bin/bash

# Script para fazer commit e push automaticamente a cada 5 minutos
# Para parar o script, use: kill $(cat /tmp/auto_commit.pid)

echo $$ > /tmp/auto_commit.pid
echo "Script iniciado! PID: $$"
echo "Para parar, execute: kill $(cat /tmp/auto_commit.pid)"

counter=1

while true; do
    echo ""
    echo "========================================="
    echo "Execução #$counter - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    
    # Verificar status
    git_status=$(git status --porcelain)
    
    if [ -z "$git_status" ]; then
        echo "Nenhuma alteração detectada. Criando commit vazio..."
        
        # Criar commit vazio
        git commit --allow-empty -m "Auto-sync #$counter - $(date '+%Y-%m-%d %H:%M:%S')

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
        
    else
        echo "Alterações detectadas. Adicionando e commitando..."
        
        # Adicionar todas as alterações
        git add .
        
        # Criar commit com as mudanças
        git commit -m "Auto-commit #$counter - $(date '+%Y-%m-%d %H:%M:%S')

Arquivos modificados automaticamente

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
    fi
    
    # Fazer push
    echo "Fazendo push para o GitHub..."
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ Push realizado com sucesso!"
    else
        echo "❌ Erro ao fazer push. Tentando novamente no próximo ciclo..."
    fi
    
    counter=$((counter + 1))
    
    echo "Aguardando 5 minutos até a próxima execução..."
    echo "Para parar o script, execute: kill $(cat /tmp/auto_commit.pid)"
    
    # Aguardar 5 minutos (300 segundos)
    sleep 300
done