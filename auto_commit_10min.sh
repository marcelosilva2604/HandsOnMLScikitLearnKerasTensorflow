#!/bin/bash

# Script para fazer commit e push automaticamente a cada 10 minutos
# Para parar o script, use: kill $(cat /tmp/auto_commit_10min.pid)

# Salvar PID do processo
echo $$ > /tmp/auto_commit_10min.pid

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🚀 Script de Auto-Commit iniciado!${NC}"
echo -e "${YELLOW}PID: $$${NC}"
echo -e "${YELLOW}Intervalo: 10 minutos${NC}"
echo -e "${RED}Para parar, execute: kill $(cat /tmp/auto_commit_10min.pid)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

counter=1

while true; do
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║ Execução #$counter - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    
    # Verificar status
    git_status=$(git status --porcelain)
    
    if [ -z "$git_status" ]; then
        echo -e "${YELLOW}📝 Nenhuma alteração detectada. Criando commit de sincronização...${NC}"
        
        # Criar commit vazio
        commit_msg="Auto-sync #$counter - $(date '+%Y-%m-%d %H:%M:%S')

Sincronização automática a cada 10 minutos

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
        
        git commit --allow-empty -m "$commit_msg"
        
    else
        echo -e "${GREEN}📦 Alterações detectadas:${NC}"
        echo "$git_status" | head -10
        
        # Contar arquivos modificados
        modified_count=$(echo "$git_status" | wc -l)
        echo -e "${YELLOW}Total de arquivos com mudanças: $modified_count${NC}"
        
        # Adicionar todas as alterações
        echo -e "${YELLOW}➕ Adicionando arquivos...${NC}"
        git add .
        
        # Criar commit com as mudanças
        commit_msg="Auto-commit #$counter - $(date '+%Y-%m-%d %H:%M:%S')

$modified_count arquivo(s) modificado(s) automaticamente

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
        
        git commit -m "$commit_msg"
    fi
    
    # Fazer push
    echo -e "${YELLOW}📤 Enviando para o GitHub...${NC}"
    
    if git push origin main; then
        echo -e "${GREEN}✅ Push realizado com sucesso!${NC}"
        
        # Mostrar resumo
        echo ""
        echo -e "${GREEN}📊 Resumo da Execução #$counter:${NC}"
        echo -e "   ⏰ Horário: $(date '+%H:%M:%S')"
        echo -e "   📝 Tipo: $([ -z "$git_status" ] && echo "Commit vazio (sincronização)" || echo "Commit com alterações")"
        echo -e "   ✅ Status: Sucesso"
        
    else
        echo -e "${RED}❌ Erro ao fazer push. Tentando novamente no próximo ciclo...${NC}"
    fi
    
    counter=$((counter + 1))
    
    echo ""
    echo -e "${YELLOW}⏳ Aguardando 10 minutos até a próxima execução...${NC}"
    echo -e "${YELLOW}💡 Dica: Para verificar o status, veja o arquivo: /tmp/auto_commit_10min.log${NC}"
    echo -e "${RED}🛑 Para parar: kill $(cat /tmp/auto_commit_10min.pid)${NC}"
    
    # Salvar log
    echo "Execução #$((counter-1)) concluída em $(date)" >> /tmp/auto_commit_10min.log
    
    # Aguardar 10 minutos (600 segundos)
    sleep 600
done