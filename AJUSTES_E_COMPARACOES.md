# Ajustes no código e comparação dos treinos (11 e 12/11/2026)

Este arquivo registra o que foi alterado em `trabalho_pcr_dce_mri.py` com apoio do Claude Code (ferramenta de IA) e o que os treinos feitos depois mostraram. Serve de base para a seção de ajuste de hiperparâmetros do relatório e para a declaração de uso de IA.

## Resumo

- O recorte da imagem em torno do tumor estabilizou a rede nova (AUC de teste de 0,513 para 0,698 na média das sementes, com 3 fases). Na ResNet50 o ganho foi pequeno e na CNN não houve ganho.
- Sortear o mesmo número de cortes por paciente e congelar mais camadas da rede nova não trouxeram ganho.
- A média das previsões de 9 modelos (comitê) deu a melhor AUC de teste, 0,746, mas com sensibilidade baixa (0,25), por causa do limiar.
- A hipótese das 3 fases continua sem confirmação: vence em 1 de 3 modelos na rodada com recorte.
- Nenhuma diferença aqui é estatisticamente segura. O teste tem 35 pacientes (12 com pCR), o que dá um erro-padrão de AUC de cerca de ±0,10, e a AUC de validação (33 pacientes) não acompanha a de teste.

## Como ler os números

- **AUC de teste e de validação:** calculadas por paciente, com a agregação "média dos cortes", salvo indicação em contrário.
- **Partição:** 104 pacientes de treino (26 com pCR), 33 de validação (11 com pCR) e 35 de teste (12 com pCR), conforme a coluna `test` do CSV.
- **Hiperparâmetros comuns a todas as rodadas comparadas:** learning rate 1e-4, weight decay 0,01, margem de cortes 0, 12 épocas no máximo, 2 variações de aumento de dados por corte, lote 32, perda com `pos_weight`, melhor época escolhida pela AUC de validação. Semente 42, exceto onde indicado.
- **Tags das rodadas** (sufixo no nome dos arquivos em `saida_trabalho/`):

| Tag | O que é |
|---|---|
| `lr1e-4_m0` | referência sem recorte, treinada pelo grupo em 08/11 |
| `lr1e-4_m0_seed7` | referência sem recorte, semente 7, só 3 fases |
| `roi` | com recorte do tumor, 12 treinos |
| `roi_s7`, `roi_s13` | com recorte, sementes 7 e 13, só 3 fases |
| `roi_bal` | com recorte e mesmo número de cortes por paciente, só 3 fases |
| `roi_frz3` | com recorte e rede nova congelada até o layer3 |

## O que foi alterado no código

### Correções

| Problema | Correção |
|---|---|
| `--avaliar` e a interface avaliavam modelos treinados com margem 0 usando margem 1 | A margem é lida do checkpoint; a interface ganhou o campo "Margem (cortes)" e recusa avaliar com margem diferente da do modelo |
| O cache de pré-processamento era reaproveitado mesmo se o CSV ganhasse pacientes | O cache guarda a lista de pacientes do CSV e só é reaproveitado se ela for idêntica |
| Máscara com formato diferente da imagem virava zeros sem aviso | Passa a registrar um aviso no log |
| O scheduler reduzia o learning rate pela loss de validação, que sobe desde a 1ª época | Com critério `auc`, o scheduler acompanha a AUC de validação |

### Opções novas (todas desligadas por padrão)

| Opção | Terminal | Interface | O que faz |
|---|---|---|---|
| Recorte do tumor | `--roi` | caixa "Recortar tumor" na barra superior | Recorta um quadrado em torno da caixa do tumor (folga de 25% de cada lado, lado mínimo de 48 px) antes de reduzir para 128 px. O mesmo quadrado vale para todos os cortes do paciente |
| Balanceamento por paciente | `--pat-balance` | caixa na aba Treinamento | Em cada época sorteia o mesmo número de cortes por paciente (a mediana), e o peso da classe pCR passa a ser contado por paciente |
| Congelamento da rede nova | `--novel-freeze 1\|2\|3` | seletor na aba Treinamento | Congela a ResNet18 até o layer indicado (1 é o comportamento anterior) |
| Comitê de modelos | `--avaliar a.pt,b.pt,c.pt` | não disponível | Faz a média das probabilidades dos modelos; o limiar é a média dos limiares |

A avaliação confere se o modelo e o dataset carregado usam o mesmo tamanho de imagem, a mesma margem e o mesmo recorte.

### Ambiente

- `mise.toml`: Python 3.12 e uv, ambiente virtual em `.venv`, PyTorch na versão CPU. Tarefas `install`, `selftest`, `gui` e `train`.
- `.gitignore`: `.venv/`, `__pycache__/` e `saida_trabalho/cache/`.

### Motivo do recorte

Nos 172 pacientes, a caixa do tumor ocupa em mediana 8,7% da área da imagem de 256×256. Sem recorte, a rede recebe mais de 90% de tecido de fundo depois da redução para 128 px.

## Comparações

### 1. Recorte do tumor: os 12 treinos

| Modelo | Entrada | AUC val sem recorte | AUC val com recorte | AUC teste sem recorte | AUC teste com recorte | Diferença no teste |
|---|---|---|---|---|---|---|
| CNN | pré | 0,488 | 0,508 | 0,594 | 0,649 | +0,055 |
| CNN | precoce | 0,566 | 0,558 | 0,699 | 0,630 | −0,069 |
| CNN | tardio | 0,533 | 0,607 | 0,659 | 0,591 | −0,068 |
| CNN | 3 fases | 0,645 | 0,483 | 0,692 | 0,645 | −0,047 |
| ResNet50 | pré | 0,492 | 0,483 | 0,616 | 0,707 | +0,091 |
| ResNet50 | precoce | 0,529 | 0,500 | 0,620 | 0,699 | +0,079 |
| ResNet50 | tardio | 0,500 | 0,488 | 0,630 | 0,725 | +0,095 |
| ResNet50 | 3 fases | 0,504 | 0,508 | 0,692 | 0,696 | +0,004 |
| Rede nova | pré | 0,678 | 0,591 | 0,554 | 0,638 | +0,084 |
| Rede nova | precoce | 0,694 | 0,566 | 0,580 | 0,685 | +0,105 |
| Rede nova | tardio | 0,653 | 0,587 | 0,540 | 0,692 | +0,152 |
| Rede nova | 3 fases | 0,628 | 0,554 | 0,417 | 0,732 | +0,315 |
| **Média** | | **0,576** | **0,536** | **0,608** | **0,674** | **+0,066** |

- O teste melhorou em 9 dos 12 pares; a validação melhorou em 3 dos 12.
- Diferença média no teste por modelo: CNN −0,033, ResNet50 +0,067, rede nova +0,164.

### 2. Três fases com várias sementes (AUC de teste)

| Modelo | Sem recorte: sementes 42 e 7 | Média | Com recorte: sementes 42, 7 e 13 | Média | Desvio-padrão |
|---|---|---|---|---|---|
| CNN | 0,692 e 0,681 | 0,687 | 0,645, 0,754 e 0,656 | 0,685 | 0,060 |
| ResNet50 | 0,692 e 0,707 | 0,700 | 0,696, 0,728 e 0,699 | 0,708 | 0,018 |
| Rede nova | 0,417 e 0,609 | 0,513 | 0,732, 0,652 e 0,710 | 0,698 | 0,041 |

Com mais de uma semente, o ganho do recorte se mantém só na rede nova. Com recorte, os três modelos ficam entre 0,69 e 0,71.

### 3. Variantes sobre o recorte

| Rodada | Modelo | Entrada | AUC val | AUC teste | AUC teste só com recorte |
|---|---|---|---|---|---|
| `roi_bal` | CNN | 3 fases | 0,434 | 0,649 | 0,645 |
| `roi_bal` | ResNet50 | 3 fases | 0,479 | 0,623 | 0,696 |
| `roi_bal` | Rede nova | 3 fases | 0,595 | 0,685 | 0,732 |
| `roi_frz3` | Rede nova | pré | 0,566 | 0,627 | 0,638 |
| `roi_frz3` | Rede nova | precoce | 0,570 | 0,645 | 0,685 |
| `roi_frz3` | Rede nova | tardio | 0,607 | 0,663 | 0,692 |
| `roi_frz3` | Rede nova | 3 fases | 0,603 | 0,703 | 0,732 |

- Balanceamento por paciente: diferença média no teste de −0,039 (melhorou em 1 de 3).
- Congelamento até o layer3: diferença média no teste de −0,027 (melhorou em 0 de 4); na validação, +0,012.

### 4. Comitê: média das previsões no teste

Calculado a partir das previsões de teste salvas nos JSONs das rodadas `roi`, `roi_s7` e `roi_s13` (3 fases), sem executar os modelos de novo.

| Comitê | AUC | Acurácia | Sensibilidade | Especificidade | F1 |
|---|---|---|---|---|---|
| CNN, 3 sementes | 0,696 | 0,686 | 0,58 | 0,74 | 0,56 |
| ResNet50, 3 sementes | 0,699 | 0,657 | 0,33 | 0,83 | 0,40 |
| Rede nova, 3 sementes | 0,707 | 0,743 | 0,33 | 0,96 | 0,47 |
| ResNet50 + rede nova, 6 modelos | 0,725 | 0,743 | 0,33 | 0,96 | 0,47 |
| Os 9 modelos | 0,746 | 0,686 | 0,25 | 0,91 | 0,35 |

O comitê dos 9 acerta 3 das 12 pacientes com pCR e 21 das 23 sem pCR. A sensibilidade baixa vem do limiar, que é a média dos limiares individuais e fica alto para as probabilidades médias. Com a agregação "média ponderada pela área", o mesmo comitê tem AUC 0,707, sensibilidade 0,50 e especificidade 0,78.

### 5. Hipótese das 3 fases, na rodada com recorte (AUC de teste)

| Modelo | Pré | Precoce | Tardio | 3 fases | 3 fases vence a melhor fase única? |
|---|---|---|---|---|---|
| CNN | 0,649 | 0,630 | 0,591 | 0,645 | não |
| ResNet50 | 0,707 | 0,699 | 0,725 | 0,696 | não |
| Rede nova | 0,638 | 0,685 | 0,692 | 0,732 | sim |

Somando com as rodadas sem recorte (`base` e `lr1e-4_m0`), as 3 fases vencem em 3 de 9 comparações.

### 6. Estratégias de agregação (AUC de teste média dos 12 treinos)

| Agregação | Sem recorte | Com recorte |
|---|---|---|
| Corte central | 0,571 | 0,646 |
| Corte de maior área | 0,618 | 0,628 |
| Média dos cortes | 0,608 | 0,674 |
| Máximo dos cortes | 0,524 | 0,620 |
| Média dos 3 maiores | 0,540 | 0,620 |
| Média ponderada pela área | 0,625 | 0,647 |

### 7. Tempo de treino com recorte (CPU, Intel i5-13420H, sem GPU)

| Modelo | Fase única | 3 fases |
|---|---|---|
| CNN | 10 a 15 min | 14 min |
| ResNet50 | 17 a 26 min | 20 min |
| Rede nova | 12 a 17 min | 45 min |

## Limitações desta comparação

- **Amostra pequena:** com 35 pacientes de teste, diferenças de AUC abaixo de ~0,10 cabem no ruído.
- **Validação e teste discordam:** o recorte melhora o teste e piora a validação na média. Não há como dizer qual dos dois está certo com este tamanho de amostra.
- **A comparação com e sem recorte não é perfeitamente controlada:** a rodada `lr1e-4_m0` foi treinada antes da mudança do scheduler (seguia a loss) e em outra máquina; as rodadas `roi` usam o scheduler pela AUC.
- **Escolha olhando o teste:** as rodadas foram comparadas pela AUC de teste. Se o relatório apresentar a melhor delas como resultado final, o número fica otimista.
- **O recorte depende da máscara:** se o conjunto-teste surpresa vier sem máscara, o programa usa a imagem inteira, e um modelo treinado com recorte fica fora da sua condição de treino.

## O que não foi feito ou não foi testado

- O comitê não está na interface gráfica, só no terminal.
- O limiar do comitê não é escolhido na validação; é a média dos limiares individuais.
- Os botões "Classificar" e "Sobreposição Grad-CAM" não conferem se o dataset carregado usa o mesmo recorte e a mesma margem do modelo; só "Avaliar no conjunto de teste" confere.
- As variantes `roi_bal` e `roi_frz3` têm uma semente só.
- O cabeçalho do código ainda tem campos `<PREENCHER>` e o README não descreve as opções novas.
