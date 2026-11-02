# Classificação da resposta ao tratamento do câncer de mama (pCR) com redes neurais

**PUC Minas – ICEI – Ciência da Computação**
**Disciplina:** Processamento e Análise de Imagens – Prof. Alexei Machado
**Grupo:** Caio Gomes, Daniel Salgado, Nalberth, Vitor Alexandre*

---

## 1. Intuito do trabalho

Em pacientes com câncer de mama tratadas com quimioterapia neoadjuvante, a resposta patológica completa (pCR) indica que todas as células tumorais invasivas desapareceram antes da cirurgia. Prever a pCR antes do tratamento ajuda o médico a individualizar o tratamento e planejar a cirurgia.

O exame de DCE-MRI (ressonância magnética dinâmica com contraste) gera imagens em vários momentos. Neste trabalho usamos três:

| Fase | Significado |
|---|---|
| Pré-contraste | antes da injeção do contraste |
| Pós-contraste precoce | pico de realce |
| Pós-contraste tardio | dinâmica de "washout" |

**Objetivo:** desenvolver e avaliar um sistema com redes neurais que classifica pacientes em pCR ausente (0) ou presente (1) usando exclusivamente as imagens DCE-MRI, e investigar a hipótese:

> Usar conjuntamente as três fases temporais melhora a classificação da pCR em comparação com usar cada fase isoladamente.

**Dados:** subconjunto I-SPY1 MinCrop do dataset BreastDCEDL (Fridman et al., *Scientific Data* 13, 264, 2026): 172 pacientes, 3 fases por paciente, volumes recortados em torno do tumor, máscaras de segmentação e metadados em CSV (partição treino/validação/teste na coluna `test`: 0 = treino, 2 = validação, 1 = teste).

O que o programa faz

| Item | Descrição |
|---|---|
| a) Visualizador | Lê `.nii`/`.nii.gz`, mostra o plano xy com navegação em z e a máscara ao lado. |
| b) Aumento de dados | 5 rotações (−20, −10, 0, 10, 20°) × {original, espelhada} = 10 variações, no treino e na validação. Verifica o desbalanceamento e o trata (`pos_weight`, focal loss ou nenhum). |
| c) Cortes 3D → 2D | 6 estratégias de agregação por paciente: corte central do tumor, corte de maior área, média, máximo, média dos 3 maiores e média ponderada pela área. |
| d) Redes | CNN própria (5 conv + 2 FC), ResNet50 (ImageNet, treina só a parte FC) e uma rede nova (ResNet18 compartilhada + atenção entre fases). Cada uma com 4 entradas: pré, precoce, tardio e as 3 fases juntas (12 experimentos). |
| e) Grad-CAM | Classifica um paciente e mostra as regiões usadas pela rede, com o contorno do tumor. |
| Opcional | Máscara como canal extra (`channel`) ou aplicada à imagem (`apply`). |

---

## 2. Estrutura de pastas

O programa possui a seguinte estrutura:

```
TP1/
├── trabalho_pcr_dce_mri.py      ← programa
├── README.md
├── dataset/                     ← CSV + pastas de DCE e de máscaras
└── saida_trabalho/              ← criada automaticamente
    ├── cache/                   (banco de cortes pré-processado)
    ├── modelos/                 (pesos .pt de cada experimento)
    └── resultados/              (métricas e históricos .json)
```

O CSV é procurado automaticamente dentro de `dataset/`. Os volumes seguem o padrão `<pid>_..._acq<N>.nii.gz` e as máscaras `<pid>_..mask...nii.gz`.

---

## 3. Instalação

Requer Python 3.9+ (com Tkinter, já incluído no instalador oficial do Python para Windows).

```
pip install torch torchvision nibabel numpy pandas pillow matplotlib
```

Para verificar se há GPU NVIDIA disponível:

```
python -c "import torch; print(torch.cuda.is_available())"
```

Se der `False` e você tiver uma placa NVIDIA, instale o PyTorch com suporte a CUDA (veja pytorch.org).

Na primeira execução com pesos ImageNet, a ResNet50/ResNet18 são baixadas automaticamente, e precisa de internet para funcionar.

---

## 4. Comandos

### 4.1 Teste rápido sem uso dos dados

Gera dados sintéticos e exercita o pipeline inteiro (pré-processamento, treino, salvamento, Grad-CAM). O esperado é que termine com `SELFTEST OK` para mostrar que funcionou completamente. As métricas 1.000 são esperadas, pois os dados são sintéticos.

```
python trabalho_pcr_dce_mri.py --selftest
```

### 4.2 Interface gráfica

```
python trabalho_pcr_dce_mri.py
```

1. Selecione os caminhos do dataset e das imagens.
2. Clique em **Carregar dataset** e aguarde `Dataset pronto: N pacientes, M cortes` (na 1ª vez leva alguns minutos; depois usa o cache).

### 4.3 Treinamento pelo terminal (sem interface)

Treina os 12 experimentos (3 modelos × 4 entradas):

```
python trabalho_pcr_dce_mri.py --train-all
```

Teste curto nos dados reais (recomendado antes do treino completo):

```
python trabalho_pcr_dce_mri.py --train-all --epochs 2 --aug-train 1 --aug-val 1 --models cnn --phases all
```

Treino por etapas (exemplos):

```
python trabalho_pcr_dce_mri.py --train-all --epochs 20 --aug-train 3 --models cnn
python trabalho_pcr_dce_mri.py --train-all --epochs 15 --aug-train 3 --models resnet50,novel
```

### 4.4 Opções da linha de comando

| Opção | Descrição | Padrão |
|---|---|---|
| `--selftest` | Teste com dados sintéticos | – |
| `--train-all` | Treina sem interface | – |
| `--data-dir` | Pasta do dataset | `./dataset` |
| `--csv` | Caminho do CSV | procura em `dataset/` |
| `--img-size` | Lado da imagem em pixels | 128 |
| `--epochs` | Número de épocas | 25 |
| `--mask` | `none`, `channel` ou `apply` | `none` |
| `--aug-train` | Variações por corte no treino (1–10) | 10 |
| `--aug-val` | Variações por corte na validação (1–10) | 10 |
| `--models` | `cnn`, `resnet50`, `novel` (separados por vírgula) | todos |
| `--phases` | `pre`, `early`, `late`, `all` (separados por vírgula) | todas |

---

## 5. As abas da interface

| Aba | Para que serve |
|---|---|
| **Visualizador** | Escolha o paciente e a fase; navegue em z; veja imagem e segmentação lado a lado. "Girar 90°" e "Espelhar" ajustam a orientação. "Mostrar 10 variações" exibe o aumento de dados. |
| **Treinamento** | Define épocas, lote, learning rate, variações por corte, tratamento do desbalanceamento, máscara, modelos e entradas. "Treinar selecionados" roda as combinações em sequência, com log e curva de loss. |
| **Resultados** | Tabela de métricas no teste (acurácia, sensibilidade, especificidade, precisão, F1, AUC, tempo), por estratégia de agregação. Ao clicar numa linha: curva de loss, matriz de confusão e pacientes errados. Gráfico comparativo e resumo da hipótese (3 fases × melhor fase única). |
| **Classificar + Grad-CAM** | Escolha um modelo salvo e um paciente; vê a probabilidade de pCR, a predição contra o valor real e o Grad-CAM com o contorno do tumor. O botão de sobreposição mede quanto do Grad-CAM cai dentro do tumor. |

---

## 6. Resultados salvos

- `saida_trabalho/modelos/<modelo>_<fase>_<mascara>.pt` – pesos e limiares (para **não** precisar retreinar na apresentação).
- `saida_trabalho/resultados/<modelo>_<fase>_<mascara>.json` – métricas, histórico de loss, tempo de treino e previsões do teste.

---

## 7. Solução de problemas

| Problema | Causa / solução |
|---|---|
| `FileNotFoundError` do CSV | Coloque o CSV em `dataset/` ou use `--csv "caminho/arquivo.csv"`. |
| "sem as 3 fases, ignorado" | O programa não achou os 3 volumes do paciente. Confira os nomes dos arquivos em `dataset/`. |
| Arquivo com nome terminando em `_nii.gz` | O sufixo deve ser `.nii.gz`; renomeie. |
| Treino parece travado | Veja o log: ele mostra `lote X/Y` a cada 25 lotes. Na CPU, reduza `--aug-train`. |
| ResNet50 com ~24 milhões de parâmetros treináveis | Os pesos ImageNet não foram baixados (sem internet). O esperado é cerca de 525 mil. |
| Mudou o tamanho da imagem | Clique em "Carregar dataset" novamente (o cache depende do tamanho). |

---

## 8. Referência

[1] Fridman, N. et al. *BreastDCEDL: A standardized deep learning-ready breast DCE-MRI dataset of 2,070 patients.* Scientific Data 13(1), 264, 2026.
