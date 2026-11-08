"""
Trabalho Prático: Classificação da resposta ao tratamento do câncer de mama (pCR) usando redes neurais e DCE-MRI (BreastDCEDL / I-SPY1)

PUC Minas - ICEI - Ciência da Computação - Campus <PREENCHER>
Processamento e Análise de Imagens - Prof. Alexei Machado

GRUPO (nome - matrícula):
  1. Caio Gomes              - <PREENCHER>
  2. Daniel Salgado Magalhães - 821429
  3. Nalberth <SOBRENOME>     - <PREENCHER>
  4. Vitor Alexandre <SOBRENOME> - <PREENCHER>
===============================================================================

COMO EXECUTAR
-------------
  pip install torch torchvision nibabel numpy pandas pillow matplotlib
  python trabalho_pcr_dce_mri.py                 -> abre a interface gráfica
  python trabalho_pcr_dce_mri.py --train-all     -> treina tudo sem interface
  python trabalho_pcr_dce_mri.py --selftest      -> teste rápido com dados sintéticos
  python trabalho_pcr_dce_mri.py --avaliar saida_trabalho/modelos/cnn_all_none.pt --csv novo.csv --data-dir nova_pasta
                                                 -> avalia um modelo salvo em outro conjunto (ex.: teste surpresa)

"""

# ------------------------------------------------------------------------------
# 1. imports
# ------------------------------------------------------------------------------
import os
import re
import sys
import json
import math
import time
import glob
import queue
import random
import argparse
import threading
import traceback

import numpy as np
import pandas as pd
import nibabel as nib                      # leitura de arquivos NIfTI (função elementar de E/S)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models             # ResNet50 / ResNet18 pré-treinadas

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    GUI_OK = True
except Exception:                                   # pragma: no cover
    GUI_OK = False


# ------------------------------------------------------------------------------
# 2. CONFIGURAÇÕES GLOBAIS (todos os hiperparâmetros ficam aqui, parametrizados)
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # contém caminhos e hiperparâmetros. A interface altera os atributos.
    data_dir = os.path.join(BASE_DIR, "dataset")            # pasta com os .nii.gz e o CSV
    csv_path = ""                                           # vazio = procura o CSV automaticamente
    work_dir = os.path.join(BASE_DIR, "saida_trabalho")     # modelos, resultados, cache
    # --- pré-processamento ---
    img_size = 128        # lado (px) após redimensionar; 128 = bom custo/benefício
    slice_margin = 1      # cortes extras acima/abaixo do tumor usados no treino
    # --- treinamento ---
    epochs = 25
    batch_size = 32
    lr = 1e-3             # taxa de aprendizado da "cabeça" (camadas novas)
    backbone_lr_factor = 0.1   # backbone pré-treinado (se descongelado) usa lr*0.1
    weight_decay = 1e-4   # regularização L2 (AdamW) contra overfitting
    patience = 7          # early stopping (épocas sem melhora na validação)
    criterio = "auc"      # como escolher a melhor época: 'auc' (AUC de validação por paciente) ou 'loss'
    tag = ""              # sufixo no nome do experimento (não sobrescreve rodadas anteriores)
    aug_train = 10        # variações por corte sorteadas por época (10 = todas)
    aug_val = 10          # variações por corte na validação
    loss_mode = "pos_weight"   # 'pos_weight' | 'focal' | 'none'
    focal_gamma = 2.0
    ft_layer4 = False     # ResNet50: também ajusta o layer4 (False = só a parte FC)
    pretrained = True     # usa pesos ImageNet (precisa de internet na 1ª vez)
    seed = 42

    def to_dict(self):
        # Apenas hiperparâmetros (sem propriedades e sem caminhos da máquina).
        skip = ("data_dir", "csv_path", "work_dir")
        return {k: getattr(self, k) for k in dir(self)
                if not k.startswith("_") and k not in skip
                and not isinstance(getattr(type(self), k, None), property)
                and not callable(getattr(self, k))}

    @property
    def model_dir(self):
        return os.path.join(self.work_dir, "modelos")

    @property
    def result_dir(self):
        return os.path.join(self.work_dir, "resultados")

    @property
    def cache_dir(self):
        return os.path.join(self.work_dir, "cache")


# Fases da DCE-MRI: índice do canal dentro do tensor de 4 canais [pré, precoce, tardia, máscara]
PHASES = {"pre": [0], "early": [1], "late": [2], "all": [0, 1, 2]}
PHASE_LABEL = {"pre": "pré-contraste", "early": "pós-precoce",
               "late": "pós-tardio", "all": "3 fases"}
MODELS = {"cnn": "CNN própria (5 conv)", "resnet50": "ResNet50 (ImageNet)",
          "novel": "Nova: ResNet18 + atenção entre fases"}
MASK_MODES = ["none", "channel", "apply"]

# Aumento de dados: (ângulo em graus, espelhado?) -> 5 x 2 = 10 variações
VARIANTS = [(a, f) for f in (0, 1) for a in (-20, -10, 0, 10, 20)]

# Estratégias de agregação das previsões dos cortes (item c da especificação)
AGGS = ["central", "maxarea", "mean", "max", "top3", "area_weighted"]
AGG_LABEL = {"central": "corte central do tumor", "maxarea": "corte de maior área",
             "mean": "média dos cortes", "max": "máximo dos cortes",
             "top3": "média dos 3 maiores (outra)", "area_weighted": "média ponderada pela área (outra)"}


def get_device():
    # Usa GPU se existir; caso contrário CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    # Fixa as sementes para reprodutibilidade.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_name(spec, tag=""):
    # Nome único de um experimento: modelo_fase_mascara[_tag].
    return "%s_%s_%s" % tuple(spec) + ("_" + tag if tag else "")


# ------------------------------------------------------------------------------
# 3. LEITURA DOS DADOS (CSV, localização dos arquivos .nii, volumes)
# ------------------------------------------------------------------------------
def find_csv(cfg):
    # usa cfg.csv_path se existir; senão procura um .csv (de preferência com 'metadata') na pasta de dados.
    if cfg.csv_path and os.path.isfile(cfg.csv_path):
        return cfg.csv_path
    cands = []
    for base in (cfg.data_dir, BASE_DIR):
        for root, _, files in os.walk(base):
            if os.path.abspath(root).startswith(os.path.abspath(cfg.work_dir)):
                continue
            cands += [os.path.join(root, f) for f in files if f.lower().endswith(".csv")]
        if cands:
            break
    cands.sort(key=lambda f: ("metadata" not in os.path.basename(f).lower(), len(f)))
    return cands[0] if cands else None


def read_meta(csv_path):
    # Lê o CSV de metadados (pid, pCR, test). O pandas aceita o fim de linha '\\r' do arquivo.
    df = pd.read_csv(csv_path)
    if "test" not in df.columns:                 # ex.: CSV do conjunto-teste surpresa
        df["test"] = 1                           # sem partição -> todos os pacientes viram "teste"
    if "pCR" not in df.columns:                  # sem rótulo -> -1 (só gera previsões, sem métricas)
        df["pCR"] = -1
    df = df.dropna(subset=["pCR", "test"]).copy()
    df["pCR"] = df["pCR"].astype(int)
    df["test"] = df["test"].astype(int)          # 0 = treino, 1 = teste, 2 = validação
    return df.reset_index(drop=True)


def index_dataset(data_dir, pids):
    """
    Percorre data_dir recursivamente e associa a cada paciente seus volumes
    (imagens DCE por aquisição) e a máscara. Retorna {pid: {'imgs': [...], 'mask': path}}.
    """
    idx = {p: {"imgs": [], "mask": None} for p in pids}
    for root, _, files in os.walk(data_dir):
        for f in files:
            low = f.lower()
            if not (low.endswith(".nii") or low.endswith(".nii.gz")):
                continue
            m = re.match(r"^([A-Za-z0-9]+_\d+)", f)       # ex.: ISPY1_1001
            if not m or m.group(1) not in idx:
                continue
            pid, path = m.group(1), os.path.join(root, f)
            if "mask" in low or "seg" in low:
                idx[pid]["mask"] = path
            else:
                a = re.search(r"a[cq]{2}_?(\d+)", low)   # aceita "acq" e "aqc"
                idx[pid]["imgs"].append((int(a.group(1)) if a else None, path))
    return idx


def phase_paths(info, row):
    """
    Escolhe os 3 volumes (pré, precoce, tardio) de um paciente.
    * Versão MinCrop traz exatamente 3 volumes -> usa a ordem das aquisições.
    * Se houver mais de 3, usa os índices pre/post_early/post_late do CSV.
    """
    entries = sorted(info["imgs"], key=lambda t: (10**6 if t[0] is None else t[0], t[1]))
    paths = [p for _, p in entries]
    if len(paths) == 3:
        return paths
    ids = [int(row["pre"]), int(row["post_early"]), int(row["post_late"])]
    acq = {a: p for a, p in entries if a is not None}
    if all(i in acq for i in ids):
        return [acq[i] for i in ids]
    if len(paths) > max(ids):
        return [paths[i] for i in ids]
    return None


def load_volume(path):
    """Lê um NIfTI como array float32 (X, Y, Z); descarta dimensões extras."""
    a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    while a.ndim > 3:
        a = a[..., 0]
    # ATENÇÃO: nos arquivos do BreastDCEDL o array vem como (Z, linhas, colunas), ex.:
    # (30, 256, 256), com o eixo dos cortes PRIMEIRO. Movemos o eixo de menor tamanho
    # (nº de cortes) para o fim, ficando (linhas, colunas, Z): a[:, :, z] é a imagem 2D
    # na orientação natural (a mesma exibida pelo corte do arquivo).
    zax = int(np.argmin(a.shape))
    return np.moveaxis(a, zax, -1)


# ------------------------------------------------------------------------------
# 4. PRÉ-PROCESSAMENTO E "BANCO DE CORTES" EM MEMÓRIA
# ------------------------------------------------------------------------------
class Bank:
    """
    Guarda todos os cortes tumorais de todos os pacientes como um único tensor
    X (N, 4, S, S) em float16, canais = [pré, precoce, tardia, máscara].
    Metadados por corte: paciente, partição, rótulo, índice z e área do tumor.
    """
    pass


def preprocess_patient(paths, mask_path, row, S, margin):
    # carrega as 3 fases + máscara, normaliza, seleciona cortes tumorais e redimensiona.
    vols = [load_volume(p) for p in paths]
    shape = vols[0].shape
    if mask_path is not None:
        mask = load_volume(mask_path) > 0
        if mask.shape != shape:                      # segurança: formato diferente
            mask = np.zeros(shape, bool)
    else:
        mask = np.zeros(shape, bool)
    stack = np.stack(vols, 0)                        # (3, X, Y, Z)

    # Normalização conjunta: mesmo fator para as 3 fases (preserva o realce relativo)
    scale = float(np.percentile(stack, 99.5))
    if scale <= 0:
        scale = float(stack.max()) + 1e-6
    stack = np.clip(stack / scale, 0, 1)

    # Cortes que contêm tumor (pela máscara); fallback: faixa do CSV; depois: miolo do volume
    Z = shape[2]
    zs = np.where(mask.any(axis=(0, 1)))[0]
    if len(zs) == 0:
        s, e = int(row.get("mask_start", Z * 0.2)), int(row.get("mask_end", Z * 0.8))
        zs = np.arange(max(0, s), min(Z - 1, e) + 1)
    z0, z1 = max(int(zs.min()) - margin, 0), min(int(zs.max()) + margin, Z - 1)
    zsel = np.arange(z0, z1 + 1)

    # (Zs, 3, X, Y) -> redimensiona para (S, S)
    t = torch.from_numpy(stack[:, :, :, zsel]).permute(3, 0, 1, 2).contiguous()
    m = torch.from_numpy(mask[:, :, zsel].astype(np.float32)).permute(2, 0, 1)[:, None]
    t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False, antialias=True)
    m = F.interpolate(m, size=(S, S), mode="nearest")
    area = m.sum(dim=(1, 2, 3)).numpy()              # área tumoral por corte (px)
    x4 = torch.cat([t, m], dim=1).clamp(0, 1).half() # (Zs, 4, S, S)
    return x4, area, zsel


def build_bank(meta, index, cfg, log=print):
    # Constrói (ou carrega do cache) o banco de cortes de todos os pacientes.
    os.makedirs(cfg.cache_dir, exist_ok=True)
    # o nome do cache inclui o CSV: assim o dataset de treino e um conjunto novo não se sobrescrevem
    csv_name = os.path.splitext(os.path.basename(cfg.csv_path or "dados"))[0]
    cache = os.path.join(cfg.cache_dir, "bank_%s_s%d_m%d.pt" % (csv_name, cfg.img_size, cfg.slice_margin))
    if os.path.exists(cache):
        try:
            d = torch.load(cache, weights_only=False)
            if set(d["pids"]) <= set(meta["pid"]):                     # mesmos pacientes do CSV
                log("Cache de pré-processamento carregado: " + cache)
                b = Bank()
                b.__dict__.update(d)
                return b
        except Exception as e:
            log("Cache inválido, refazendo (%s)" % e)

    Xs, y_row, pid_row, split_row, area_row, z_row = [], [], [], [], [], []
    pids, y_pat, split_pat = [], [], []
    for i, row in meta.iterrows():
        pid = row["pid"]
        info = index.get(pid)
        paths = phase_paths(info, row) if info else None
        if paths is None:
            log("  [aviso] %s sem as 3 fases, ignorado" % pid)
            continue
        x4, area, zsel = preprocess_patient(paths, info["mask"], row, cfg.img_size, cfg.slice_margin)
        k = len(pids)
        pids.append(pid)
        y_pat.append(int(row["pCR"]))
        split_pat.append(int(row["test"]))
        n = x4.shape[0]
        Xs.append(x4)
        y_row += [float(row["pCR"])] * n
        pid_row += [k] * n
        split_row += [int(row["test"])] * n
        area_row += list(area)
        z_row += list(zsel)
        if (k + 1) % 20 == 0:
            log("  pré-processados %d pacientes..." % (k + 1))
    b = Bank()
    b.X = torch.cat(Xs, 0)
    b.y_row = np.array(y_row, np.float32)
    b.pid_row = np.array(pid_row, np.int64)
    b.split_row = np.array(split_row, np.int64)
    b.area = np.array(area_row, np.float32)
    b.z_row = np.array(z_row, np.int64)
    b.pids, b.y_pat, b.split_pat = pids, np.array(y_pat), np.array(split_pat)
    torch.save(b.__dict__, cache)
    log("Banco: %d pacientes, %d cortes (cache salvo)." % (len(pids), b.X.shape[0]))
    return b


def describe_balance(bank, cfg, log=print):
    # Item (b): verifica o desbalanceamento entre pCR ausente/presente e propõe estratégia.
    names = {0: "treino", 2: "validação", 1: "teste"}
    for s in (0, 2, 1):
        sel = bank.split_pat == s
        n, pos = int(sel.sum()), int((bank.y_pat[sel] == 1).sum())
        rows = bank.split_row == s
        log("  %-9s: %3d pacientes (pCR=%d, %.1f%%) | %4d cortes" %
            (names[s], n, pos, 100.0 * pos / max(n, 1), int(rows.sum())))
    tr = bank.split_row == 0
    pos = bank.y_row[tr].sum()
    neg = tr.sum() - pos
    ratio = neg / max(pos, 1)
    log("Razão negativos:positivos no treino = %.2f:1 -> %s" %
        (ratio, "DESBALANCEAMENTO RELEVANTE" if ratio > 1.5 else "balanceado"))
    detalhe = {"pos_weight": "BCE com peso %.2f para a classe pCR" % ratio,
               "focal": "focal loss (gamma = %.1f)" % cfg.focal_gamma,
               "none": "BCE sem peso (sem tratamento)"}[cfg.loss_mode]
    log("Estratégia adotada: %s; limiar escolhido na validação." % detalhe)


# ------------------------------------------------------------------------------
# 5. AUMENTO DE DADOS (item b): 5 rotações x {original, espelhada} = 10 variações
# ------------------------------------------------------------------------------
def augment_batch(x, variant_ids):
    """
    Aplica, em lote, rotação (-20..20 graus) e espelhamento horizontal.
    x: (B, C, H, W); variant_ids: tensor (B,) com índices em VARIANTS.
    Usa uma matriz afim por amostra (affine_grid + grid_sample, interpolação bilinear,
    fundo preto). Rotação em torno do centro, sem alterar o tamanho da imagem.
    """
    ang = torch.tensor([math.radians(VARIANTS[int(v)][0]) for v in variant_ids],
                       dtype=x.dtype, device=x.device)
    flip = torch.tensor([-1.0 if VARIANTS[int(v)][1] else 1.0 for v in variant_ids],
                        dtype=x.dtype, device=x.device)
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(x.shape[0], 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0] = c * flip          # theta = R . diag(flip, 1)
    theta[:, 0, 1] = -s
    theta[:, 1, 0] = s * flip
    theta[:, 1, 1] = c
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def make_pairs(rows, k, rng):
    # Cria pares (linha_do_corte, variante). k=10 usa todas; k<10 sorteia k por corte.
    if k >= len(VARIANTS):
        v = np.tile(np.arange(len(VARIANTS)), (len(rows), 1))
    else:
        v = np.stack([rng.permutation(len(VARIANTS))[:k] for _ in rows])
    r = np.repeat(rows[:, None], v.shape[1], axis=1)
    return np.stack([r.ravel(), v.ravel()], 1)


def channels_for(phase, mask_mode):
    # Retorna (nº de canais de entrada da rede, nº de canais de imagem).
    n_img = len(PHASES[phase])
    return n_img + (1 if mask_mode == "channel" else 0), n_img


def make_input(x4, phase, mask_mode):
    """
    Seleciona os canais de entrada a partir do tensor [pré, precoce, tardia, máscara]:
      * fase única -> 1 canal; 'all' -> 3 canais (canal1=pré, 2=precoce, 3=tardia)
      * mask_mode 'channel' -> acrescenta a máscara como canal extra
      * mask_mode 'apply'   -> zera o fundo (multiplica as imagens pela máscara)
    """
    img = x4[:, PHASES[phase]]
    m = (x4[:, 3:4] > 0.5).to(x4.dtype)       # binariza (a máscara pode ter sido interpolada)
    if mask_mode == "channel":
        return torch.cat([img, m], 1)
    if mask_mode == "apply":
        return img * m
    return img


# ------------------------------------------------------------------------------
# 6. ARQUITETURAS (item d)
# ------------------------------------------------------------------------------
def _freeze_bn_eval(module):
    # BatchNorm cujos parâmetros estão congelados deve ficar em modo eval (estatísticas fixas).
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


class CustomCNN(nn.Module):
    """
    CNN própria. Camadas COM parâmetros: 5 convoluções + 2 totalmente conectadas = 7 (>= 5).
      Conv3x3(32)-BN-ReLU-MaxPool | Conv(64) | Conv(128) | Conv(128) | Conv(256)-BN-ReLU
      -> Global Average Pooling -> FC(128)+ReLU+Dropout(0.5) -> FC(1) (logit de pCR)
    Justificativas: filtros 3x3 empilhados (campo receptivo crescente com poucos
    parâmetros), BatchNorm (convergência mais rápida/estável), GAP (menos parâmetros
    que 'flatten', reduz overfitting e permite Grad-CAM), Dropout 0.5 na parte densa.
    """
    def __init__(self, in_ch, dropout=0.5):
        super().__init__()
        chans, layers, c = [32, 64, 128, 128, 256], [], in_ch
        for i, o in enumerate(chans):
            layers += [nn.Conv2d(c, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                       nn.ReLU(inplace=False)]
            if i < len(chans) - 1:
                layers.append(nn.MaxPool2d(2))
            c = o
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1, self.drop, self.fc2 = nn.Linear(256, 128), nn.Dropout(dropout), nn.Linear(128, 1)

    def gradcam_layer(self):                     # última ReLU convolucional
        return self.features[-1]

    def forward(self, x):
        f = self.gap(self.features(x)).flatten(1)
        return self.fc2(self.drop(F.relu(self.fc1(f)))).squeeze(1)


def _adapt_conv1(net, in_ch):
    # Adapta a 1ª convolução da ResNet para in_ch canais reutilizando os pesos ImageNet.
    if in_ch == 3:
        return
    old = net.conv1.weight.data                           # (64, 3, 7, 7)
    new = nn.Conv2d(in_ch, old.shape[0], 7, 2, 3, bias=False)
    if in_ch == 1:
        new.weight.data = old.sum(1, keepdim=True)        # equivale a replicar a imagem em RGB
    else:
        new.weight.data = old.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3.0 / in_ch
    net.conv1 = new


def _load_resnet(ctor, weights_enum, pretrained):
    # Cria a ResNet com pesos ImageNet; se não houver internet, usa pesos aleatórios.
    if pretrained:
        try:
            return ctor(weights=weights_enum), True
        except Exception as e:
            print("[aviso] Não foi possível obter pesos ImageNet (%s). Usando pesos aleatórios." % e)
    return ctor(weights=None), False


class ResNet50Cls(nn.Module):
    """
    ResNet50 com pesos ImageNet. Entrada adaptada a in_ch canais; a cabeça (fc) é
    substituída por Dropout-FC(256)-ReLU-Dropout-FC(1). O backbone é CONGELADO
    (só a parte totalmente conectada é treinada), conforme o enunciado; opcionalmente
    libera o layer4 (cfg.ft_layer4). Se a 1ª conv foi adaptada a um nº de canais
    diferente de 1/3 (caso da máscara como canal), ela também é treinável.
    """
    def __init__(self, in_ch, pretrained=True, ft_layer4=False):
        super().__init__()
        net, ok = _load_resnet(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, pretrained)
        _adapt_conv1(net, in_ch)
        net.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 256), nn.ReLU(inplace=True),
                               nn.Dropout(0.3), nn.Linear(256, 1))
        if ok:                                            # congela o backbone
            for p in net.parameters():
                p.requires_grad = False
            for p in net.fc.parameters():
                p.requires_grad = True
            if in_ch not in (1, 3):
                for p in net.conv1.parameters():
                    p.requires_grad = True
            if ft_layer4:
                for p in net.layer4.parameters():
                    p.requires_grad = True
        self.net = net

    def train(self, mode=True):
        super().train(mode)
        if mode:
            _freeze_bn_eval(self.net)
        return self

    def gradcam_layer(self):
        return self.net.layer4

    def forward(self, x):
        x = (x - 0.449) / 0.226                           # normalização estilo ImageNet (média dos canais)
        return self.net(x).squeeze(1)


class PhaseAttnNet(nn.Module):
    """
    NOVA SOLUÇÃO: ResNet18 compartilhada + atenção entre fases.
      1) Cada canal de entrada (fase) passa, separadamente, pelo MESMO tronco ResNet18
         (1 canal de entrada; stem e layer1 congelados) -> vetor f_c (512).
      2) Atenção aditiva aprende pesos a_c (softmax sobre as fases): fused = sum_c a_c f_c.
      3) Com >1 fase, concatena-se também (f_última - f_pré), que codifica o
         "washout"/dinâmica de contraste. Cabeça: FC(128)-Dropout-FC(1).
    Vantagem: ramos independentes por fase (sugestão do enunciado) com poucos
    parâmetros (tronco compartilhado) e atenção interpretável sobre qual fase pesa mais.
    """
    def __init__(self, in_ch, n_img_ch, pretrained=True):
        super().__init__()
        trunk, ok = _load_resnet(models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, pretrained)
        _adapt_conv1(trunk, 1)
        trunk.fc = nn.Identity()
        if ok:                                            # congela filtros de baixo nível
            for mod in (trunk.conv1, trunk.bn1, trunk.layer1):
                for p in mod.parameters():
                    p.requires_grad = False
        self.trunk, self.n_img = trunk, n_img_ch
        self.att = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        dim = 512 * (2 if n_img_ch > 1 else 1)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(dim, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            _freeze_bn_eval(self.trunk)
        return self

    def gradcam_layer(self):
        return self.trunk.layer4

    def forward(self, x):
        B, C, H, W = x.shape
        x = (x - 0.449) / 0.226
        f = self.trunk(x.reshape(B * C, 1, H, W)).view(B, C, -1)         # (B, C, 512)
        a = torch.softmax(self.att(f).squeeze(-1), dim=1)                # (B, C)
        z = (a.unsqueeze(-1) * f).sum(1)
        if self.n_img > 1:
            z = torch.cat([z, f[:, self.n_img - 1] - f[:, 0]], 1)
        return self.head(z).squeeze(1)


def build_model(name, in_ch, n_img, cfg):
    # Fábrica de modelos.
    if name == "cnn":
        return CustomCNN(in_ch)
    if name == "resnet50":
        return ResNet50Cls(in_ch, cfg.pretrained, cfg.ft_layer4)
    if name == "novel":
        return PhaseAttnNet(in_ch, n_img, cfg.pretrained)
    raise ValueError(name)


def param_groups(model, lr, factor):
    # Separa 'cabeça' (lr) de 'backbone' pré-treinado descongelado (lr*factor).
    bb, hd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = (n.startswith("net.") and not n.startswith("net.fc")) or n.startswith("trunk.")
        (bb if is_bb else hd).append(p)
    g = [{"params": hd, "lr": lr}] if hd else []
    if bb:
        g.append({"params": bb, "lr": lr * factor})
    return g


# ------------------------------------------------------------------------------
# 7. FUNÇÕES DE PERDA (tratamento do desbalanceamento)
# ------------------------------------------------------------------------------
def make_loss(mode, pos_weight, gamma, neg_frac, dev):
    """
    'pos_weight': BCE com peso maior para a classe pCR (rara);
    'focal'     : focal loss (Lin et al.) com alpha = fração de negativos;
    'none'      : BCE simples.
    """
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=dev)

    def loss_fn(logits, y):
        if mode == "pos_weight":
            return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        if mode == "focal":
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            pt = torch.exp(-bce)
            alpha = y * neg_frac + (1 - y) * (1 - neg_frac)
            return (alpha * (1 - pt) ** gamma * bce).mean()
        return F.binary_cross_entropy_with_logits(logits, y)
    return loss_fn


# ------------------------------------------------------------------------------
# 8. PREVISÃO, AGREGAÇÃO DE CORTES E MÉTRICAS (item c e relatório)
# ------------------------------------------------------------------------------
@torch.no_grad()
def predict_rows(model, bank, rows, phase, mask_mode, dev, bs=64):
    # Probabilidade de pCR (sigmoid) para cada corte indicado, SEM aumento de dados.
    model.eval()
    out = []
    for b in range(0, len(rows), bs):
        idx = torch.from_numpy(rows[b:b + bs])
        x = make_input(bank.X[idx].to(dev).float(), phase, mask_mode)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def aggregate_patient(p, area, strategy):
    """
    Combina as probabilidades dos cortes de UM paciente (cortes em ordem de z):
      central       : corte no centro da faixa tumoral
      maxarea       : corte com maior área de tumor na máscara
      mean / max    : média / máximo de todos os cortes
      top3          : média das 3 maiores probabilidades (modelo de agregação alternativo)
      area_weighted : média ponderada pela área tumoral (alternativo)
    """
    if strategy == "central":
        return float(p[len(p) // 2])
    if strategy == "maxarea":
        return float(p[int(np.argmax(area))])
    if strategy == "mean":
        return float(p.mean())
    if strategy == "max":
        return float(p.max())
    if strategy == "top3":
        return float(np.sort(p)[-3:].mean())
    if strategy == "area_weighted":
        return float((p * area).sum() / area.sum()) if area.sum() > 0 else float(p.mean())
    raise ValueError(strategy)


def patient_predictions(model, bank, rows, phase, mask_mode, dev):
    # Prediz todos os cortes de `rows` e agrega por paciente com todas as estratégias.
    probs = predict_rows(model, bank, rows, phase, mask_mode, dev)
    prow = bank.pid_row[rows]
    pids, ys, out = [], [], {a: [] for a in AGGS}
    for pi in np.unique(prow):
        sel = np.where(prow == pi)[0]            # `rows` está em ordem crescente => ordem de z
        p, ar = probs[sel], bank.area[rows[sel]]
        for a in AGGS:
            out[a].append(aggregate_patient(p, ar, a))
        pids.append(bank.pids[pi])
        ys.append(int(bank.y_pat[pi]))
    return pids, np.array(ys), {a: np.array(v) for a, v in out.items()}


def auc_score(y, p):
    # AUC-ROC pela estatística de Mann-Whitney (com postos médios para empates).
    y = np.asarray(y)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def binary_metrics(y, p, thr):
    # Sensibilidade, especificidade, precisão, acurácia, F1 e AUC no limiar `thr`.
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / max(len(y), 1)
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"sens": sens, "spec": spec, "prec": prec, "acc": acc, "f1": f1,
            "auc": auc_score(y, p), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def choose_threshold(y, p):
    # Limiar que maximiza o índice de Youden (sens+spec-1) na VALIDAÇÃO (desempate: perto de 0,5).
    best, thr = -9.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        m = binary_metrics(y, p, t)
        j = m["sens"] + m["spec"] - 1 - 1e-3 * abs(t - 0.5)
        if j > best:
            best, thr = j, float(t)
    return thr


# ------------------------------------------------------------------------------
# 9. TREINAMENTO DE UM EXPERIMENTO (modelo x fase x máscara)
# ------------------------------------------------------------------------------
def eval_loss(model, bank, pairs, spec, loss_fn, cfg, dev):
    # Loss média na validação (com as variações do aumento de dados).
    _, phase, mask_mode = spec
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(pairs), cfg.batch_size):
            pr = pairs[b:b + cfg.batch_size]
            idx = torch.from_numpy(pr[:, 0])
            x = augment_batch(bank.X[idx].to(dev).float(), pr[:, 1])
            x = make_input(x, phase, mask_mode)
            y = torch.from_numpy(bank.y_row[pr[:, 0]]).to(dev)
            tot += float(loss_fn(model(x), y)) * len(pr)
            n += len(pr)
    return tot / max(n, 1)


def save_loss_plot(name, hist, best_ep, out_dir):
    # Salva a curva de convergência (loss treino x validação) em PNG, para usar no relatório.
    try:
        from matplotlib.figure import Figure as _Fig   # Figure sem janela (funciona sem GUI)
        fig = _Fig(figsize=(6, 3.5), dpi=110)
        ax = fig.subplots()
        ep = np.arange(1, len(hist["train_loss"]) + 1)
        ax.plot(ep, hist["train_loss"], label="treino")
        ax.plot(ep, hist["val_loss"], label="validação")
        ax.axvline(best_ep, ls="--", c="gray", label="melhor época")
        ax.set_xlabel("época"); ax.set_ylabel("loss"); ax.set_title("Loss - " + name)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name + "_loss.png"))
    except Exception as e:                                # gráfico é opcional: não derruba o treino
        print("[aviso] não foi possível salvar o gráfico de loss: %s" % e)


def append_history_csv(result, cfg):
    """
    Acrescenta UMA linha por treino em resultados/historico_experimentos.csv
    (hiperparâmetros + métricas). Diferente do .json, que é sobrescrito quando o mesmo
    experimento é treinado de novo, este arquivo guarda TODAS as tentativas, servindo
    para comparar os resultados intermediários do ajuste de hiperparâmetros no relatório.
    """
    m_val, m_te = result["metrics"]["mean"]["val"], result["metrics"]["mean"]["test"]
    linha = {"data": time.strftime("%Y-%m-%d %H:%M"), "run": result["run"],
             "epochs": cfg.epochs, "batch": cfg.batch_size, "lr": cfg.lr, "wd": cfg.weight_decay,
             "loss": cfg.loss_mode, "aug_train": cfg.aug_train, "img": cfg.img_size,
             "margem": cfg.slice_margin, "criterio": cfg.criterio, "ft_layer4": cfg.ft_layer4, "seed": cfg.seed,
             "melhor_epoca": result["best_epoch"], "tempo_s": round(result["train_time_s"], 1),
             "auc_val": round(m_val["auc"], 3), "auc_teste": round(m_te["auc"], 3),
             "acc_teste": round(m_te["acc"], 3), "sens_teste": round(m_te["sens"], 3),
             "espec_teste": round(m_te["spec"], 3), "f1_teste": round(m_te["f1"], 3)}
    path = os.path.join(cfg.result_dir, "historico_experimentos.csv")
    pd.DataFrame([linha]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def train_one(spec, cfg, bank, log=print, stop=None, on_epoch=None):
    """
    Treina um experimento e salva checkpoint + JSON de resultados.
    spec = (modelo, fase, modo_mascara). Retorna o dicionário de resultados.
    """
    model_name, phase, mask_mode = spec
    name = run_name(spec, cfg.tag)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    seed_everything(cfg.seed)
    dev = get_device()
    in_ch, n_img = channels_for(phase, mask_mode)
    model = build_model(model_name, in_ch, n_img, cfg).to(dev)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("=== %s | %s | entrada: %s | máscara: %s | dispositivo: %s | parâmetros treináveis: %d ==="
        % (name, MODELS[model_name], PHASE_LABEL[phase], mask_mode, dev, ntrain))

    rows_tr = np.where(bank.split_row == 0)[0]
    rows_va = np.where(bank.split_row == 2)[0]
    rows_te = np.where(bank.split_row == 1)[0]
    pos = float(bank.y_row[rows_tr].sum())
    neg = len(rows_tr) - pos
    loss_fn = make_loss(cfg.loss_mode, neg / max(pos, 1), cfg.focal_gamma, neg / (pos + neg), dev)

    opt = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.backbone_lr_factor),
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    rng = np.random.RandomState(cfg.seed)
    val_pairs = make_pairs(rows_va, cfg.aug_val, np.random.RandomState(0))   # fixo em todas as épocas

    hist = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}
    best_score, best_state, best_ep, bad = -float("inf"), None, 0, 0
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        if stop is not None and stop.is_set():
            log("Treinamento interrompido pelo usuário.")
            break
        model.train()                                         # ativa dropout/BN (exceto partes congeladas)
        pairs = make_pairs(rows_tr, cfg.aug_train, rng)
        rng.shuffle(pairs)
        tot, n = 0.0, 0
        n_batches = math.ceil(len(pairs) / cfg.batch_size)
        for b in range(0, len(pairs), cfg.batch_size):
            if stop is not None and stop.is_set():
                break
            if (b // cfg.batch_size) % 25 == 0:          # mostra progresso (evita parecer travado)
                log("   ép %d: lote %d/%d | %.0fs" % (ep, b // cfg.batch_size + 1, n_batches, time.time() - t0))
            pr = pairs[b:b + cfg.batch_size]
            if len(pr) < 2:                                   # BatchNorm não aceita lote de 1
                continue
            idx = torch.from_numpy(pr[:, 0])
            x = augment_batch(bank.X[idx].to(dev).float(), pr[:, 1])      # aumento de dados
            x = make_input(x, phase, mask_mode)
            y = torch.from_numpy(bank.y_row[pr[:, 0]]).to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # estabilidade
            opt.step()
            tot += float(loss.detach()) * len(pr)
            n += len(pr)
        tr_loss = tot / max(n, 1)
        log("   ép %d: validando..." % ep)
        va_loss = eval_loss(model, bank, val_pairs, spec, loss_fn, cfg, dev)
        _, yv, pv = patient_predictions(model, bank, rows_va, phase, mask_mode, dev)
        va_auc = auc_score(yv, pv["mean"])
        sched.step(va_loss)
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["val_auc"].append(va_auc); hist["lr"].append(opt.param_groups[0]["lr"])
        log("ép %02d/%d | loss treino %.4f | loss val %.4f | AUC val (média) %.3f | lr %.1e | %.0fs"
            % (ep, cfg.epochs, tr_loss, va_loss, va_auc, opt.param_groups[0]["lr"], time.time() - t0))
        if on_epoch:
            on_epoch(name, hist)
        # Escolha da melhor época (early stopping). Com 'loss', a loss de validação subia desde a
        # época 1 (o modelo fica "confiante demais" nos erros) e o modelo salvo era sempre o da
        # 1ª época. Com 'auc', guarda-se a época que melhor ORDENA os pacientes da validação,
        # que é o que importa para classificar pCR. Empate de AUC: desempata pela menor loss.
        score = (va_auc - 1e-3 * va_loss) if cfg.criterio == "auc" and not np.isnan(va_auc) else -va_loss
        if score > best_score:                                # guarda o melhor modelo (early stopping)
            best_score, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log("Early stopping na época %d (melhor: %d)." % (ep, best_ep))
                break
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- avaliação final: limiar escolhido na validação, métricas na validação e no teste -----
    _, yv, pv = patient_predictions(model, bank, rows_va, phase, mask_mode, dev)
    pids_te, yt, pt = patient_predictions(model, bank, rows_te, phase, mask_mode, dev)
    metrics, thr_dict = {}, {}
    for a in AGGS:
        thr = choose_threshold(yv, pv[a])
        thr_dict[a] = thr
        metrics[a] = {"thr": thr, "val": binary_metrics(yv, pv[a], thr),
                      "test": binary_metrics(yt, pt[a], thr)}
    result = {"run": name, "model": model_name, "phase": phase, "mask": mask_mode, "tag": cfg.tag,
              "train_time_s": train_time, "best_epoch": best_ep, "epochs_run": len(hist["train_loss"]),
              "history": hist, "device": str(dev), "metrics": metrics,
              "test_patients": {"pids": pids_te, "y": yt.tolist(),
                                "p": {a: pt[a].tolist() for a in AGGS}},
              "cfg": {k: v for k, v in cfg.to_dict().items() if k not in ("data_dir", "csv_path")}}
    with open(os.path.join(cfg.result_dir, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    save_loss_plot(name, hist, best_ep, cfg.result_dir)
    append_history_csv(result, cfg)
    # checkpoint: pesos + configuração, para NÃO precisar retreinar na apresentação
    torch.save({"state": model.state_dict(), "spec": spec, "thr": thr_dict,
                "cfg": cfg.to_dict()}, os.path.join(cfg.model_dir, name + ".pt"))
    m = metrics["mean"]["test"]
    log("Teste (agregação 'média'): acc %.3f | sens %.3f | espec %.3f | prec %.3f | F1 %.3f | AUC %.3f | tempo %.0fs"
        % (m["acc"], m["sens"], m["spec"], m["prec"], m["f1"], m["auc"], train_time))
    return result


def run_experiments(specs, cfg, bank, log=print, stop=None, on_epoch=None):
    # Executa vários experimentos em sequência (usado pela GUI e pelo modo --train-all).
    for i, spec in enumerate(specs):
        if stop is not None and stop.is_set():
            break
        log("\n##### Experimento %d/%d #####" % (i + 1, len(specs)))
        try:
            train_one(spec, cfg, bank, log, stop, on_epoch)
        except Exception:
            log("ERRO em %s:\n%s" % (run_name(spec, cfg.tag), traceback.format_exc()))


def load_checkpoint(path, cfg=None):
    # Recarrega um modelo salvo (pesos + limiares). Retorna (modelo, ckpt).
    ck = torch.load(path, weights_only=False, map_location="cpu")
    mname, phase, mask = ck["spec"]
    c = Config()
    for k, v in ck["cfg"].items():
        if hasattr(c, k) and not isinstance(getattr(type(c), k, None), property):
            setattr(c, k, v)
    c.pretrained = False                      # os pesos virão do checkpoint (não baixa nada)
    in_ch, n_img = channels_for(phase, mask)
    model = build_model(mname, in_ch, n_img, c)
    model.load_state_dict(ck["state"])
    return model.to(get_device()).eval(), ck


def evaluate_saved(model, ck, bank, split=1):
    """
    Avalia um modelo JÁ TREINADO (checkpoint) nos pacientes da partição `split`
    (1 = teste). Usado para o conjunto-teste surpresa: não treina nada, só prediz.
    Retorna um texto com as métricas por agregação e a lista de previsões por paciente.
    """
    _, phase, mask_mode = ck["spec"]
    rows = np.where(bank.split_row == split)[0]
    if len(rows) == 0:
        return "Nenhum paciente na partição de teste.", []
    pids, y, p = patient_predictions(model, bank, rows, phase, mask_mode, get_device())
    linhas = ["Pacientes avaliados: %d" % len(pids)]
    if (y >= 0).all():                                     # só calcula métricas se houver rótulo
        linhas.append("%-38s %5s %5s %5s %5s %5s %5s" % ("agregação", "acc", "sens", "espec", "prec", "F1", "AUC"))
        for a in AGGS:
            m = binary_metrics(y, p[a], ck["thr"][a])
            linhas.append("%-38s %.3f %.3f %.3f %.3f %.3f %.3f" % (AGG_LABEL[a], m["acc"], m["sens"],
                                                                  m["spec"], m["prec"], m["f1"], m["auc"]))
    else:
        linhas.append("CSV sem coluna pCR: apenas as previsões foram geradas.")
    prev = [{"pid": pid, "pCR_real": int(yy), "prob_media": round(float(pp), 4),
             "pred_media": int(pp >= ck["thr"]["mean"])} for pid, yy, pp in zip(pids, y, p["mean"])]
    return "\n".join(linhas), prev


def load_all_results(cfg):
    # Lê todos os JSONs de resultados salvos.
    out = []
    for f in sorted(glob.glob(os.path.join(cfg.result_dir, "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except Exception:
            pass
    return out


# ------------------------------------------------------------------------------
# 10. GRAD-CAM (item e)
# ------------------------------------------------------------------------------
def grad_cam(model, x):
    """
    Grad-CAM: pondera os mapas de ativação da última camada convolucional pelo
    gradiente médio do logit de pCR em relação a eles; ReLU e normalização [0,1].
    x: (B, C, H, W). Retorna (cams (B,H,W) numpy, prob (B,) numpy).
    Para a rede com ramos por fase, o mapa é a média entre os ramos.
    """
    layer, store = model.gradcam_layer(), {}

    def fwd_hook(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    model.eval()
    x = x.clone().requires_grad_(True)       # garante grafo mesmo com backbone congelado
    with torch.enable_grad():
        logit = model(x)
        model.zero_grad()
        logit.sum().backward()
    h.remove()
    a, g = store["a"], store["g"]
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))          # (N, h, w)
    B = x.shape[0]
    if cam.shape[0] != B:                                                # ramos por fase
        cam = cam.view(B, -1, cam.shape[-2], cam.shape[-1]).mean(1)
    cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mn = cam.amin(dim=(1, 2), keepdim=True)
    mx = cam.amax(dim=(1, 2), keepdim=True)
    cam = (cam - mn) / (mx - mn + 1e-8)
    return cam.detach().cpu().numpy(), torch.sigmoid(logit).detach().cpu().numpy()


def cam_in_tumor(cam, mask):
    # Fração da 'energia' do Grad-CAM que cai dentro do tumor (0-1).
    s = float(cam.sum())
    return float((cam * mask).sum() / s) if s > 0 else float("nan")


# ------------------------------------------------------------------------------
# 11. UTILITÁRIOS DE EXIBIÇÃO
# ------------------------------------------------------------------------------
def to_display(arr2d, rot=0, flip=False):
    """
    Prepara um corte 2D (linhas, colunas) para exibição na orientação natural do
    arquivo (como em a[z] no array original). `rot` (múltiplos de 90°) e `flip`
    permitem ajustar a orientação pela interface, se necessário.
    """
    out = np.rot90(arr2d, rot)
    return out[:, ::-1] if flip else out


# ------------------------------------------------------------------------------
# 12. INTERFACE GRÁFICA (Tkinter + Matplotlib)
# ------------------------------------------------------------------------------
if GUI_OK:

    def make_canvas(parent, figsize, nrows=1, ncols=1):
        # cria uma figura Matplotlib embutida no Tkinter.
        fig = Figure(figsize=figsize, dpi=90)
        axes = fig.subplots(nrows, ncols, squeeze=False)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, axes, canvas

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Classificação de pCR em DCE-MRI (I-SPY1 / BreastDCEDL)")
            self.geometry("1320x860")
            self.cfg = Config()
            self.meta, self.index, self.bank = None, None, None
            self.q = queue.Queue()                    # fila thread -> interface
            self.stop = threading.Event()
            self.vis_vol = self.vis_mask = None
            self.vis_rot, self.vis_flip = 0, False
            self.cls = None
            self._build_top()
            self.nb = ttk.Notebook(self)
            self.nb.pack(fill="both", expand=True, padx=4, pady=4)
            self._build_viewer()
            self._build_train()
            self._build_results()
            self._build_cls()
            self.after(100, self._poll)

        # ---------------------------- barra superior ----------------------------
        def _build_top(self):
            f = ttk.Frame(self)
            f.pack(fill="x", padx=6, pady=4)
            self.v_data = tk.StringVar(value=self.cfg.data_dir)
            self.v_csv = tk.StringVar(value=find_csv(self.cfg) or "")
            self.v_size = tk.StringVar(value=str(self.cfg.img_size))
            ttk.Label(f, text="Pasta das imagens:").grid(row=0, column=0, sticky="e")
            ttk.Entry(f, textvariable=self.v_data, width=60).grid(row=0, column=1)
            ttk.Button(f, text="...", width=3, command=lambda: self.v_data.set(
                filedialog.askdirectory() or self.v_data.get())).grid(row=0, column=2)
            ttk.Label(f, text="CSV:").grid(row=1, column=0, sticky="e")
            ttk.Entry(f, textvariable=self.v_csv, width=60).grid(row=1, column=1)
            ttk.Button(f, text="...", width=3, command=lambda: self.v_csv.set(
                filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]) or self.v_csv.get())
                       ).grid(row=1, column=2)
            ttk.Label(f, text="Tamanho (px):").grid(row=0, column=3, padx=6)
            ttk.Entry(f, textvariable=self.v_size, width=6).grid(row=0, column=4)
            ttk.Button(f, text="Carregar dataset", command=self.load_dataset).grid(row=0, column=5, padx=8)
            self.lbl_status = ttk.Label(f, text="Dataset não carregado.")
            self.lbl_status.grid(row=1, column=3, columnspan=3, sticky="w", padx=6)

        def load_dataset(self):
            # Indexa arquivos e constrói o banco de cortes (em thread, para não travar a GUI).
            self.cfg.data_dir, self.cfg.csv_path = self.v_data.get(), self.v_csv.get()
            csv = find_csv(self.cfg)
            if csv is None:
                return messagebox.showerror("Erro", "CSV não encontrado. Coloque-o na pasta 'dataset' ou escolha-o manualmente.")
            self.cfg.csv_path = csv
            self.v_csv.set(csv)
            try:
                self.cfg.img_size = int(self.v_size.get())
            except ValueError:
                return messagebox.showerror("Erro", "Tamanho inválido.")
            self.lbl_status.config(text="Carregando... (pode levar alguns minutos na 1ª vez)")

            def work():
                try:
                    meta = read_meta(self.cfg.csv_path)
                    index = index_dataset(self.cfg.data_dir, list(meta["pid"]))
                    ok = sum(1 for v in index.values() if len(v["imgs"]) >= 3)
                    self.q.put(("log", "Pacientes no CSV: %d | com >=3 volumes encontrados: %d" % (len(meta), ok)))
                    self.meta, self.index = meta, index
                    self.q.put(("index",))
                    bank = build_bank(meta, index, self.cfg, lambda s: self.q.put(("log", s)))
                    self.bank = bank
                    describe_balance(bank, self.cfg, lambda s: self.q.put(("log", s)))
                    self.q.put(("bank",))
                except Exception:
                    self.q.put(("log", "ERRO ao carregar:\n" + traceback.format_exc()))
                    self.q.put(("status", "Erro ao carregar (veja o log)."))
            threading.Thread(target=work, daemon=True).start()

        # ----------------------- fila de mensagens da thread -----------------------
        def _poll(self):
            try:
                while True:
                    msg = self.q.get_nowait()
                    k = msg[0]
                    if k == "log":
                        self.log.insert("end", msg[1] + "\n"); self.log.see("end")
                    elif k == "status":
                        self.lbl_status.config(text=msg[1])
                    elif k == "index":
                        pids = [p for p, v in self.index.items() if len(v["imgs"]) >= 3]
                        self.cb_pat["values"] = pids
                        if pids:
                            self.cb_pat.set(pids[0]); self.vis_load_patient()
                    elif k == "bank":
                        pids = list(self.bank.pids)
                        self.cb_cpat["values"] = pids
                        if pids:
                            self.cb_cpat.set(pids[0])
                        self.lbl_status.config(text="Dataset pronto: %d pacientes, %d cortes."
                                                    % (len(pids), self.bank.X.shape[0]))
                        self.refresh_runs()
                    elif k == "curve":
                        self.draw_curve(msg[1], msg[2])
                    elif k == "done":
                        self.btn_train.config(state="normal")
                        self.refresh_results(); self.refresh_runs()
            except queue.Empty:
                pass
            self.after(100, self._poll)

        # ------------------------- ABA 1: VISUALIZADOR (a, b) -------------------------
        def _build_viewer(self):
            tab = ttk.Frame(self.nb); self.nb.add(tab, text="Visualizador")
            ctl = ttk.Frame(tab); ctl.pack(fill="x", pady=4)
            ttk.Label(ctl, text="Paciente:").pack(side="left")
            self.cb_pat = ttk.Combobox(ctl, width=14, state="readonly"); self.cb_pat.pack(side="left", padx=4)
            self.cb_pat.bind("<<ComboboxSelected>>", lambda e: self.vis_load_patient())
            ttk.Label(ctl, text="Fase:").pack(side="left")
            self.cb_vphase = ttk.Combobox(ctl, width=14, state="readonly",
                                          values=["pré-contraste", "pós-precoce", "pós-tardio"])
            self.cb_vphase.set("pós-precoce"); self.cb_vphase.pack(side="left", padx=4)
            self.cb_vphase.bind("<<ComboboxSelected>>", lambda e: self.vis_load_patient())
            ttk.Button(ctl, text="Abrir .nii (imagem)", command=self.vis_open_img).pack(side="left", padx=4)
            ttk.Button(ctl, text="Abrir .nii (máscara)", command=self.vis_open_mask).pack(side="left")
            ttk.Button(ctl, text="Girar 90°", command=self.vis_rotate).pack(side="left", padx=4)
            ttk.Button(ctl, text="Espelhar", command=self.vis_mirror).pack(side="left")
            ttk.Button(ctl, text="Mostrar 10 variações (aumento de dados)",
                       command=self.show_aug).pack(side="left", padx=8)
            self.vis_z = tk.Scale(tab, from_=0, to=1, orient="horizontal", label="Corte z",
                                  command=lambda v: self.vis_draw())
            self.vis_z.pack(fill="x", padx=10)
            self.vis_fig, self.vis_ax, self.vis_cv = make_canvas(tab, (10, 5), 1, 2)
            w = self.vis_cv.get_tk_widget()
            w.bind("<MouseWheel>", lambda e: self.vis_step(1 if e.delta > 0 else -1))   # Windows/macOS
            w.bind("<Button-4>", lambda e: self.vis_step(1))                            # Linux
            w.bind("<Button-5>", lambda e: self.vis_step(-1))

        def vis_step(self, d):
            self.vis_z.set(min(max(self.vis_z.get() + d, 0), int(self.vis_z.cget("to"))))

        def vis_set_volumes(self, img, mask):
            self.vis_vol, self.vis_mask = img, mask
            self.vis_z.config(to=img.shape[2] - 1)
            self.vis_z.set(img.shape[2] // 2)
            nz = img[img > 0]
            self.vis_lo, self.vis_hi = (float(np.percentile(nz, 1)), float(np.percentile(nz, 99.5))) \
                if nz.size else (0.0, 1.0)
            self.vis_draw()

        def vis_load_patient(self):
            pid = self.cb_pat.get()
            if not pid or self.index is None:
                return
            info = self.index[pid]
            row = self.meta[self.meta["pid"] == pid].iloc[0]
            paths = phase_paths(info, row)
            if paths is None:
                return
            k = ["pré-contraste", "pós-precoce", "pós-tardio"].index(self.cb_vphase.get())
            mask = load_volume(info["mask"]) if info["mask"] else None
            self.vis_set_volumes(load_volume(paths[k]), mask)

        def vis_open_img(self):
            p = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
            if p:
                self.vis_set_volumes(load_volume(p), None)

        def vis_open_mask(self):
            p = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
            if p and self.vis_vol is not None:
                self.vis_mask = load_volume(p); self.vis_draw()

        def vis_rotate(self):
            self.vis_rot = (self.vis_rot + 1) % 4; self.vis_draw()

        def vis_mirror(self):
            self.vis_flip = not self.vis_flip; self.vis_draw()

        def vis_draw(self):
            # Mostra o corte z da imagem (esquerda) e da segmentação (direita).
            if self.vis_vol is None:
                return
            z = int(self.vis_z.get())
            a1, a2 = self.vis_ax[0]
            a1.clear(); a2.clear()
            a1.imshow(to_display(self.vis_vol[:, :, z], self.vis_rot, self.vis_flip), cmap="gray",
                      vmin=self.vis_lo, vmax=self.vis_hi, origin="upper")
            a1.set_title("Imagem - corte z = %d / %d" % (z, self.vis_vol.shape[2] - 1))
            if self.vis_mask is not None and self.vis_mask.shape == self.vis_vol.shape:
                a2.imshow(to_display(self.vis_mask[:, :, z] > 0, self.vis_rot, self.vis_flip),
                          cmap="gray", vmin=0, vmax=1, origin="upper")
                a2.set_title("Segmentação do tumor (%d px)" % int((self.vis_mask[:, :, z] > 0).sum()))
            else:
                a2.text(0.5, 0.5, "sem máscara", ha="center", va="center", transform=a2.transAxes)
            for a in (a1, a2):
                a.axis("off")
            self.vis_cv.draw_idle()

        def show_aug(self):
            # item (b): exibe as 10 variações (5 rotações x original/espelhada) do corte atual.
            if self.vis_vol is None:
                return messagebox.showinfo("Aviso", "Carregue uma imagem primeiro.")
            z = int(self.vis_z.get())
            sl = np.clip((self.vis_vol[:, :, z] - self.vis_lo) / (self.vis_hi - self.vis_lo + 1e-8), 0, 1)
            x = torch.from_numpy(sl.copy())[None, None].float().repeat(len(VARIANTS), 1, 1, 1)
            out = augment_batch(x, torch.arange(len(VARIANTS)))[:, 0].numpy()
            win = tk.Toplevel(self); win.title("Aumento de dados: 10 variações")
            fig, axes, cv = make_canvas(win, (11, 5), 2, 5)
            for i, (ang, fl) in enumerate(VARIANTS):
                ax = axes[i // 5][i % 5]
                ax.imshow(to_display(out[i], self.vis_rot, self.vis_flip), cmap="gray", vmin=0, vmax=1)
                ax.set_title("%+d°%s" % (ang, " espelhada" if fl else ""), fontsize=9); ax.axis("off")
            cv.draw()

        # ------------------------- ABA 2: TREINAMENTO (b, c, d) -------------------------
        def _build_train(self):
            tab = ttk.Frame(self.nb); self.nb.add(tab, text="Treinamento")
            left = ttk.Frame(tab); left.pack(side="left", fill="y", padx=6, pady=6)
            self.pv = {}
            for label, key, val in [("Épocas", "epochs", self.cfg.epochs), ("Tamanho do lote", "batch_size", self.cfg.batch_size),
                                    ("Learning rate", "lr", self.cfg.lr), ("Weight decay", "weight_decay", self.cfg.weight_decay),
                                    ("Paciência (early stop)", "patience", self.cfg.patience),
                                    ("Variações/corte no treino (1-10)", "aug_train", self.cfg.aug_train),
                                    ("Variações/corte na validação (1-10)", "aug_val", self.cfg.aug_val)]:
                r = ttk.Frame(left); r.pack(fill="x")
                ttk.Label(r, text=label, width=32).pack(side="left")
                self.pv[key] = tk.StringVar(value=str(val))
                ttk.Entry(r, textvariable=self.pv[key], width=9).pack(side="left")
            r = ttk.Frame(left); r.pack(fill="x", pady=2)
            ttk.Label(r, text="Perda / desbalanceamento", width=32).pack(side="left")
            self.pv["loss_mode"] = tk.StringVar(value=self.cfg.loss_mode)
            ttk.Combobox(r, textvariable=self.pv["loss_mode"], width=10, state="readonly",
                         values=["pos_weight", "focal", "none"]).pack(side="left")
            r = ttk.Frame(left); r.pack(fill="x", pady=2)
            ttk.Label(r, text="Máscara", width=32).pack(side="left")
            self.v_mask = tk.StringVar(value="none")
            ttk.Combobox(r, textvariable=self.v_mask, width=10, state="readonly", values=MASK_MODES).pack(side="left")
            self.v_pre = tk.BooleanVar(value=True); self.v_ft4 = tk.BooleanVar(value=False)
            ttk.Checkbutton(left, text="Usar pesos ImageNet", variable=self.v_pre).pack(anchor="w")
            ttk.Checkbutton(left, text="ResNet50: ajustar também o layer4", variable=self.v_ft4).pack(anchor="w")
            ttk.Label(left, text="\nModelos:").pack(anchor="w")
            self.v_models = {m: tk.BooleanVar(value=True) for m in MODELS}
            for m, v in self.v_models.items():
                ttk.Checkbutton(left, text=MODELS[m], variable=v).pack(anchor="w")
            ttk.Label(left, text="\nEntradas (fases):").pack(anchor="w")
            self.v_phases = {p: tk.BooleanVar(value=True) for p in PHASES}
            for p, v in self.v_phases.items():
                ttk.Checkbutton(left, text=PHASE_LABEL[p], variable=v).pack(anchor="w")
            self.btn_train = ttk.Button(left, text="Treinar selecionados", command=self.start_training)
            self.btn_train.pack(fill="x", pady=8)
            ttk.Button(left, text="Parar", command=self.stop.set).pack(fill="x")
            right = ttk.Frame(tab); right.pack(side="left", fill="both", expand=True)
            self.log = scrolledtext.ScrolledText(right, height=18)
            self.log.pack(fill="both", expand=False)
            self.cur_fig, self.cur_ax, self.cur_cv = make_canvas(right, (7, 3), 1, 1)

        def apply_params(self):
            # copia os valores digitados na GUI para a configuração.
            c = self.cfg
            for k in ("epochs", "batch_size", "patience", "aug_train", "aug_val"):
                setattr(c, k, int(self.pv[k].get()))
            c.aug_train = min(max(c.aug_train, 1), 10); c.aug_val = min(max(c.aug_val, 1), 10)
            c.lr, c.weight_decay = float(self.pv["lr"].get()), float(self.pv["weight_decay"].get())
            c.loss_mode, c.pretrained, c.ft_layer4 = self.pv["loss_mode"].get(), self.v_pre.get(), self.v_ft4.get()

        def start_training(self):
            if self.bank is None:
                return messagebox.showinfo("Aviso", "Carregue o dataset primeiro.")
            try:
                self.apply_params()
            except ValueError:
                return messagebox.showerror("Erro", "Parâmetro numérico inválido.")
            specs = [(m, p, self.v_mask.get()) for m in MODELS if self.v_models[m].get()
                     for p in PHASES if self.v_phases[p].get()]
            if not specs:
                return messagebox.showinfo("Aviso", "Selecione ao menos um modelo e uma entrada.")
            self.stop.clear(); self.btn_train.config(state="disabled")

            def work():
                run_experiments(specs, self.cfg, self.bank, lambda s: self.q.put(("log", s)), self.stop,
                                lambda n, h: self.q.put(("curve", n, {k: list(v) for k, v in h.items()})))
                self.q.put(("log", "\n*** Treinamentos finalizados. ***"))
                self.q.put(("done",))
            threading.Thread(target=work, daemon=True).start()

        def draw_curve(self, title, h):
            # curva de convergência da loss (treino x validação) do experimento corrente.
            ax = self.cur_ax[0][0]; ax.clear()
            ax.plot(h["train_loss"], label="treino"); ax.plot(h["val_loss"], label="validação")
            ax.set_title("Loss - " + title); ax.set_xlabel("época"); ax.legend(); ax.grid(alpha=.3)
            self.cur_fig.tight_layout(); self.cur_cv.draw_idle()

        # ------------------------- ABA 3: RESULTADOS -------------------------
        def _build_results(self):
            tab = ttk.Frame(self.nb); self.nb.add(tab, text="Resultados")
            top = ttk.Frame(tab); top.pack(fill="x", pady=4)
            ttk.Label(top, text="Agregação de cortes:").pack(side="left")
            self.cb_agg = ttk.Combobox(top, state="readonly", width=34, values=[AGG_LABEL[a] for a in AGGS])
            self.cb_agg.set(AGG_LABEL["mean"]); self.cb_agg.pack(side="left", padx=4)
            self.cb_agg.bind("<<ComboboxSelected>>", lambda e: self.refresh_results())
            ttk.Label(top, text="Métrica do gráfico:").pack(side="left", padx=6)
            self.cb_met = ttk.Combobox(top, state="readonly", width=8, values=["auc", "f1", "acc", "sens", "spec"])
            self.cb_met.set("auc"); self.cb_met.pack(side="left")
            self.cb_met.bind("<<ComboboxSelected>>", lambda e: self.refresh_results())
            # rodada = máscara + tag: o gráfico e a hipótese comparam só experimentos da MESMA rodada
            ttk.Label(top, text="Rodada (máscara | tag):").pack(side="left", padx=6)
            self.cb_rod = ttk.Combobox(top, state="readonly", width=28); self.cb_rod.pack(side="left")
            self.cb_rod.bind("<<ComboboxSelected>>", lambda e: self.refresh_results())
            ttk.Button(top, text="Atualizar", command=self.refresh_results).pack(side="left", padx=8)
            cols = ("experimento", "entrada", "mascara", "limiar", "acc", "sens", "espec", "prec", "f1", "auc", "tempo(s)")
            self.tree = ttk.Treeview(tab, columns=cols, show="headings", height=9)
            for c in cols:
                self.tree.heading(c, text=c)
                self.tree.column(c, width=200 if c == "experimento" else 80, anchor="center")
            self.tree.pack(fill="x")
            self.tree.bind("<<TreeviewSelect>>", self.on_select_result)
            body = ttk.Frame(tab); body.pack(fill="both", expand=True)
            lf = ttk.Frame(body); lf.pack(side="left", fill="both", expand=True)
            rf = ttk.Frame(body); rf.pack(side="left", fill="both", expand=True)
            self.res_fig, self.res_ax, self.res_cv = make_canvas(lf, (5, 3.2), 1, 1)
            self.cmp_fig, self.cmp_ax, self.cmp_cv = make_canvas(rf, (5, 3.2), 1, 1)
            self.txt_res = scrolledtext.ScrolledText(tab, height=9); self.txt_res.pack(fill="x")
            self.res_rows = {}

        def cur_agg(self):
            return AGGS[[AGG_LABEL[a] for a in AGGS].index(self.cb_agg.get())]

        def refresh_results(self):
            # preenche a tabela (métricas no TESTE), o gráfico comparativo e o resumo da hipótese.
            agg, met = self.cur_agg(), self.cb_met.get()
            res = load_all_results(self.cfg)
            self.tree.delete(*self.tree.get_children()); self.res_rows = {}
            for r in res:
                m = r["metrics"][agg]["test"]
                iid = self.tree.insert("", "end", values=(
                    r["run"], PHASE_LABEL[r["phase"]], r["mask"], "%.2f" % r["metrics"][agg]["thr"],
                    "%.3f" % m["acc"], "%.3f" % m["sens"], "%.3f" % m["spec"], "%.3f" % m["prec"],
                    "%.3f" % m["f1"], "%.3f" % m["auc"], "%.0f" % r["train_time_s"]))
                self.res_rows[iid] = r
            # gráfico comparativo: barras por modelo, cor = entrada
            ax = self.cmp_ax[0][0]; ax.clear()
            rodada = lambda r: "%s | %s" % (r["mask"], r.get("tag", "") or "-")
            rodadas = sorted(set(rodada(r) for r in res))
            self.cb_rod["values"] = rodadas
            if rodadas and self.cb_rod.get() not in rodadas:
                self.cb_rod.set(rodadas[0])
            sel = [r for r in res if rodada(r) == self.cb_rod.get()]
            models_ = [m for m in MODELS if any(r["model"] == m for r in sel)]
            w = 0.2
            for j, ph in enumerate(PHASES):
                vals = []
                for m in models_:
                    v = [r["metrics"][agg]["test"][met] for r in sel if r["model"] == m and r["phase"] == ph]
                    vals.append(v[0] if v else 0)
                ax.bar(np.arange(len(models_)) + j * w, vals, w, label=PHASE_LABEL[ph])
            ax.set_xticks(np.arange(len(models_)) + 1.5 * w); ax.set_xticklabels(models_)
            ax.set_title("%s no teste (%s)" % (met.upper(), AGG_LABEL[agg]), fontsize=9)
            ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y"); self.cmp_fig.tight_layout(); self.cmp_cv.draw_idle()
            # resumo da hipótese: 3 fases vs. melhor fase única
            lines = ["HIPÓTESE: as 3 fases juntas superam cada fase isolada? (métrica %s, teste, agregação '%s')" % (met, agg)]
            for m in models_:
                d = {r["phase"]: r["metrics"][agg]["test"][met] for r in sel if r["model"] == m}
                if "all" in d and len(d) > 1:
                    best1 = max((v, k) for k, v in d.items() if k != "all")
                    lines.append("  %-9s 3 fases = %.3f | melhor fase única (%s) = %.3f -> %s"
                                 % (m, d["all"], PHASE_LABEL[best1[1]], best1[0],
                                    "3 fases MELHOR" if d["all"] > best1[0] else "fase única igual/melhor"))
            lines.append("Obs.: o teste tem poucos pacientes (~35, ~12 com pCR); diferenças pequenas são "
                         "estatisticamente frágeis. Discuta isso no relatório.")
            self.txt_res.delete("1.0", "end"); self.txt_res.insert("end", "\n".join(lines))

        def on_select_result(self, _):
            # ao selecionar uma linha: curva de loss, matriz de confusão e pacientes errados.
            s = self.tree.selection()
            if not s:
                return
            r, agg = self.res_rows[s[0]], self.cur_agg()
            h = r["history"]; ax = self.res_ax[0][0]; ax.clear()
            ax.plot(h["train_loss"], label="treino"); ax.plot(h["val_loss"], label="validação")
            ax.axvline(r["best_epoch"] - 1, ls="--", c="gray")
            ax.set_title("Convergência - " + r["run"], fontsize=9); ax.legend(); ax.grid(alpha=.3)
            self.res_fig.tight_layout(); self.res_cv.draw_idle()
            m, tp = r["metrics"][agg]["test"], r["test_patients"]
            thr = r["metrics"][agg]["thr"]
            fp = [p for p, y, v in zip(tp["pids"], tp["y"], tp["p"][agg]) if y == 0 and v >= thr]
            fn = [p for p, y, v in zip(tp["pids"], tp["y"], tp["p"][agg]) if y == 1 and v < thr]
            vp = [p for p, y, v in zip(tp["pids"], tp["y"], tp["p"][agg]) if y == 1 and v >= thr]
            vn = [p for p, y, v in zip(tp["pids"], tp["y"], tp["p"][agg]) if y == 0 and v < thr]
            self.txt_res.delete("1.0", "end")
            self.txt_res.insert("end", "%s | agregação: %s | limiar %.2f\nMatriz de confusão (teste): VP=%d VN=%d FP=%d FN=%d\n"
                                "Falsos positivos (previu pCR, não ocorreu): %s\nFalsos negativos (pCR não detectado): %s\n"
                                "Acertos pCR presente (VP): %s\nAcertos pCR ausente (VN, até 10): %s\n"
                                "Tempo de treino: %.0fs | melhor época: %d | épocas executadas: %d"
                                % (r["run"], AGG_LABEL[agg], thr, m["tp"], m["tn"], m["fp"], m["fn"],
                                   ", ".join(fp) or "-", ", ".join(fn) or "-", ", ".join(vp) or "-",
                                   ", ".join(vn[:10]) or "-", r["train_time_s"],
                                   r["best_epoch"], r["epochs_run"]))

        # ------------------------- ABA 4: CLASSIFICAR + GRAD-CAM (e) -------------------------
        def _build_cls(self):
            tab = ttk.Frame(self.nb); self.nb.add(tab, text="Classificar + Grad-CAM")
            top = ttk.Frame(tab); top.pack(fill="x", pady=4)
            ttk.Label(top, text="Modelo salvo:").pack(side="left")
            self.cb_run = ttk.Combobox(top, state="readonly", width=26); self.cb_run.pack(side="left", padx=4)
            ttk.Label(top, text="Paciente:").pack(side="left")
            self.cb_cpat = ttk.Combobox(top, state="readonly", width=14); self.cb_cpat.pack(side="left", padx=4)
            ttk.Label(top, text="Agregação:").pack(side="left")
            self.cb_cagg = ttk.Combobox(top, state="readonly", width=30, values=[AGG_LABEL[a] for a in AGGS])
            self.cb_cagg.set(AGG_LABEL["mean"]); self.cb_cagg.pack(side="left", padx=4)
            ttk.Button(top, text="Classificar", command=self.classify).pack(side="left", padx=6)
            ttk.Button(top, text="Sobreposição Grad-CAM x tumor (teste)", command=self.cam_overlap_test).pack(side="left")
            ttk.Button(top, text="Avaliar no conjunto de teste", command=self.evaluate_test).pack(side="left", padx=6)
            self.lbl_cls = ttk.Label(tab, text="", font=("TkDefaultFont", 11, "bold")); self.lbl_cls.pack(fill="x")
            self.cls_z = tk.Scale(tab, from_=0, to=1, orient="horizontal", label="Corte (índice dentro da faixa tumoral)",
                                  command=lambda v: self.cls_draw())
            self.cls_z.pack(fill="x", padx=10)
            self.cls_fig, self.cls_ax, self.cls_cv = make_canvas(tab, (10, 5), 1, 2)

        def refresh_runs(self):
            runs = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(self.cfg.model_dir, "*.pt")))
            self.cb_run["values"] = runs
            if runs and not self.cb_run.get():
                self.cb_run.set(runs[0])

        def classify(self):
            # Classifica o paciente escolhido, agrega os cortes e prepara o Grad-CAM.
            if self.bank is None or not self.cb_run.get():
                return messagebox.showinfo("Aviso", "Carregue o dataset e treine/selecione um modelo.")
            model, ck = load_checkpoint(os.path.join(self.cfg.model_dir, self.cb_run.get() + ".pt"))
            _, phase, mask_mode = ck["spec"]
            dev = get_device()
            pi = self.bank.pids.index(self.cb_cpat.get())
            rows = np.where(self.bank.pid_row == pi)[0]
            agg = AGGS[[AGG_LABEL[a] for a in AGGS].index(self.cb_cagg.get())]
            probs = predict_rows(model, self.bank, rows, phase, mask_mode, dev)
            area = self.bank.area[rows]
            p = aggregate_patient(probs, area, agg)
            thr = ck["thr"][agg]
            pred, real = int(p >= thr), int(self.bank.y_pat[pi])
            self.cls = dict(model=model, rows=rows, probs=probs, phase=phase, mask_mode=mask_mode, dev=dev)
            self.lbl_cls.config(text="P(pCR) = %.3f | limiar = %.2f | Predição: pCR %s | Real: pCR %s  -> %s"
                                % (p, thr, "PRESENTE" if pred else "ausente", "PRESENTE" if real else "ausente",
                                   "ACERTO" if pred == real else "ERRO"))
            # corte exibido por padrão conforme a estratégia
            k = {"central": len(rows) // 2, "maxarea": int(np.argmax(area))}.get(agg, int(np.argmax(probs)))
            self.cls_z.config(to=len(rows) - 1); self.cls_z.set(k); self.cls_draw()

        def cls_draw(self):
            # Mostra o corte (com contorno do tumor) e o mapa Grad-CAM sobreposto.
            if not self.cls:
                return
            c, k = self.cls, int(self.cls_z.get())
            row = c["rows"][k]
            x4 = self.bank.X[row:row + 1].to(c["dev"]).float()
            x = make_input(x4, c["phase"], c["mask_mode"])
            cam, prob = grad_cam(c["model"], x)
            n_img = len(PHASES[c["phase"]])
            show = 1 if n_img == 3 else 0                          # fase exibida: precoce (ou a única)
            img = x4[0, PHASES[c["phase"]][show]].cpu().numpy()
            mask = x4[0, 3].cpu().numpy() > 0.5
            r, f = self.vis_rot, self.vis_flip
            a1, a2 = self.cls_ax[0]; a1.clear(); a2.clear()
            a1.imshow(to_display(img, r, f), cmap="gray", vmin=0, vmax=1)
            a2.imshow(to_display(img, r, f), cmap="gray", vmin=0, vmax=1)
            a2.imshow(to_display(cam[0], r, f), cmap="jet", alpha=0.45, vmin=0, vmax=1)
            if mask.any():
                for a in (a1, a2):
                    a.contour(to_display(mask.astype(float), r, f), levels=[0.5], colors="lime", linewidths=1.2)
            a1.set_title("Corte %d (z=%d) - contorno do tumor" % (k, self.bank.z_row[row]))
            a2.set_title("Grad-CAM | P(pCR) do corte = %.2f | %.0f%% da ativação dentro do tumor"
                         % (prob[0], 100 * cam_in_tumor(cam[0], mask)), fontsize=9)
            for a in (a1, a2):
                a.axis("off")
            self.cls_fig.tight_layout(); self.cls_cv.draw_idle()

        def evaluate_test(self):
            # Avalia o modelo salvo nos pacientes de teste do dataset carregado. Para o teste
            # surpresa: carregue a pasta/CSV novos (sem coluna 'test' = todos viram teste) e clique aqui.
            if self.bank is None or not self.cb_run.get():
                return messagebox.showinfo("Aviso", "Carregue o dataset e selecione um modelo.")
            model, ck = load_checkpoint(os.path.join(self.cfg.model_dir, self.cb_run.get() + ".pt"))
            if ck["cfg"]["img_size"] != self.cfg.img_size:
                return messagebox.showerror("Erro", "O modelo foi treinado com imagens de %d px; recarregue o "
                                            "dataset com esse tamanho." % ck["cfg"]["img_size"])
            texto, prev = evaluate_saved(model, ck, self.bank)
            erros = [d["pid"] for d in prev if d["pCR_real"] >= 0 and d["pred_media"] != d["pCR_real"]]
            win = tk.Toplevel(self); win.title("Avaliação - " + self.cb_run.get())
            txt = scrolledtext.ScrolledText(win, width=95, height=20, font=("Courier", 10))
            txt.pack(fill="both", expand=True)
            txt.insert("end", texto + "\n\nErros (agregação média): " + (", ".join(erros) or "-"))

        def cam_overlap_test(self):
            # Quantifica, em todos os cortes de teste, quanto do Grad-CAM cai sobre o tumor.
            if self.bank is None or not self.cb_run.get():
                return
            model, ck = load_checkpoint(os.path.join(self.cfg.model_dir, self.cb_run.get() + ".pt"))
            _, phase, mask_mode = ck["spec"]
            dev = get_device()
            rows = np.where((self.bank.split_row == 1) & (self.bank.area > 0))[0]
            fr, base = [], []
            for b in range(0, len(rows), 16):
                rb = rows[b:b + 16]
                x4 = self.bank.X[torch.from_numpy(rb)].to(dev).float()
                cam, _ = grad_cam(model, make_input(x4, phase, mask_mode))
                for i, r in enumerate(rb):
                    m = x4[i, 3].cpu().numpy() > 0.5
                    fr.append(cam_in_tumor(cam[i], m)); base.append(m.mean())
            messagebox.showinfo("Grad-CAM x tumor",
                                "%s\nCortes de teste com tumor: %d\nEnergia média do Grad-CAM dentro do tumor: %.1f%%\n"
                                "Fração média da imagem ocupada pelo tumor (referência): %.1f%%"
                                % (self.cb_run.get(), len(fr), 100 * np.nanmean(fr), 100 * np.mean(base)))


# ------------------------------------------------------------------------------
# 13. MODOS SEM INTERFACE: --train-all e --selftest
# ------------------------------------------------------------------------------
def headless_train(args):
    # Treina os 12 experimentos (3 modelos x 4 entradas) sem GUI.
    cfg = Config()
    cfg.data_dir, cfg.csv_path, cfg.img_size = args.data_dir, args.csv, args.img_size
    if args.epochs:
        cfg.epochs = args.epochs
    # hiperparâmetros opcionais (se não forem passados, ficam os valores padrão da classe Config)
    for arg, attr in (("lr", "lr"), ("weight_decay", "weight_decay"), ("batch_size", "batch_size"),
                      ("patience", "patience"), ("slice_margin", "slice_margin"), ("loss", "loss_mode"),
                      ("criterio", "criterio"), ("seed", "seed")):
        if getattr(args, arg) is not None:
            setattr(cfg, attr, getattr(args, arg))
    cfg.ft_layer4, cfg.tag = args.ft_layer4, args.tag
    cfg.csv_path = find_csv(cfg)
    if not cfg.csv_path:
        sys.exit("CSV não encontrado em '%s'. Use --csv." % cfg.data_dir)
    meta = read_meta(cfg.csv_path)
    bank = build_bank(meta, index_dataset(cfg.data_dir, list(meta["pid"])), cfg)
    describe_balance(bank, cfg)
    if args.aug_train:
        cfg.aug_train = min(max(args.aug_train, 1), 10)
    if args.aug_val:
        cfg.aug_val = min(max(args.aug_val, 1), 10)
    models_ = [m for m in args.models.split(",") if m in MODELS] if args.models else list(MODELS)
    phases_ = [p for p in args.phases.split(",") if p in PHASES] if args.phases else list(PHASES)
    specs = [(m, p, args.mask) for m in models_ for p in phases_]
    run_experiments(specs, cfg, bank)


def headless_evaluate(args):
    # --avaliar: avalia um modelo salvo num dataset/CSV (ex.: conjunto-teste surpresa), sem treinar.
    model, ck = load_checkpoint(args.avaliar)
    cfg = Config()
    cfg.data_dir, cfg.csv_path = args.data_dir, args.csv
    cfg.img_size = ck["cfg"]["img_size"]               # mesmo tamanho usado no treino do modelo
    cfg.csv_path = find_csv(cfg)
    if not cfg.csv_path:
        sys.exit("CSV não encontrado em '%s'. Use --csv." % cfg.data_dir)
    meta = read_meta(cfg.csv_path)
    bank = build_bank(meta, index_dataset(cfg.data_dir, list(meta["pid"])), cfg)
    texto, prev = evaluate_saved(model, ck, bank)
    print("\nModelo: %s\n%s" % (os.path.basename(args.avaliar), texto))
    os.makedirs(cfg.result_dir, exist_ok=True)
    saida = os.path.join(cfg.result_dir, "previsoes_" + os.path.basename(args.avaliar)[:-3] + ".csv")
    pd.DataFrame(prev).to_csv(saida, index=False)
    print("Previsões por paciente salvas em:", saida)


def selftest():
    # Gera dados sintéticos minúsculos e exercita todo o pipeline (sem internet).
    import tempfile
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "dados"); os.makedirs(d)
    rng, rows = np.random.RandomState(0), []
    xx, yy, zz = np.meshgrid(np.arange(48), np.arange(48), np.arange(10), indexing="ij")
    for i in range(24):
        pid, y = "ISPY1_%d" % (1000 + i), int(i % 3 == 0)
        blob = np.exp(-(((xx - 24) ** 2 + (yy - 24) ** 2) / (2 * (5 + 3 * y) ** 2) + (zz - 5) ** 2 / 8.0))
        for k, g in enumerate([0.1, 0.6 + 0.3 * y, 0.5]):
            vol = (rng.rand(48, 48, 10) * 0.1 + g * blob) * 1000
            nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)), os.path.join(d, "%s_spy1_acq%d.nii.gz" % (pid, k)))
        nib.save(nib.Nifti1Image((blob > 0.4).astype(np.uint8), np.eye(4)), os.path.join(d, "%s_mask.nii.gz" % pid))
        rows.append(dict(pid=pid, pCR=y, test=0 if i < 14 else (2 if i < 19 else 1), pre=0, post_early=1, post_late=2))
    csv = os.path.join(tmp, "meta.csv"); pd.DataFrame(rows).to_csv(csv, index=False)
    cfg = Config()
    cfg.work_dir, cfg.img_size, cfg.epochs, cfg.batch_size = os.path.join(tmp, "out"), 32, 2, 16
    cfg.aug_train = cfg.aug_val = 2
    cfg.pretrained = False
    meta = read_meta(csv)
    bank = build_bank(meta, index_dataset(d, list(meta["pid"])), cfg)
    describe_balance(bank, cfg)
    specs = [(m, p, mk) for m in MODELS for p, mk in (("all", "none"), ("pre", "channel"))]
    run_experiments(specs, cfg, bank)
    assert len(load_all_results(cfg)) == len(specs), "algum experimento falhou"
    model, ck = load_checkpoint(os.path.join(cfg.model_dir, run_name(specs[0]) + ".pt"))
    x = make_input(bank.X[:2].float().to(get_device()), "all", "none")
    cam, pr = grad_cam(model, x)
    print("Grad-CAM:", cam.shape, "prob:", pr)
    texto, _ = evaluate_saved(model, ck, bank)
    print(texto)
    assert os.path.exists(os.path.join(cfg.result_dir, "historico_experimentos.csv"))
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description="Classificação de pCR em DCE-MRI (trabalho prático)")
    ap.add_argument("--selftest", action="store_true", help="teste rápido com dados sintéticos")
    ap.add_argument("--train-all", action="store_true", help="treina os 12 experimentos sem GUI")
    ap.add_argument("--data-dir", default=Config.data_dir)
    ap.add_argument("--csv", default="", help="padrão: procura na pasta dataset")
    ap.add_argument("--img-size", type=int, default=Config.img_size)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--mask", default="none", choices=MASK_MODES)
    ap.add_argument("--aug-train", type=int, default=0, help="variações por corte no treino (1-10)")
    ap.add_argument("--aug-val", type=int, default=0, help="variações por corte na validação (1-10)")
    ap.add_argument("--models", default="", help="ex.: cnn,resnet50,novel")
    ap.add_argument("--phases", default="", help="ex.: pre,early,late,all")
    ap.add_argument("--avaliar", default="", help="caminho de um modelo .pt para avaliar no CSV/pasta dados")
    # ajuste de hiperparâmetros pelo terminal (padrão = valores da classe Config)
    ap.add_argument("--lr", type=float, default=None, help="learning rate (padrão 1e-3)")
    ap.add_argument("--weight-decay", type=float, default=None, help="regularização L2 (padrão 1e-4)")
    ap.add_argument("--batch-size", type=int, default=None, help="tamanho do lote (padrão 32)")
    ap.add_argument("--patience", type=int, default=None, help="épocas sem melhora até parar (padrão 7)")
    ap.add_argument("--slice-margin", type=int, default=None, help="cortes extras fora do tumor (padrão 1)")
    ap.add_argument("--loss", default=None, choices=["pos_weight", "focal", "none"])
    ap.add_argument("--criterio", default=None, choices=["auc", "loss"], help="escolha da melhor época (padrão auc)")
    ap.add_argument("--seed", type=int, default=None, help="semente aleatória (padrão 42)")
    ap.add_argument("--ft-layer4", action="store_true", help="ResNet50: também ajusta o layer4")
    ap.add_argument("--tag", default="", help="sufixo no nome dos resultados (não sobrescreve rodadas)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.avaliar:
        return headless_evaluate(args)
    if args.train_all:
        return headless_train(args)
    if not GUI_OK:
        sys.exit("Tkinter/Matplotlib indisponíveis: instale-os ou use --train-all.")
    App().mainloop()


if __name__ == "__main__":
    main()
