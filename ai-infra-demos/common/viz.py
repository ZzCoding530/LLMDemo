# -*- coding: utf-8 -*-
"""绘图辅助：统一使用 Agg 后端，图片落在 outputs/ 目录。

约定：图中文字一律用英文，避免服务器缺中文字体导致豆腐块；
代码注释/print 输出用中文（讲解用）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs")


def save(fig, name, out_dir=None, dpi=110):
    """保存图片并返回绝对路径；打印位置便于讲解时快速定位。"""
    os.makedirs(out_dir or OUT_DIR, exist_ok=True)
    path = os.path.join(out_dir or OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[fig] saved -> {path}")
    return path


def close(fig):
    plt.close(fig)
