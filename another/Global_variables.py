#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/5 22:58        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction：  全局变量配置 - 已修复动态棋盘大小支持
"""
import re

# ==========================================
# 在这里修改棋盘大小
# 标准五子棋为 15，你可以改为 8 进行测试
BOARD_SIZE = 8
# ==========================================

black = [[0 for a in range(BOARD_SIZE)] for b in range(BOARD_SIZE)]
white = [[0 for a in range(BOARD_SIZE)] for b in range(BOARD_SIZE)]
flag = [[0 for a in range(BOARD_SIZE)] for b in range(BOARD_SIZE)]

# 评分模式 (Regex Patterns)
# 100000 - 连五
pattern_5 = [re.compile(r'11111')]
# 10000 - 活四
pattern_alive_4 = [re.compile(r'011110')]
# 8000 - 冲四
pattern_to_4 = [re.compile(r'11011'), re.compile(r'011112'), re.compile(r'10111'), re.compile(r'201111')]
# 5000 - 双活三
pattern_double_alive_3 = [re.compile(r'0011100'), re.compile(r'2011100')]
# 1000 - 活三眠
pattern_alive_sleep_3 = [re.compile(r'0011102')]
# 200 - 活三
pattern_alive_3 = [re.compile(r'010110')]
# 100 - 双活二
pattern_double_alive_2 = [re.compile(r'001100'), re.compile(r'001102'), re.compile(r'001012')]
# 50 - 眠三
pattern_sleep_3 = [re.compile(r'001112'), re.compile(r'010112'), re.compile(r'011012'), 
                   re.compile(r'10011'), re.compile(r'10101'), re.compile(r'2011102')]
# 10 - 活二眠
pattern_alive_sleep_2 = [re.compile(r'0010100'), re.compile(r'00100100')]
# 5 - 活二
pattern_alive_2 = [re.compile(r'201010'), re.compile(r'2010010'),  re.compile(r'20100102'),  re.compile(r'2010102')]
# 3 - 眠二
pattern_sleep_2 = [re.compile(r'000112'), re.compile(r'001012'), re.compile(r'010012'), 
                   re.compile(r'10001'), re.compile(r'2010102'), re.compile(r'2011002')]
# -5 - 死四/死三/死二
pattern_dead_4 = [re.compile(r'2\d{3}12'), re.compile(r'2\d{2}1\d{2}2')]
pattern_dead_3 = [re.compile(r'2\d{2}12')]
pattern_dead_2 = [re.compile(r'2\d12')]

all_patterns = [pattern_5, pattern_alive_4, pattern_to_4, pattern_double_alive_3, pattern_alive_sleep_3, pattern_alive_3, 
                pattern_double_alive_2, pattern_sleep_3, pattern_alive_sleep_2, pattern_alive_2, pattern_sleep_2, 
                pattern_dead_4, pattern_dead_3, pattern_dead_2]

all_scores = [100000, 10000, 8000, 5000, 1000, 200, 100, 50, 10, 5, 3, -5, -5, -5]

def _gen_board_scores(n):
    # 生成位置分，越靠近中心分数越高
    scores = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            scores[i][j] = min(i, j, n - 1 - i, n - 1 - j)
    return scores

board_scores = _gen_board_scores(BOARD_SIZE)

search_range = []

def prepare(n):
    global BOARD_SIZE, black, white, flag, board_scores
    BOARD_SIZE = n
    black = [[0 for _ in range(n)] for _ in range(n)]
    white = [[0 for _ in range(n)] for _ in range(n)]
    flag = [[0 for _ in range(n)] for _ in range(n)]
    board_scores = _gen_board_scores(n)