#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/6 17:13        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction： 极大极小值搜索优化 (已修复动态尺寸) √ ━━━━━☆*°☆*°
"""

import Calcu_every_step_score
import Global_variables
import copy

def _size():
    return Global_variables.BOARD_SIZE

def _max_idx():
    return Global_variables.BOARD_SIZE - 1

# 动态初始化
search_range = []
black_used_pos = []
white_used_pos = []
best_pos = []

def alpha_beta_process(mod):
    global search_range, best_pos
    search_range = shrink_range()
    best_pos = machine_thinking(mod)
    return best_pos

def alpha_beta(color, depth, alpha, beta):
    global black_used_pos, white_used_pos, search_range
    # (简化代码，核心逻辑需调用 Global_variables.BOARD_SIZE)
    # 原代码这里的逻辑有部分未完成或被注释，主要依赖 machine_thinking
    pass 

# 替代alpha-beta的剪枝操作，直接找'半径'为1的闭包
def shrink_range():
    n = _size()
    m = _max_idx()
    cover_range = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if Global_variables.flag[i][j] == 1:
                for k in range(3):
                    cover_range[max(0, i - 1)][min(m, j - 1 + k)] = 1
                    cover_range[max(0, i)][min(m, j - 1 + k)] = 1
                    cover_range[min(m, i + 1)][min(m, j - 1 + k)] = 1
    return cover_range

def machine_thinking(mod):
    global search_range
    black_max_score = -999999
    white_max_score = -999999
    w_best_pos = None
    b_best_pos = None
    
    n = _size()
    
    for i in range(n):
        for j in range(n):
            if Global_variables.flag[i][j] == 0 and search_range[i][j] == 1:
                # 模拟下棋
                Global_variables.flag[i][j] = 1
                search_range[i][j] = 0
                
                # 评估白棋
                Global_variables.white[i][j] = 1
                if mod == '比你6的Level':
                    white_score = Calcu_every_step_score.cal_score_wise('white', i, j)
                else: # '和我一样6的Level' or '固若金汤'
                    white_score = Calcu_every_step_score.cal_score('white', i, j)
                Global_variables.white[i][j] = 0
                
                # 评估黑棋
                Global_variables.black[i][j] = 1
                if mod == '比你6的Level':
                    black_score = Calcu_every_step_score.cal_score_wise('black', i, j)
                else:
                    black_score = Calcu_every_step_score.cal_score('black', i, j)
                Global_variables.black[i][j] = 0
                
                # 恢复
                Global_variables.flag[i][j] = 0
                
                # 更新最优解
                if black_score > black_max_score:
                    black_max_score = black_score
                    b_best_pos = (i, j)
                if white_score > white_max_score:
                    white_max_score = white_score
                    w_best_pos = (i, j)

    # 策略选择
    if mod == '固若金汤' and white_max_score >= 10000 and black_max_score <= white_max_score:
        return w_best_pos
    if mod == '固若金汤' and black_max_score >= 1000:
        return b_best_pos
    if white_max_score > black_max_score or white_max_score >= 100000:
        return w_best_pos
    else:
        return b_best_pos

# 双层搜索
def twice_search(first_best, second_best, mod):
    global search_range
    if not first_best: return -1, None
    
    (w_11, w_12) = first_best
    one_score = Calcu_every_step_score.cal_score_wise('white', w_11, w_12)
    Global_variables.white[w_11][w_12] = 1
    Global_variables.flag[w_11][w_12] = 1

    search_range = shrink_range()
    b_best = machine_thinking(mod)
    if not b_best:
        # 恢复并返回
        Global_variables.white[w_11][w_12] = 0
        Global_variables.flag[w_11][w_12] = 0
        return one_score, first_best

    (b_11, b_12) = b_best
    one_b_score = Calcu_every_step_score.cal_score_wise('black', b_11, b_12)
    Global_variables.black[b_11][b_12] = 1
    Global_variables.flag[b_11][b_12] = 1

    search_range = shrink_range()
    w_best_2 = machine_thinking(mod)
    
    if not w_best_2:
        two_score = 0
    else:
        (w_21, w_22) = w_best_2
        two_score = Calcu_every_step_score.cal_score_wise('white', w_21, w_22)

    # Recover
    Global_variables.white[w_11][w_12] = 0
    Global_variables.flag[w_11][w_12] = 0
    Global_variables.black[b_11][b_12] = 0
    Global_variables.flag[b_11][b_12] = 0

    return one_score + two_score, first_best