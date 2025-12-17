#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/5 11:12        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction： 五子棋人机/机机对战 (已修复动态尺寸) √ ━━━━━☆*°☆*°
"""
from graphics import *
import Global_variables
import Alpha_beta_optimize
import Calcu_every_step_score

N = Global_variables.BOARD_SIZE
p = [[0 for a in range(N)] for b in range(N)]
q = [[0 for a in range(N)] for b in range(N)]
win = ''

def Create_Board():
    global win
    N = Global_variables.BOARD_SIZE
    for i in range(N):
        for j in range(N):
            p[i][j] = Point(i*30+30, j*30+30)
            p[i][j].draw(win)
    for r in range(N):
        Line(p[r][0], p[r][N-1]).draw(win)
        Line(p[0][r], p[N-1][r]).draw(win)
    
    center_idx = N // 2
    center = Circle(p[center_idx][center_idx], 3)
    center.draw(win)
    center.setFill('black')

def human_vs_machine(cnt, mod):
    global win
    N = Global_variables.BOARD_SIZE
    win_w = win.getWidth()
    win_h = win.getHeight()
    
    if cnt is None: return
    
    if cnt % 2 == 0:
        human_flag = False
        while not human_flag:
            try:
                p1 = win.getMouse()
                x1 = p1.getX()
                y1 = p1.getY()
            except:
                return -1
            for i in range(N):
                for j in range(N):
                    sqrdis = ((x1 - p[i][j].getX()) ** 2 + (y1 - p[i][j].getY()) ** 2)
                    if sqrdis <= 200 and Global_variables.flag[i][j] == 0:
                        Global_variables.black[i][j] = 1
                        q[i][j] = Circle(p[i][j], 10)
                        q[i][j].draw(win)
                        q[i][j].setFill('black')
                        human_flag = True
                        break
                if human_flag: break
    else:
        machine_pos = Alpha_beta_optimize.alpha_beta_process(mod)
        if not machine_pos:
            Text(Point(win_w/2, win_h - 100), '机器对战已结束...').draw(win)
            return -1
        i = machine_pos[0]
        j = machine_pos[1]
        Global_variables.white[i][j] = 1
        q[i][j] = Circle(p[i][j], 10)
        q[i][j].draw(win)
        q[i][j].setFill('white')
    cnt += 1
    Global_variables.flag[i][j] = 1
    return cnt

def machine_vs_machine(cnt, mod):
    global win
    win_w = win.getWidth()
    win_h = win.getHeight()
    
    machine_pos = Alpha_beta_optimize.alpha_beta_process(mod)
    if not machine_pos:
        Text(Point(win_w/2, win_h - 100), '机器对战已结束...').draw(win)
        return -1
    
    i = machine_pos[0]
    j = machine_pos[1]
    
    if cnt % 2 == 0:
        Global_variables.black[i][j] = 1
        q[i][j] = Circle(p[i][j], 10)
        q[i][j].draw(win)
        q[i][j].setFill('black')
    else:
        Global_variables.white[i][j] = 1
        q[i][j] = Circle(p[i][j], 10)
        q[i][j].draw(win)
        q[i][j].setFill('white')
        
    cnt += 1
    Global_variables.flag[i][j] = 1
    return cnt

def human_with_machine_vs_machine(cnt, mod):
    global win
    N = Global_variables.BOARD_SIZE
    win_w = win.getWidth()
    win_h = win.getHeight()

    if cnt % 2 == 0:
        human_flag = False
        machine_pos = Alpha_beta_optimize.alpha_beta_process(mod)
        if not machine_pos:
            Text(Point(win_w/2, win_h - 100), '对战已结束...').draw(win)
            return -1
            
        ii = machine_pos[0]
        jj = machine_pos[1]
        
        # 提示点
        temp_hint = Circle(p[ii][jj], 3)
        temp_hint.draw(win)
        temp_hint.setFill('black')
        
        # 文字提示
        # Text(Point(win_w/2, win_h - 50), 'Black point: current system suggestion').draw(win)
        
        while not human_flag:
            try:
                p1 = win.getMouse()
                x1 = p1.getX()
                y1 = p1.getY()
            except:
                return -1
            for i in range(N):
                for j in range(N):
                    sqrdis = ((x1 - p[i][j].getX()) ** 2 + (y1 - p[i][j].getY()) ** 2)
                    if sqrdis <= 200 and Global_variables.flag[i][j] == 0:
                        temp_hint.undraw() # 清除提示
                        Global_variables.black[i][j] = 1
                        q[i][j] = Circle(p[i][j], 10)
                        q[i][j].draw(win)
                        q[i][j].setFill('black')
                        human_flag = True
                        break
                if human_flag: break
    else:
        machine_pos = Alpha_beta_optimize.alpha_beta_process(mod)
        if not machine_pos:
            Text(Point(win_w/2, win_h - 100), '对战已结束...').draw(win)
            return -1
        i = machine_pos[0]
        j = machine_pos[1]
        Global_variables.white[i][j] = 1
        q[i][j] = Circle(p[i][j], 10)
        q[i][j].draw(win)
        q[i][j].setFill('white')
    cnt += 1
    Global_variables.flag[i][j] = 1
    return cnt

def Check():
    N = Global_variables.BOARD_SIZE
    # 动态检查胜利条件
    # 横向
    for i in range(N):
        for j in range(N-4):
            if Global_variables.black[i][j:j+5] == [1, 1, 1, 1, 1]: return 'black'
            elif Global_variables.white[i][j:j+5] == [1, 1, 1, 1, 1]: return 'white'
    # 纵向
    for i in range(N):
        for j in range(N-4):
            if Global_variables.black[j][i] and Global_variables.black[j+1][i] and Global_variables.black[j+2][i] and Global_variables.black[j+3][i] and Global_variables.black[j+4][i]:
                return 'black'
            elif Global_variables.white[j][i] and Global_variables.white[j+1][i] and Global_variables.white[j+2][i] and Global_variables.white[j+3][i] and Global_variables.white[j+4][i]:
                return 'white'
    # 正斜
    for i in range(N-4):
        for j in range(N-4):
            if Global_variables.black[i][j] and Global_variables.black[i+1][j+1] and Global_variables.black[i+2][j+2] and Global_variables.black[i+3][j+3] and Global_variables.black[i+4][j+4]:
                return 'black'
            elif Global_variables.white[i][j] and Global_variables.white[i+1][j+1] and Global_variables.white[i+2][j+2] and Global_variables.white[i+3][j+3] and Global_variables.white[i+4][j+4]:
                return 'white'
    # 反斜
    for i in range(N-4):
        for j in range(4, N):
            if Global_variables.black[i][j] and Global_variables.black[i+1][j-1] and Global_variables.black[i+2][j-2] and Global_variables.black[i+3][j-3] and Global_variables.black[i+4][j-4]:
                return 'black'
            elif Global_variables.white[i][j] and Global_variables.white[i+1][j-1] and Global_variables.white[i+2][j-2] and Global_variables.white[i+3][j-3] and Global_variables.white[i+4][j-4]:
                return 'white'

def Run_chess(mod, option):
    global win
    N = Global_variables.BOARD_SIZE
    
    # 动态计算窗口大小
    win_w = (N + 1) * 30 + 30
    win_h = win_w + 120
    win = GraphWin(mod + option, win_w, win_h)
    
    Create_Board()
    cnt = 0
    
    # 机器或开局的第一步棋：下在天元(中心)
    # 原代码写死为[7][7]，在8x8棋盘是右下角，现改为动态计算
    center_idx = N // 2
    Global_variables.white[center_idx][center_idx] = 1
    q[center_idx][center_idx] = Circle(p[center_idx][center_idx], 10)
    q[center_idx][center_idx].draw(win)
    q[center_idx][center_idx].setFill('white')
    Global_variables.flag[center_idx][center_idx] = 1
    
    while 1:
        if option == 'MM':
            cnt = machine_vs_machine(cnt, mod)
        elif option == 'HM':
            cnt = human_vs_machine(cnt, mod)
        elif option == 'HMM':
            cnt = human_with_machine_vs_machine(cnt, mod)
        
        winner = Check()
        if cnt == -1: break
        
        if winner == 'black':
            Text(Point(win_w/2, win_h - 60), 'Black wins').draw(win)
            break
        if winner == 'white':
            Text(Point(win_w/2, win_h - 60), 'White wins').draw(win)
            break
    try:
        win.getMouse()
    except:
        pass