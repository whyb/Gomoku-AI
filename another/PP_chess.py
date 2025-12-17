#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/5 11:12        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction： 人人对战模式 (已修复动态尺寸) √ ━━━━━☆*°☆*°
"""
from graphics import *
import Global_variables

win = ''
# 动态获取尺寸
N = Global_variables.BOARD_SIZE
p = [[0 for a in range(N)] for b in range(N)]
q = [[0 for a in range(N)] for b in range(N)]

def Create_Board():
    global win
    N = Global_variables.BOARD_SIZE
    # 绘制点(隐形锚点)
    for i in range(N):
        for j in range(N):
            p[i][j] = Point(i*30+30, j*30+30)
            p[i][j].draw(win)
    # 绘制线
    for r in range(N):
        Line(p[r][0], p[r][N-1]).draw(win) # 竖线
        Line(p[0][r], p[N-1][r]).draw(win) # 横线
    
    # 绘制天元(中心点)
    center_idx = N // 2
    center = Circle(p[center_idx][center_idx], 3)
    center.draw(win)
    center.setFill('black')

def Click(cnt):
    N = Global_variables.BOARD_SIZE
    human_flag = False
    while not human_flag:
        try:
            p1 = win.getMouse()
            x1 = p1.getX()
            y1 = p1.getY()
        except GraphicsError:
            return cnt # 窗口关闭保护

        for i in range(N):
            for j in range(N):
                # 判定点击有效半径 (200的平方根约14像素)
                sqrdis = ((x1 - p[i][j].getX()) ** 2 + (y1 - p[i][j].getY()) ** 2)
                if sqrdis <= 200 and Global_variables.flag[i][j] == 0:
                    if cnt % 2 == 0:
                        Global_variables.black[i][j] = 1
                        q[i][j] = Circle(p[i][j], 10)
                        q[i][j].draw(win)
                        q[i][j].setFill('black')
                        human_flag = True
                    else:
                        Global_variables.white[i][j] = 1
                        q[i][j] = Circle(p[i][j], 10)
                        q[i][j].draw(win)
                        q[i][j].setFill('white')
                        human_flag = True
                    cnt += 1
                    Global_variables.flag[i][j] = 1
                    break
        if win.isClosed(): return -1
    return cnt


def Check():
    N = Global_variables.BOARD_SIZE
    # 横向检查
    for i in range(N):
        for j in range(N - 4):
            if Global_variables.black[i][j:j+5] == [1, 1, 1, 1, 1]:
                return 'black'
            elif Global_variables.white[i][j:j+5] == [1, 1, 1, 1, 1]:
                return 'white'
    # 纵向检查
    for i in range(N):
        for j in range(N - 4):
            if Global_variables.black[j][i] and Global_variables.black[j+1][i] and Global_variables.black[j+2][i] and Global_variables.black[j+3][i] and Global_variables.black[j+4][i]:
                return 'black'
            elif Global_variables.white[j][i] and Global_variables.white[j+1][i] and Global_variables.white[j+2][i] and Global_variables.white[j+3][i] and Global_variables.white[j+4][i]:
                return 'white'
    # 正斜向 (\)
    for i in range(N - 4):
        for j in range(N - 4):
            if Global_variables.black[i][j] and Global_variables.black[i+1][j+1] and Global_variables.black[i+2][j+2] and Global_variables.black[i+3][j+3] and Global_variables.black[i+4][j+4]:
                return 'black'
            elif Global_variables.white[i][j] and Global_variables.white[i+1][j+1] and Global_variables.white[i+2][j+2] and Global_variables.white[i+3][j+3] and Global_variables.white[i+4][j+4]:
                return 'white'
    # 反斜向 (/)
    for i in range(N - 4):
        for j in range(4, N):
            if Global_variables.black[i][j] and Global_variables.black[i+1][j-1] and Global_variables.black[i+2][j-2] and Global_variables.black[i+3][j-3] and Global_variables.black[i+4][j-4]:
                return 'black'
            elif Global_variables.white[i][j] and Global_variables.white[i+1][j-1] and Global_variables.white[i+2][j-2] and Global_variables.white[i+3][j-3] and Global_variables.white[i+4][j-4]:
                return 'white'


def pp_main(mod, option):
    global win
    N = Global_variables.BOARD_SIZE
    # 动态计算窗口大小: (格子数+1)*30 + 边距
    win_w = (N + 1) * 30 + 30
    win_h = win_w + 120
    
    win = GraphWin(mod, win_w, win_h)
    Create_Board()
    cnt = 0
    while 1:
        cnt = Click(cnt)
        if cnt == -1: break
        
        winner = Check()
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