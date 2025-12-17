#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/7 12:22        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction：五子棋主入口界面  √ ━━━━━☆*°☆*°
"""
import multiprocessing
from tkinter import messagebox
from tkinter import *
from Chess import *
from PP_chess import *
win_num = 0
if __name__ == '__main__':
	top = Tk()
	top.title("一盘五子棋")
	top.iconbitmap('me.ico')
	def play(mod, option):
		global win_num
		confirm = messagebox.askokcancel('确认窗口', '确认开始' + mod + '模式 ' + option +' 对战吗?')
		if confirm:
			if mod == '单机':
				p = multiprocessing.Process(target=pp_main, args=(mod, option))
				p.start()
			else:
				p = multiprocessing.Process(target=Run_chess, args=(mod, option))
				p.start()
		else:
			return
	Button(top, text="[单机] 人人对战", fg="blue", width=28, command=lambda: play('单机', '')).pack()
	Button(top, text="比你6的Level   人机", fg="blue",  width=28, command=lambda: play('比你6的Level', 'HM')).pack()
	Button(top, text="比你6的Level  机器", fg="blue", width=28, command=lambda: play('比你6的Level', 'MM')).pack()
	Button(top, text="比你6的Level  辅助", fg="blue", width=28, command=lambda: play('比你6的Level', 'HMM')).pack()
	Button(top, text="和我一样6的Level  人机", fg="blue",  width=28, command=lambda: play('和我一样6的Level', 'HM')).pack()
	Button(top, text="和我一样6的Level  机器", fg="blue", width=28, command=lambda: play('和我一样6的Level', 'MM')).pack()
	Button(top, text="和我一样6的Level  辅助", fg="blue", width=28, command=lambda: play('和我一样6的Level', 'HMM')).pack()
	Button(top, text="固若金汤  人机", fg="blue", width=28, command=lambda: play('固若金汤', 'HM')).pack()
	Button(top, text="固若金汤  机器", fg="blue", width=28, command=lambda: play('固若金汤', 'MM')).pack()
	# Level 3在继续开发中
	# Button(top, text="Difficult 人机", fg="blue",  width=28, command=lambda: play('高级', 'HM')).pack()
	# Button(top, text="Difficult 机器", fg="blue", width=28, command=lambda: play('高级', 'MM')).pack()
	# Button(top, text="Difficult 辅助", fg="blue", width=28, command=lambda: play('高级', 'HMM')).pack()
	# Button(top, text="退出", fg="blue", bd=2, width=28, command=exit).pack()

	top.mainloop()
