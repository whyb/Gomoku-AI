#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/6 16:39        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction：        √ ━━━━━☆*°☆*°
"""
import Global_variables
import re
def cal_score(color, i, j):
	N = Global_variables.BOARD_SIZE
	M = N - 1
	# 加一个初边界处理 修复 bug * 4
	pos_row = '2'
	pos_col = '2'
	bia_right = '2'
	bia_left = '2'
	for ii in range(N):
		if Global_variables.black[i][ii] == 1:
			if color == 'black':
				pos_row += '1'
			else:
				pos_row += '2'
		elif Global_variables.white[i][ii] == 1:
			if color == 'black':
				pos_row += '2'
			else:
				pos_row += '1'
		else:
			pos_row += '0'
	pos_row += '2'
	for ii in range(N):
		if Global_variables.black[ii][j] == 1:
			if color == 'black':
				pos_col += '1'
			else:
				pos_col += '2'
		elif Global_variables.white[ii][j] == 1:
			if color == 'black':
				pos_col += '2'
			else:
				pos_col += '1'
		else:
			pos_col += '0'
	pos_col += '2'
	# 要保存两个斜线上组成的字符串中原下棋点的位置
	bia_left_pos = -1
	bia_right_pos = -1
	bia_left_pos_rev = -1
	bia_right_pos_rev = -1
	# 按列数递增遍历，与按行数递增遍历一样的效果
	for ii in range(max(0, i-j), min(i+(M - j) + 1, N)):
		if bia_right_pos == -1 and ii == i:
			bia_right_pos = ii - max(0, i-j)  # 修复bug*2
			# print 'bia_right_pos: ' + str(bia_right_pos)
			bia_right_pos_rev = min(i+(M - j), N) - 1 - bia_right_pos
		if Global_variables.black[ii][ii + j - i] == 1:
			if color == 'black':
				bia_right += '1'
			else:
				bia_right += '2'
		elif Global_variables.white[ii][ii + j - i] == 1:
			if color == 'black':
				bia_right += '2'
			else:
				bia_right += '1'
		else:
			bia_right += '0'
	# 加一个末边界处理 修复 bug * 4
	bia_right += '2'
	# print 'bia_right:' + str(len(bia_right))
	# 记得如果想用行数遍历的话，要把j当做第一个下标，修复bug => too long
	for ii in range(max(0, i-(M - j)), min(i + j + 1, N)):  # 修复 bug *2 ↑
		if bia_left_pos == -1 and ii == i:
			bia_left_pos = ii - max(0, i-(M - j))  # 修复bug *2
			# print 'bia_left_pos: ' + str(bia_left_pos)
			bia_left_pos_rev = min(i + j + 1, N) - 1 - bia_left_pos
		if Global_variables.black[ii][j - (ii - i)] == 1:
			if color == 'black':
				bia_left += '1'
			else:
				bia_left += '2'
		elif Global_variables.white[ii][j - (ii - i)] == 1:
			if color == 'black':
				bia_left += '2'
			else:
				bia_left += '1'
		else:
			bia_left += '0'
	bia_left += '2'
	# print 'bia_left:' + str(len(bia_left))
	search_flag = False
	# pos_col = pos_col[max(0, j-6):min(j+6, N)]
	rev_col = pos_col[::-1]
	# pos_row = pos_row[max(0, i-6):min(i+6, N)]
	rev_row = pos_row[::-1]
	rev_bia_left = bia_left[::-1]
	rev_bia_right = bia_right[::-1]
	score = 0
	# 加了末边界处理后要统一将下标都加1 修复 bug * 6
	i += 1
	j += 1
	bia_left_pos += 1
	bia_right_pos += 1
	bia_left_pos_rev += 1
	bia_right_pos_rev += 1
	for patterns in Global_variables.all_patterns:
		for p in patterns:
			result = re.search(p, pos_col)
			# 修复bug * 16 不等式的等于号需要加！
			if not result or not ((result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, rev_col)
			if not result or not ((result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, pos_row)
			if not result or not ((result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, rev_row)
			if result:
				search_range = result.span()
				if (search_range[0] <= i <= search_range[1]) or (search_range[0] <= j <= search_range[1]):
				# just to test
					score = Global_variables.all_scores[Global_variables.all_patterns.index(patterns)]
					# print result.span()
					search_flag = True
					break
			# 处理两种斜线上的情况
			pos = -1
			if not result or not ((result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, bia_left)
				if result:  # bug修复 * 4
					pos = bia_left_pos
					# print 'test:' + str(result.span())
					# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:  # 修复bug * 3 ，pos要固定
				result = re.search(p, rev_bia_left)
				if result:
					pos = bia_left_pos_rev
					# print 'test:' + str(result.span())
					# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:
				result = re.search(p, bia_right)
				if result:
					pos = bia_right_pos
					# print 'test:' + str(result.span())
					# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:
				result = re.search(p, rev_bia_right)
				if result:
					pos = bia_right_pos_rev
					# print 'test:' + str(result.span())
					# print 'pos:' + str(pos)
			if result:
				search_range = result.span()
				if search_range[0] <= pos <= search_range[1]:
					score = Global_variables.all_scores[Global_variables.all_patterns.index(patterns)]
					# print result.span()
					search_flag = True
					break
		if search_flag:
			break
	# 修复bug，i-1
	# print 'origin: ' + str(Global_variables.board_scores[i - 1][j - 1]) + ' point: ' + str('%d,%d' % (i-1, j-1))
	# print 'score: ' + str(score)
	return score + Global_variables.board_scores[i-1][j-1]

def cal_score_wise(color, i, j):
	N = Global_variables.BOARD_SIZE
	M = N - 1
	scores = []
	first_pattern = -1
	second_pattern = -1
	p_index = -1
	# 加一个初边界处理 修复 bug * 4
	pos_row = '2'
	pos_col = '2'
	bia_right = '2'
	bia_left = '2'
	for ii in range(N):
		if Global_variables.black[i][ii] == 1:
			if color == 'black':
				pos_row += '1'
			else:
				pos_row += '2'
		elif Global_variables.white[i][ii] == 1:
			if color == 'black':
				pos_row += '2'
			else:
				pos_row += '1'
		else:
			pos_row += '0'
	pos_row += '2'
	for ii in range(N):
		if Global_variables.black[ii][j] == 1:
			if color == 'black':
				pos_col += '1'
			else:
				pos_col += '2'
		elif Global_variables.white[ii][j] == 1:
			if color == 'black':
				pos_col += '2'
			else:
				pos_col += '1'
		else:
			pos_col += '0'
	pos_col += '2'
	# 要保存两个斜线上组成的字符串中原下棋点的位置
	bia_left_pos = -1
	bia_right_pos = -1
	bia_left_pos_rev = -1
	bia_right_pos_rev = -1
	# 按列数递增遍历，与按行数递增遍历一样的效果
	for ii in range(max(0, i-j), min(i+(M - j) + 1, N)):
		if bia_right_pos == -1 and ii == i:
			bia_right_pos = ii - max(0, i-j)  # 修复bug*2
			# print 'bia_right_pos: ' + str(bia_right_pos)
			bia_right_pos_rev = min(i+(M - j), N) - 1 - bia_right_pos
		if Global_variables.black[ii][ii + j - i] == 1:
			if color == 'black':
				bia_right += '1'
			else:
				bia_right += '2'
		elif Global_variables.white[ii][ii + j - i] == 1:
			if color == 'black':
				bia_right += '2'
			else:
				bia_right += '1'
		else:
			bia_right += '0'
	# 加一个末边界处理 修复 bug * 4
	bia_right += '2'
	# print 'bia_right:' + bia_right
	# 记得如果想用行数遍历的话，要把j当做第一个下标，修复bug => too long
	for ii in range(max(0, i-(M - j)), min(i + j + 1, N)):  # 修复 bug *2 ↑
		if bia_left_pos == -1 and ii == i:
			bia_left_pos = ii - max(0, i-(M - j))  # 修复bug *2
			# print 'bia_left_pos: ' + str(bia_left_pos)
			bia_left_pos_rev = min(i + j + 1, N) - 1 - bia_left_pos
		if Global_variables.black[ii][j - (ii - i)] == 1:
			if color == 'black':
				bia_left += '1'
			else:
				bia_left += '2'
		elif Global_variables.white[ii][j - (ii - i)] == 1:
			if color == 'black':
				bia_left += '2'
			else:
				bia_left += '1'
		else:
			bia_left += '0'
	bia_left += '2'
	# print 'bia_left:' + bia_left
	search_flag = False
	# pos_col = pos_col[max(0, j-6):min(j+6, N)]
	rev_col = pos_col[::-1]
	# pos_row = pos_row[max(0, i-6):min(i+6, N)]
	rev_row = pos_row[::-1]
	rev_bia_left = bia_left[::-1]
	rev_bia_right = bia_right[::-1]
	score = 0
	# 加了末边界处理后要统一将下标都加1 修复 bug * 6
	i += 1
	j += 1
	bia_left_pos += 1
	bia_right_pos += 1
	bia_left_pos_rev += 1
	bia_right_pos_rev += 1
	for patterns in Global_variables.all_patterns:
		for p in patterns:
			result = re.search(p, pos_col)
			# 修复bug * 16 不等式的等于号需要加！
			if not result or not ((result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, rev_col)
			if not result or not ((result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, pos_row)
			if not result or not ((result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, rev_row)
			if result:
				search_range = result.span()
				if (search_range[0] <= i <= search_range[1]) or (search_range[0] <= j <= search_range[1]):
				# just to test
				# bug * 4修复 fatal
					score = Global_variables.all_scores[Global_variables.all_patterns.index(patterns)]
					first_pattern = Global_variables.all_patterns.index(patterns)
					scores.append(score)
					p_index = patterns.index(p)
					# print result.span()
					search_flag = True
					break
			# 处理两种斜线上的情况
			pos = -1
			if not result or not ((result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, bia_left)
				if result:  # bug修复 * 4
					pos = bia_left_pos
					# print 'test:' + str(result.span())
					# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:  # 修复bug * 3 ，pos要固定
				result = re.search(p, rev_bia_left)
				if result:
					pos = bia_left_pos_rev
					# print 'test:' + str(result.span())
					# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:
				result = re.search(p, bia_right)
				if result:
					pos = bia_right_pos
					# print 'test:' + str(result.span())
					# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:
				result = re.search(p, rev_bia_right)
				if result:
					pos = bia_right_pos_rev
					# print 'test:' + str(result.span())
					# print 'pos:' + str(pos)
			if result:
				search_range = result.span()
				if search_range[0] <= pos <= search_range[1]:
					score = Global_variables.all_scores[Global_variables.all_patterns.index(patterns)]
					first_pattern = Global_variables.all_patterns.index(patterns)
					scores.append(score)
					p_index = patterns.index(p)
					# print result.span()
					search_flag = True
					break
		if search_flag:
			break
	# bug修复，缩进
	search_flag = False
	for patterns in Global_variables.all_patterns[first_pattern:]:
		for p in patterns:
			if patterns == Global_variables.all_patterns[first_pattern]:
				if patterns.index(p) <= p_index:
					continue
			result = re.search(p, pos_col)
			# 修复bug * 16 不等式的等于号需要加！
			if not result or not (
				(result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, rev_col)
			if not result or not (
				(result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, pos_row)
			if not result or not (
				(result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, rev_row)
			if result:
				search_range = result.span()
				if (search_range[0] <= i <= search_range[1]) or (search_range[0] <= j <= search_range[1]):
					# just to test
					score = Global_variables.all_scores[Global_variables.all_patterns.index(patterns)]
					# 修复bug,score列表
					first_pattern = Global_variables.all_patterns.index(patterns)
					scores.append(score)
					p_index = patterns.index(p)
					# print result.span()
					search_flag = True
					break
			# 处理两种斜线上的情况
			pos = -1
			if not result or not (
				(result.span()[0] <= i <= result.span()[1]) or (result.span()[0] <= j <= result.span()[1])):
				result = re.search(p, bia_left)
				if result:  # bug修复 * 4
					pos = bia_left_pos
				# print 'test:' + str(result.span())
				# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:  # 修复bug * 3 ，pos要固定
				result = re.search(p, rev_bia_left)
				if result:
					pos = bia_left_pos_rev
				# print 'test:' + str(result.span())
				# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:
				result = re.search(p, bia_right)
				if result:
					pos = bia_right_pos
				# print 'test:' + str(result.span())
				# print 'pos:' + str(pos)
			if not result or not result.span()[0] <= pos <= result.span()[1]:
				result = re.search(p, rev_bia_right)
				if result:
					pos = bia_right_pos_rev
				# print 'test:' + str(result.span())
				# print 'pos:' + str(pos)
			if result:
				search_range = result.span()
				# print search_range
				if search_range[0] <= pos <= search_range[1]:
					score = Global_variables.all_scores[Global_variables.all_patterns.index(patterns)]
					first_pattern = Global_variables.all_patterns.index(patterns)
					scores.append(score)
					p_index = patterns.index(p)
					# print result.span()
					search_flag = True
					break
		if search_flag:
			break
	# 修复bug，i-1
	# print 'origin: ' + str(Global_variables.board_scores[i - 1][j - 1]) + ' point: ' + str('%d,%d' % (i-1, j-1))
	# print 'score: ' + str(score)
	# print scores
	if len(scores) == 2:
		score = sum(scores)
	elif len(scores) != 0:
		score = max(scores)
	# print sum(scores) + Global_variables.board_scores[i-1][j-1]
	return score + Global_variables.board_scores[i-1][j-1]
