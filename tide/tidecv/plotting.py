
from collections import defaultdict, OrderedDict
import os
import shutil

import cv2
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from .errors.main_errors import *
from .datasets import get_tide_path
import random


def print_table(rows:list, title:str=None):
	# Get all rows to have the same number of columns
	max_cols = max([len(row) for row in rows])
	for row in rows:
		while len(row) < max_cols:
			row.append('')

	# Compute the text width of each column
	col_widths = [max([len(rows[i][col_idx]) for i in range(len(rows))]) for col_idx in range(len(rows[0]))]

	divider = '--' + ('---'.join(['-' * w for w in col_widths])) + '-'
	thick_divider = divider.replace('-', '=')

	if title:
		left_pad = (len(divider) - len(title)) // 2
		print(('{:>%ds}' % (left_pad + len(title))).format(title))

	print(thick_divider)
	for row in rows:
		# Print each row while padding to each column's text width
		print('  ' + '   '.join([('{:>%ds}' % col_widths[col_idx]).format(row[col_idx]) for col_idx in range(len(row))]) + '  ')
		if row == rows[0]: print(divider)
	print(thick_divider)




class Plotter():
	""" Sets up a seaborn environment and holds the functions for plotting our figures. """
	

	def __init__(self, quality:float=1):
		# Set mpl DPI in case we want to output to the screen / notebook
		mpl.rcParams['figure.dpi'] = 150
		
		# Seaborn color palette
		sns.set_palette('muted', 10)
		current_palette = sns.color_palette()
		
		# Seaborn style
		sns.set(style="whitegrid")

		self.colors_main = OrderedDict({
			ClassError        .short_name: current_palette[9],
			BoxError          .short_name: current_palette[8],
			OtherError        .short_name: current_palette[2],
			DuplicateError    .short_name: current_palette[6],
			BackgroundError   .short_name: current_palette[4],
			MissedError       .short_name: current_palette[3],
		})

		self.colors_special = OrderedDict({
			FalsePositiveError.short_name: current_palette[0],
			FalseNegativeError.short_name: current_palette[1],
		})

		self.tide_path = get_tide_path()

		# For the purposes of comparing across models, we fix the scales on our bar plots.
		# Feel free to change these after initializing if you want to change the scale.
		self.MAX_MAIN_DELTA_AP    = 10
		self.MAX_SPECIAL_DELTA_AP = 25

		self.quality = quality
	
	def _prepare_tmp_dir(self):
		tmp_dir = os.path.join(self.tide_path, '_tmp')

		if not os.path.exists(tmp_dir):
			os.makedirs(tmp_dir)

		for _f in os.listdir(tmp_dir):
			os.remove(os.path.join(tmp_dir, _f))

		return tmp_dir
		
	
	def make_summary_plot(self, out_dir:str, errors:dict, model_name:str, rec_type:str, hbar_names:bool=False):
		"""Make a summary plot of the errors for a model, and save it to the figs folder.
		
		:param out_dir:    The output directory for the summary image. MUST EXIST.
		:param errors:     Dictionary of both main and special errors.
		:param model_name: Name of the model for which to generate the plot.
		:param rec_type:   Recognition type, either TIDE.BOX or TIDE.MASK
		:param hbar_names: Whether or not to include labels for the horizontal bars.	
		"""

		tmp_dir = self._prepare_tmp_dir()
		
		high_dpi = int(500*self.quality)
		low_dpi  = int(300*self.quality)

		# get the data frame
		error_dfs = {errtype: pd.DataFrame(data={
			'Error Type': list(errors[errtype][model_name].keys()),
			'Delta mAP': list(errors[errtype][model_name].values()),
		}) for errtype in ['main', 'special']}

		# pie plot for error type breakdown
		error_types = list(errors['main'][model_name].keys()) + list(errors['special'][model_name].keys())
		error_val = list(errors['main'][model_name].values()) + list(errors['special'][model_name].values())
		error_out = []


		for i in range(len(error_val)):
			name = error_types[i]
			# error_out.append(f'{name}')  # 标签没有数值，只有名字
			val = ('%.2f' % error_val[i])
			error_out.append(f'{name}:{val}') # 标签 添加数值


		error_sum = sum([e for e in errors['main'][model_name].values()])
		error_sizes = [e / error_sum for e in errors['main'][model_name].values()] + [0, 0]
		fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=high_dpi)

		patches, outer_text, inner_text = ax.pie(error_sizes, colors=self.colors_main.values(),wedgeprops={'linewidth': 1},
												 pctdistance =0.7,autopct='%1.1f%%', startangle=90)

		# autopct：用于指定每个扇形块标注文本的样式，比如：定义百分数精度,# pctdistance：设置圆内文本距圆心距离
		# textprops：图中文本属性,startangle：起始角度（逆时针顺序绘图),labeldistance设置标签文本距圆心位置，表示多少倍半径
		# wedges, texts = ax.pie(error_sizes,colors=self.colors_main.values())

		for text in outer_text + inner_text:
			text.set_text('')
		for i in range(len(self.colors_main)):
			# inner_text[i].set_text(list(self.colors_main.keys())[i])
			inner_text[i].set_text(error_out[i])
			# 字体大小改进地方
			if 0.025 <= error_sizes[i] < 0.04:
				inner_text[i].set_fontsize(10)
			elif error_sizes[i] < 0.025:
				inner_text[i].set_fontsize(7)
			else:
				inner_text[i].set_fontsize(10)
			inner_text[i].set_fontweight('bold')

		# wedges, texts = ax.pie(error_sizes)
		# kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"), zorder=0, va="center")
		# diy = [1,-1,1,-1,1,-1,1,-1,]
		# offset =40
		# for i,p in enumerate(patches):
		# 	# 根据matplotlib.patches.Wedge对象的theta1和theta2参数计算饼块均分点的角度
		# 	ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
		# 	# 根据角度的弧度计算 饼块均分点的坐标（引导线的起点）
		# 	y = np.sin(np.deg2rad(ang))
		# 	x = np.cos(np.deg2rad(ang))
		# 	ang_abs = abs((p.theta2 - p.theta1))
		#
		# 	print("{} ang:{}".format(i,ang_abs))
		# 	if ang_abs > 15 or ang_abs == 0 :
		# 		continue
		#
		# 	# 演示引导线起点位置
		# 	# plt.plot(x, y, "or")
		# 	# offset = random.randint(20,30)
		# 	# print('offset:{}\n'.format(offset))
		# 	# 根据x的值即角度所在象限确定引导线的对齐方式
		# 	horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
		# 	dir = int(np.sign(x))
		#
		# 	# 设置引导线的连接方式
		# 	# connectionstyle = 'angle,angleA=0,angleB={}'.format(ang)
		# 	# kw["arrowprops"].update({"connectionstyle": connectionstyle})
		# 	# 绘制注释标签和引导线
		# 	ax.annotate(error_out[i],xy=(x, y),xytext=(dir*offset, diy[i]*offset), textcoords='offset points',
		# 				horizontalalignment=horizontalalignment,
		# 		# xycoords='data',textcoords='data',
		# 		arrowprops=dict(facecolor='black'
		# 						,connectionstyle = "arc3,rad=0.2"))
		#

		rec_type = 'tide.box'
		ax.axis('equal')
		# ax.legend(patches, error_types, loc='upper right')
		model_name = model_name.split('.bbox')[0]
		# title_name = model_name.split('_')[-1]
		title_name = model_name
		plt.title(title_name,
				  fontdict={'fontsize': 30,
							# 'fontweight': 'bold',
							# 'fontstyle':'italic'
							})
		pie_path = os.path.join(tmp_dir, '{}_{}_pie.png'.format(model_name, rec_type))
		plt.savefig(pie_path, bbox_inches='tight', dpi=low_dpi)
		# plt.show()
		plt.close()

		# horizontal bar plot for main error types
		fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=high_dpi)
		sns.barplot(data=error_dfs['main'], x='Delta mAP', y='Error Type', ax=ax,
					palette=self.colors_main.values())
		ax.set_xlim(0, self.MAX_MAIN_DELTA_AP)
		ax.set_xlabel('Delta mAP', fontsize=10)
		ax.set_ylabel('Error Type', fontsize=10)
		if not hbar_names:
			ax.set_yticklabels([''] * 6)
		plt.setp(ax.get_xticklabels(), fontsize=15)
		plt.setp(ax.get_yticklabels(), fontsize=15)
		for i, val in enumerate(error_val[: 6]):
			plt.text(x=val + 0.5, y=i + 0.1, s=('%.2f' % val), ha='center')
		ax.grid(axis="x", linestyle="--")
		sns.despine(left=True, bottom=True, right=True)
		hbar_path = os.path.join(tmp_dir, '{}_{}_hbar.png'.format(model_name, rec_type))
		plt.savefig(hbar_path, bbox_inches='tight', dpi=low_dpi)
		plt.close()

		# vertical bar plot for special error types
		fig, ax = plt.subplots(1, 1, figsize = (2, 3), dpi=high_dpi)
		sns.barplot(data=error_dfs['special'], x='Error Type', y='Delta mAP', ax=ax,
					palette=self.colors_special.values())
		ax.set_ylim(0, self.MAX_SPECIAL_DELTA_AP)
		ax.set_xlabel('Error Type')
		ax.set_ylabel('Delta mAP')
		ax.set_xticklabels(['FP', 'FN'])
		plt.setp(ax.get_xticklabels(), fontsize=15)
		plt.setp(ax.get_yticklabels(), fontsize=15)
		for i, val in enumerate(error_val[6:]):
			plt.text(x=i, y=val+0.3, s=('%.2f' % val), ha='center')
		ax.grid(axis="y", linestyle="--")
		sns.despine(left=True, bottom=True, right=True)
		vbar_path = os.path.join(tmp_dir, '{}_{}_vbar.png'.format(model_name, rec_type))
		plt.savefig(vbar_path, bbox_inches='tight', dpi=low_dpi)
		plt.close()

		# get each subplot image
		pie_im  = cv2.imread(pie_path)
		hbar_im = cv2.imread(hbar_path)
		vbar_im = cv2.imread(vbar_path)

		# pad the hbar image vertically
		hbar_im = np.concatenate([np.zeros((vbar_im.shape[0] - hbar_im.shape[0], hbar_im.shape[1], 3)) + 255, hbar_im],
								axis=0)
		summary_im = np.concatenate([hbar_im, vbar_im], axis=1)

		# pad summary_im
		if summary_im.shape[1]<pie_im.shape[1]:
			lpad, rpad = int(np.ceil((pie_im.shape[1] - summary_im.shape[1])/2)), \
					int(np.floor((pie_im.shape[1] - summary_im.shape[1])/2))
			summary_im = np.concatenate([np.zeros((summary_im.shape[0], lpad, 3)) + 255,
									summary_im,
									np.zeros((summary_im.shape[0], rpad, 3)) + 255], axis=1)

		# pad pie_im
		else:
			lpad, rpad = int(np.ceil((summary_im.shape[1] - pie_im.shape[1])/2)), \
					int(np.floor((summary_im.shape[1] - pie_im.shape[1])/2))
			pie_im = np.concatenate([np.zeros((pie_im.shape[0], lpad, 3)) + 255,
									pie_im,
									np.zeros((pie_im.shape[0], rpad, 3)) + 255], axis=1)


		summary_im = np.concatenate([pie_im, summary_im], axis=0)

		if out_dir is None:
			fig = plt.figure()
			ax = plt.axes([0,0,1,1])
			ax.set_axis_off()
			fig.add_axes(ax)
			ax.imshow((summary_im / 255)[:, :, (2, 1, 0)])
			plt.show()
			plt.close()
		else:
			cv2.imwrite(os.path.join(out_dir, '{}_{}_summary.png'.format(model_name, rec_type)), summary_im)

