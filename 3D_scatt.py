from pyecharts import Map
from pyecharts import Bar
# encoding: utf-8
import time
import sys

from pyecharts import Bar, Scatter3D
from pyecharts import Page




# # bar
# attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
# v1 = [5, 20, 36, 10, 75, 90]
# v2 = [10, 25, 8, 60, 20, 80]
# bar = Bar("柱状图数据堆叠示例")
# bar.add("商家A", attr, v1, is_stack=True)
# bar.add("商家B", attr, v2, is_stack=True)
# page.add(bar)         # step 2


# value = [95.1, 23.2, 43.3, 66.4, 88.5]
# attr= ["China", "Canada", "Brazil", "Russia", "United States"]
# map = Map("World Map Example", width=1200, height=600, title_color="#2E2E2E", title_text_size=24, title_top=20, title_pos="center")
# # map = Map("World Map Example")
# map.add("", attr, value, maptype="world", is_visualmap=True, visual_text_color='#000')
# map.render(path=r'D:\project\a.html')


# scatter3D
import random
import numpy as np
import pandas as pd
import csv
import os

page = Page()         # step 1
particaption_path = r'D:\project\2019\massive_MIMO\use_data\use_particaption.csv'
particaption_data = pd.read_csv(particaption_path, encoding='gbk')



test_path = r'D:\project\2019\massive_MIMO\use_data\mdt_x_y'
files = filter(lambda x: x.endswith('.csv'), os.listdir(test_path))
files = map(lambda x: test_path + '/' + x, files)
files = sorted(files)
for f in files:
    name = f.split('.csv')[0]
    print(name)
    name = name.split('/')[-1]
    CGI_name = name.split('_')[0]
    # particaption_path = r'D:\project\2019\massive_MIMO\use_data\use_particaption.csv'
    # particaption_data = pd.read_csv(particaption_path, encoding='gbk')

    bs = particaption_data.loc[particaption_data.CGI == CGI_name]
    t = bs.iloc[0, 9]
    # t = bs['天线挂高(m)']

    bs_list = [[0, 0, t]]
    data = pd.read_csv(f)
    item_data = data[['new_x','new_y','Altitude','azimuth_list']]
    item_list = item_data.values.tolist()
    # print()
    # data = [[random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)] for _ in range(80)]
    range_color = ['#313695']
    scatter3D = Scatter3D("3D 散点图", width=1200, height=600)
    # scatter3D.add("", item_list, is_visualmap=True, visual_range_color=range_color)
    scatter3D.add("", item_list, is_visualmap=True, visual_type="color", )
    scatter3D.add("", bs_list, is_visualmap=False, effect_brushtype='fill', symbol_size=10, effect_scale=10, symbol="diamond",)


    page.add(scatter3D)  # step 2

    page.render('a.html')        # step 3
    print()




