#!/usr/bin/env python
# -*- coding:utf-8 -*-

from xlrd import *
from openpyxl import Workbook

def demoTest(provience, num):
    wb = Workbook()
    # 获取当前活跃的worksheet,默认就是第一个worksheet
    ws1 = wb.active
    ws1.title = "宏站弱覆盖"  # 修改sheet1的名字


    f = open("E:\\weakcoverage\October.txt", "r", encoding='utf-8')  # 在路径以utf-8的格式打开txt文件
    lines = f.readlines()
    line_1 = lines[0].strip().split('|')
    ws1.append(line_1)
    i=0
    for line in lines[1:len(lines) - 1]:
        p = line.strip().split('|')
        if (p[0] == provience):
            i = i + 1
            ws1.append(p)

    print(i)
    wb.save('E:\\weakcoverage\\sichuan\\xinwenjian_%s_October.xlsx' % provience)

demoTest('AH', 1)
demoTest('BJ', 1)
demoTest('CQ', 1)
demoTest('FJ', 1)
demoTest('GD', 1)
demoTest('GS', 1)
demoTest('GX', 1)
demoTest('GZ', 1)
demoTest('HA', 1)
demoTest('HB', 1)
demoTest('HE', 1)
demoTest('HI', 1)
demoTest('HL', 1)
demoTest('HN', 1)
demoTest('JL', 1)
demoTest('JS', 1)
demoTest('JX', 1)
demoTest('LN', 1)
demoTest('NM', 1)
demoTest('NX', 1)
demoTest('QH', 1)
demoTest('SC', 1)
demoTest('SD', 1)
demoTest('SH', 1)
demoTest('SN', 1)
demoTest('SX', 1)
demoTest('TJ', 1)
demoTest('YN', 1)
demoTest('ZJ', 1)
