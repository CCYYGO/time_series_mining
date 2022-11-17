import csv
import math
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, collections
import numpy.linalg as LA
import math
import cx_Oracle
import json
import sys
import configparser


plt.rc("font", family="Hiragino Sans GB W3")
plt.rc("font", family="Microsoft Yahei")
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']  # 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

startT = endT = None

# 寻找转折点：根据斜率判断转折点，存储时间点
def findTurningPoints(data,angle=45):
    '''
    :param data: 原始数据
    :param angle: 角度阈值
    :return:
    '''
    points = [] #存储转折点
    THRESHOLD = math.pi * angle / 180 # 斜率的阈值
    for i in range(1,len(data)-1):
        a1,a2 = math.atan(data[i] - data[i-1]),math.atan(data[i+1] - data[i])
        if a1 * a2 >= 0:
            t = abs(a1 - a2)
        else:
            t = abs(a1) + abs(a2)
        if t >= THRESHOLD:
            points.append(i)
            # points[0].append(i)
            # points[1].append(data[i])
    return points

# 左闭右闭区间，[seq_range[0],seq_range[1]]   seq_range是个元组(s,e)
def calculate_error(data,seq_range):
    '''
    :param data: 温度数据
    :param seq_range: 范围值
    :return:
    '''
    start,end = seq_range[0],seq_range[1] + 1
    x = np.arange(start,end)
    y = np.array(data[start:end])
    A = np.ones((len(x),2),float)
    A[:,0] = x
    # 返回回归系数、残差平方和、自变量X的秩、X的奇异值
    (p,residuals,ranks,s) = LA.lstsq(A,y,rcond=None)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    error = math.sqrt(error)
    return error

# 拟合分段:Buttom_up:T是整个时间序列，break_points是转折点的列表
def bottomUpMerge(T,break_points,merge_ths,calculate_error=calculate_error):
    '''
    :param T: 整个时间序列
    :param break_points: 转折点的列表
    :param merge_ths: 合并阈值
    :param calculate_error: 拟合误差
    :return:
    '''
    seg = []
    merge_cost = []
    s = 0
    # 左闭区间，右闭区间    将每个分段的起始位置以元组的形式存放到seg列表中
    for bp in break_points:
        seg.append((s,bp)) # s 表示 这一个分段的起点，bp 表示这一个分段的终点
        s = bp
    seg.append((s,len(T) - 1))

    #计算相邻段的误差       遍历整个seg列表，
    for i in range(0,len(seg) - 1):
        merge_cost.append(calculate_error(T,(seg[i][0],seg[i + 1][1])))

    # 寻找拟合误差最小的分段
    while min(merge_cost) < merge_ths:
        index = merge_cost.index(min(merge_cost))
        #合并当前和下一个区间，合并后将下一个分段区间删掉
        seg[index] = (seg[index][0],seg[index + 1][1])
        del seg[index + 1]

        # 更新前一个区间的拟合误差
        if index > 0:
            merge_cost[index - 1] = calculate_error(T, (seg[index - 1][0], seg[index][1]))

        # 更新下一个区间的拟合误差
        if index < len(seg) - 1:
            merge_cost[index] = calculate_error(T,(seg[index][0],seg[index + 1][1]))
            del merge_cost[index + 1]
        #没有下一个区间，遍历到拟合误差数组的最后一个，即删除最后一个
        else:
            del merge_cost[index]
    return seg

# # 连接Oracle数据库并执行查询语句,返回时间和温度组成的二维列表
# def searchDatabase(conn,cursor,startTime,endTime,sql_table):
#     sql = "select TO_CHAR(NOW_TIME,'YYYY-MM-DD HH24:MI:SS') as NOW_TIME,CONTROLTEMP1_CURRENT_T FROM HISTORY_%s WHERE NOW_TIME BETWEEN TO_DATE('%s','YYYY-MM-DD HH24:MI:SS') AND TO_DATE('%s','YYYY-MM-DD HH24:MI:SS') order by NOW_TIME"% (sql_table,startTime,endTime)
#     # sql = "SELECT NOW_TIME,T1_CURRENT FROM (select TO_CHAR(NOWTIME/(60*60*24)+TO_DATE('1970-01-01 08:00:00','YYYY-MM-DD HH24:MI:SS'),'YYYY-MM-DD HH24:MI:SS') AS NOW_TIME,T1_CURRENT FROM HISTORY_%s ORDER BY NOW_TIME )  WHERE NOW_TIME BETWEEN '%s' AND '%s'"% (sql_table,startTime,endTime)
#     cursor.execute(sql)
#     status_list = cursor.fetchall()
#     # 得到两个单独的列
#     time_list = [status[0] for status in status_list]
#     tmp_list = [status[1] for status in status_list]
#     conn.commit()
#     closeDatabase(conn,cursor)
#     return time_list,tmp_list

# 连接Oracle数据库并执行查询语句,返回时间和温度组成的二维列表
def searchDatabase(conn,cursor,startTime,endTime,sql_table):
    '''
    :param conn: 数据库连接对象
    :param cursor: 游标
    :param startTime: 开始时间
    :param endTime: 结束时间
    :param sql_table: 数据表
    :return:
    '''
    sql_table_list30 = ["1309002","1308080","1309004","1308083","1308092","1309005","1308089","1302021","1302018","1307153","1309006","1309003","1308090","1309007","1309012","1308095","1309011","1308091","1309009","1302025","1302023","1302015","1309013","1309014","1309015","1309016"]
    sql_table_list38 = ["1308090_1","1308083_1","1308066","1310003","0205184","1315144","1316025","0205188","0205187","0205183","1307152","1308050","1315201","1316026","1309008","1308027","1308065","1313004","1308024","1313015","1307151","1308088","0205195","1308054","1308049","0205196","1309001","1307142","1308044","1308096","1307007","1307066","1308091_1","1308093","1310002","1307086","1307052"]
    sql_table_list2 = ["1308098","1302028"]
    sql_table_list5 = ["1302014","1302020","1308078","1302016","1308094"]

    if sql_table =="1302501":
        sql_table = "1309013"
    if sql_table == "1302502":
        sql_table = "1309014"
    if sql_table == "1302503":
        sql_table = "1309015"
    if sql_table == "1302504":
        sql_table = "1309016"

    if sql_table in sql_table_list30:
        sql = "select TO_CHAR(NOW_TIME,'YYYY-MM-DD HH24:MI:SS') as NOW_TIME,CONTROLTEMP1_CURRENT_T FROM HISTORY_%s WHERE NOW_TIME BETWEEN TO_DATE('%s','YYYY-MM-DD HH24:MI:SS') AND TO_DATE('%s','YYYY-MM-DD HH24:MI:SS') order by NOW_TIME"% (sql_table,startTime,endTime)
    elif sql_table in sql_table_list38:
        sql = "SELECT NOW_TIME,T1_CURRENT FROM (select TO_CHAR(NOWTIME/(60*60*24)+TO_DATE('1970-01-01 08:00:00','YYYY-MM-DD HH24:MI:SS'),'YYYY-MM-DD HH24:MI:SS') AS NOW_TIME,T1_CURRENT FROM HISTORY_%s ORDER BY NOW_TIME )  WHERE NOW_TIME BETWEEN '%s' AND '%s'"% (sql_table,startTime,endTime)
    elif sql_table in sql_table_list2:
        sql = "SELECT NOW_TIME,CONTROLTEMP_CURRENT_T FROM (select TO_CHAR(NOW_TIME/(60*60*24)+TO_DATE('1970-01-01 08:00:00','YYYY-MM-DD HH24:MI:SS'),'YYYY-MM-DD HH24:MI:SS') AS NOW_TIME,CONTROLTEMP_CURRENT_T FROM HISTORY_%s ORDER BY NOW_TIME )  WHERE NOW_TIME BETWEEN '%s' AND '%s'"% (sql_table,startTime,endTime)
    elif sql_table in sql_table_list5:
        sql = "SELECT NOW_TIME,CONTROLTEMP_CURRENT_T FROM(SELECT TO_CHAR(NOW_TIME/(60*60*24)+TO_DATE('1970-01-01 08:00:00','YYYY-MM-DD HH24:MI:SS'),'YYYY-MM-DD HH24:MI:SS') AS NOW_TIME,CONTROLTEMP_CURRENT_T FROM(SELECT TO_NUMBER(NOW_TIME) AS NOW_TIME,CONTROLTEMP_CURRENT_T FROM HISTORY_%s) ORDER BY NOW_TIME) WHERE NOW_TIME BETWEEN '%s' AND '%s'"% (sql_table,startTime,endTime)
    cursor.execute(sql)
    status_list = cursor.fetchall()
    # 得到两个单独的列
    time_list = [status[0] for status in status_list]
    tmp_list = [int(float(status[1])) for status in status_list]
    conn.commit()
    closeDatabase(conn,cursor)
    return time_list,tmp_list

# 获取数据库连接
def getConn():
    config = configparser.ConfigParser()
    config.read("./config.ini")
    username = config.get("database","username")
    password = config.get("database", "password")
    url = config.get("database", "url")
    sid = config.get("database", "sid")
    conn = cx_Oracle.connect(username,password,url+"/"+sid)
    return conn

# 查询未处理的工艺
def searchDataBase_ReportWork_Copy():
    conn = getConn()
    cursor = conn.cursor()  # 创建游标
    sql = "SELECT PK,ID,STARTDATE,ENDDATE,MACHINENO FROM TEST_503_REPORTWORK_COPY WHERE PY_INDEX = 0"
    cursor.execute(sql)
    list = cursor.fetchall()
    conn.commit()
    closeDatabase(conn, cursor)
    return list

# 关闭Oracle数据库
def closeDatabase(conn,cursor):
    cursor.close()
    conn.close()

def minNums(startTime, endTime) -> int:     # 分钟为单位的时间差
    '''计算两个时间点之间的分钟数'''
    total_seconds = int((endTime - startTime).total_seconds()
                        ) + startTime.second - endTime.second
    return total_seconds // 60



# def displayDifferentData(status_list,dataset,conn, num, turning_point_ths = 45, merge_ths = 30):
def displayDifferentData(info,status_list,tmp_list,time_list, num, turning_point_ths = 45, merge_ths = 30):
    '''
    :param info: 工艺信息
    :param status_list: 状态列表
    :param tmp_list: 温度列表
    :param time_list: 时间列表
    :param num: 显示数据个数
    :param turning_point_ths:转折点阈值
    :param mearge_ths: 合并阈值
    :return: null
    '''
    col_num = math.ceil(math.sqrt(num))
    row_num = math.ceil(num / col_num)

    config = configparser.ConfigParser()
    config.read("./config.ini")
    path = config.get("picture","path")

    for id, data in enumerate(tmp_list[:num]):
        plt.subplot(row_num, col_num, id + 1)
        turning_points = findTurningPoints(data, turning_point_ths)

        slope_list = []
        for i in range(0, len(turning_points)):
            y = data[turning_points[i]]
            slope_list.append(y)

        # print(slope_list)
        if not slope_list:
            # deleteDataBase(missionID)
            if data is None:
                continue
            else:
                plt.plot(data, c='b', label='原始数据')
                min_no = minNums(datetime.strptime(time_list[0], '%Y-%m-%d %H:%M:%S'),
                              datetime.strptime(time_list[-1], '%Y-%m-%d %H:%M:%S'))
                T = np.mean(data[:])
                temp_list = [time_list[0], time_list[-1], int(T), int(T), min_no, 1,1]
                insertDataBase_TEST_503_IDENTIFY(info, temp_list, path + info[0])
            continue
        else:
            min_temp = min(slope_list)
            max_temp = max(slope_list)
            diff_temp = max_temp - min_temp
            if diff_temp <= 10:
                # deleteDataBase(missionID)
                continue
            else:
                intervals = bottomUpMerge(data, turning_points, merge_ths)

                # 关键转折点
                critical_points = set()
                critical_points_time = []
                for pos, (s, e) in enumerate(intervals):
                    critical_points.add(s)
                    critical_points.add(e)
                    if s not in critical_points_time:
                        critical_points_time.append(s)
                    critical_points_time.append(e)

                # 画图
                # plt.plot(time_list,data, c='b', label='原始数据')
                plt.plot(data, c='b', label='原始数据')
                # 转折点
                x_points = []
                for x in critical_points:
                    x_points.append(time_list[x])
                # plt.scatter([x for x in critical_points], [data[_] for _ in critical_points], c='darkorange', marker='x', label='关键转折点')

                # 关键点
                plt.scatter([x for x in critical_points], list(
                    map(lambda x: data[x], critical_points)), marker='x', label='关键转折点')

                sumTmp = 0# 总的保温时间
                index = 0# 温度段序号
                # 状态段
                for s, e in intervals:
                    # x = np.arange(s, e + 1)
                    # y = np.array(data[s: e + 1])
                    # A = np.ones((len(x), 2), float)
                    # A[:, 0] = x
                    # 返回回归系数、残差平方和、自变量X的秩、X的奇异值
                    # k, b = LA.lstsq(A, y, rcond=None)[0]
                    k = (data[e] - data[s]) / (e - s)
                    b = data[s] - s * k
                    # 状态段
                    plt.plot(range(s, e + 1), [k * _ + b for _ in range(s, e + 1)], c='y', label='状态段')
                    temp = {}
                    if k > 0.02:
                        start1 = datetime.strptime(time_list[s],'%Y-%m-%d %H:%M:%S')
                        end1 = datetime.strptime(time_list[e],'%Y-%m-%d %H:%M:%S')
                        minute1 = minNums(start1,end1)
                        # hours1 = minute1 // 60
                        # min1 = minute1 % 60
                        index += 1
                        str = "升温"
                        #print ("升温：时长为%d小时%d分钟，，从%s到%s，温度从%d至%d"%(hours1,min1,time_list[s], time_list[e], data[s], data[e-1]))
                        temp[str] = "从%s到%s，温度从%d至%d"%(time_list[s], time_list[e], data[s], data[e-1])
                        temp_list = [time_list[s], time_list[e], data[s], data[e-1],minute1,0,index]
                        insertDataBase_TEST_503_IDENTIFY(info,temp_list,path+info[0])
                    elif k < -0.02:
                        start2 = datetime.strptime(time_list[s],'%Y-%m-%d %H:%M:%S')
                        end2 = datetime.strptime(time_list[e],'%Y-%m-%d %H:%M:%S')
                        minute2 = minNums(start2,end2)
                        # hours2 = minute2 // 60
                        # min2 = minute2 % 60
                        index += 1
                        str = "降温"
                        #print("降温：时长为%d小时%d分钟，从%s到%s，温度从%d至%d"%(hours2,min2,time_list[s], time_list[e], data[s], data[e-1]))
                        temp[str] = "从%s到%s，温度从%d至%d"%(time_list[s], time_list[e], data[s], data[e-1])
                        temp_list = [time_list[s], time_list[e], data[s], data[e - 1], minute2, 2, index]
                        insertDataBase_TEST_503_IDENTIFY(info, temp_list,path+info[0])
                    else:
                        start = datetime.strptime(time_list[s],'%Y-%m-%d %H:%M:%S')
                        end = datetime.strptime(time_list[e],'%Y-%m-%d %H:%M:%S')
                        # total_seconds1 = int((end1 - start1).total_seconds()
                        #         ) + start1.second - end1.second
                        # minute1 = total_seconds1 // 60
                        minute = minNums(start,end)
                        # hours = minute // 60
                        # min = minute % 60
                        # sumTmp += minute
                        index += 1
                        str = "保温"
                        #print("第%d段保温：时长为%d小时%d分钟，从%s到%s，温度%d" % (index,hours,min,time_list[s], time_list[e], np.mean(data[s: e])))
                        temp[str] = "从%s到%s，温度%d" % (time_list[s], time_list[e], np.mean(data[s: e]))
                        temp_list = [time_list[s], time_list[e], int(np.mean(data[s: e])), int(np.mean(data[s: e])), minute, 1, index]
                        insertDataBase_TEST_503_IDENTIFY(info, temp_list,path+info[0])
                    status_list.append(temp)
                    plt.text((s + e) / 2, k * (s + e) / 2 + b + 10, str)

                hours = sumTmp // 60
                minute = sumTmp % 60
                #print("总的保温时间为：%d小时%d分钟"% (hours,minute))

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = collections.OrderedDict(zip(labels, handles))

                plt.legend(by_label.values(), by_label.keys())

                plt.xlabel("时间/min \n(%c)" % (chr(ord('a') + id)))
                plt.xticks(rotation=90)
                plt.ylabel("温度/°C")
                plt.title("热处理温度曲线")
    plt.tight_layout()
    # 保存图片
    plt.savefig(r"%s.png"%(path+info[0]),dpi=800,bbox_inches='tight')
    # plt.show()


# 向数据库中插入状态段
def insertDataBase_TEST_503_IDENTIFY(info,data,pic_path):
    '''
    :param info: 工艺信息
    :param data: 解析后的信息
    :param pic_path: 图片路径
    :return: null
    '''
    conn = getConn()
    cursor = conn.cursor()  # 创建游标
    sql = "INSERT INTO TEST_503_IDENTITY(PK,ID,S,E,TS,TE,DURATION,TREND,PRO_INDEX,PIC_PATH) VALUES ('%s','%s','%s','%s','%s','%s','%s','%s','%d','%s')"%(info[0],info[1],data[0],data[1],data[2],data[3],data[4],data[5],data[6],pic_path)
    cursor.execute(sql)
    conn.commit()

# 更新数据库
def updateDataBase_REPORTWORK_COPY(pk):
    conn = getConn()
    cursor = conn.cursor()  # 创建游标
    sql = "UPDATE TEST_503_REPORTWORK_COPY SET PY_INDEX = 1 WHERE PK = %s"%(pk)
    cursor.execute(sql)
    conn.commit()

# 主函数
def main(info):
    # 得到时间，温度数据列表
    # pk = info[0]
    # id = info[1]
    startTime = str(info[2])
    endTime = str(info[3])
    # print(startTime)
    machNum = info[4]
    tmp_list = []
    # 打开数据库连接
    conn = getConn()
    cursor = conn.cursor()  # 创建游标
    time_list, temp_list = searchDatabase(conn, cursor, startTime, endTime, machNum)
    temp_list = np.array([temp for temp in temp_list],dtype=np.float32)
    tmp_list.append(temp_list)
    #print(tmp_list)
    if not time_list:
        pass
    else:
        status_list = []
        displayDifferentData(info,status_list, tmp_list, time_list, 1, 15, 5)

# 双室炉
def main_double(info):
    # 得到时间，温度数据列表
    # pk = info[0]
    # id = info[1]
    startTime = str(info[2])
    endTime = str(info[3])
    # print(startTime)
    machNum = info[4]
    tmp_list = []
    tmp_list1 = []
    # 打开数据库连接
    conn = getConn()
    cursor = conn.cursor()  # 创建游标
    time_list, temp_list, temp_list1 = searchDatabase_double(conn, cursor, startTime, endTime, machNum)
    temp_list = np.array([temp for temp in temp_list])
    temp_list1 = np.array([temp1 for temp1 in temp_list1])
    tmp_list.append(temp_list)
    tmp_list1.append(temp_list1)
    # print(tmp_list)
    status_list = []
    status_list1 = []
    displayDifferentData(info, status_list, tmp_list, time_list, 1, 15, 5)
    displayDifferentData(info, status_list1, tmp_list1, time_list, 1, 15, 5)

def searchDatabase_double(conn,cursor,startTime,endTime,sql_table):
    sql_table_list38 = ["1308090_1", "1308083_1", "1308066", "1310003", "0205184", "1315144", "1316025", "0205188",
                        "0205187", "0205183", "1307152", "1308050", "1315201", "1316026", "1309008", "1308027",
                        "1308065", "1313004", "1308024", "1313015", "1307151", "1308088", "0205195", "1308054",
                        "1308049", "0205196", "1309001", "1307142", "1308044", "1308096", "1307007", "1307066",
                        "1308091_1", "1308093", "1310002", "1307086", "1307052"]
    if sql_table in sql_table_list38:
        sql = "SELECT NOW_TIME,T1_CURRENT FROM (select TO_CHAR(NOWTIME/(60*60*24)+TO_DATE('1970-01-01 08:00:00','YYYY-MM-DD HH24:MI:SS'),'YYYY-MM-DD HH24:MI:SS') AS NOW_TIME,T1_CURRENT FROM HISTORY_%s ORDER BY NOW_TIME )  WHERE NOW_TIME BETWEEN '%s' AND '%s'" % (sql_table, startTime, endTime)
    cursor.execute(sql)
    status_list = cursor.fetchall()
    # 得到两个单独的列
    time_list = [status[0] for status in status_list]
    tmp_list = [int(float(status[1])) for status in status_list]
    tmp_list1 = [int(float(status[2])) for status in status_list]
    conn.commit()
    closeDatabase(conn, cursor)
    return time_list, tmp_list, tmp_list1

if __name__ == '__main__':
    availableNum = ["1309002","1308080","1309004","1308083","1308092","1309005","1308089","1302021","1302018","1307153","1309006","1309003","1308090","1309007","1309012","1308095","1309011","1308091","1309009","1302025","1302023","1302015","1309013","1309014","1309015","1309016","1308090_1","1308083_1","1308066","1310003","0205184","1315144","1316025","0205188","0205187","0205183","1307152","1308050","1315201","1316026","1309008","1308027","1308065","1313004","1308024","1313015","1307151","1308088","0205195","1308054","1308049","0205196","1309001","1307142","1308044","1308096","1307007","1307066","1308091_1","1308093","1310002","1307086","1307052","1308098","1302028","1302014","1302020","1308078","1302016","1308094"]
    Tech_list = searchDataBase_ReportWork_Copy()
    print(Tech_list)
    for info in Tech_list:
        print(info)
        if info[3] is None or info[0] is None or info[2] is None or info[1] is None or info[4] not in availableNum:
            continue
        else:
            if info[3] is not None and info[0] is not None and info[2] is not None and info[1] is not None:
                startTime = str(info[2])
                endTime = str(info[3])
                minute = minNums(datetime.strptime(startTime, '%Y-%m-%d %H:%M:%S'),
                                 datetime.strptime(endTime, '%Y-%m-%d %H:%M:%S'))
                if minute <= 10:
                    updateDataBase_REPORTWORK_COPY(info[0])
                    continue
                else:
                    main(info)
                    updateDataBase_REPORTWORK_COPY(info[0])



