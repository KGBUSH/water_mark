# -*- coding: utf-8 -*-


import time
import os

######################### 普通for循环 #########################
# def long_time_task():
#     print('当前进程: {}'.format(os.getpid()))
#     time.sleep(2)
#     print("结果: {}".format(8 ** 20))
#
#
# if __name__ == "__main__":
#     print('当前母进程: {}'.format(os.getpid()))
#     start = time.time()
#     for i in range(2):
#         long_time_task()
#
#     end = time.time()
#     print("用时{}秒".format((end - start)))


######################### 多进程计算 #########################
# # encoding:utf-8
# from multiprocessing import Process
# import os, time, random
#
#
# # 线程启动后实际执行的代码块
# def r1(process_name):
#     for i in range(5):
#         print process_name, os.getpid()  # 打印出当前进程的id
#         time.sleep(random.random())
#
#
# def r2(process_name):
#     for i in range(5):
#         print process_name, os.getpid()  # 打印出当前进程的id
#         time.sleep(random.random())
#
#
# if __name__ == "__main__":
#     print "main process run..."
#     p1 = Process(target=r1, args=('process_name1',))  # target:指定进程执行的函数，args:该函数的参数，需要使用tuple
#     p2 = Process(target=r2, args=('process_name2',))
#
#     p1.start()  # 通过调用start方法启动进程，跟线程差不多。
#     p2.start()  # 但run方法在哪呢？待会说。。。
#     p1.join()  # join方法也很有意思，寻思了一下午，终于理解了。待会演示。
#     p2.join()
#     print "main process runned all lines..."


######################### apply_async #########################
from multiprocessing import Pool, cpu_count
import os
import time


def long_time_task(i):
    print('子进程: {} - 任务{}'.format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))


if __name__=='__main__':
    print("CPU内核数:{}".format(cpu_count()))
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('等待所有子进程完成。')
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))