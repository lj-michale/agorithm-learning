# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         UnitTest
# Description:
# Author:       orange
# Date:         2021/7/11
# -------------------------------------------------------------------------------
from turtle import Screen

from selenium import webdriver
import unittest



# class testcals(unittest.UnitTest001):
#     def setUp(self):
#         self.driver = webdriver.Firefox()
#         self.base = Screen(self.driver)  # 实例化自定义类commlib.baselib
#
#     def login(self):
#         url_login = "http://www.baidu.com"
#         self.driver.get(url_login)
#
#     def test_01_run_mail(self):
#         try:
#             self.login()
#             logger.info(self.img)
#         except Exception as msg:
#             logger.error("异常原因 [ %s ]" % msg)
#             logger.error(self.img)
#             raise
#
#     def test_02_case(self):
#         u'''【test_case】'''
#         logger.error("首页error 日志")
#         logger.debug("订单页debug 日志")
#         logger.info("活动页info 日志")
#         logger.critical("支付critical 日志")
#
#     def tearDown(self):
#         self.driver.quit()
#
#
# if __name__ == '__main__':
#     unittest.main()
