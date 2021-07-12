# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         runPredictApp
# Description:
# Author:       orange
# Date:         2021/7/11
# -------------------------------------------------------------------------------

from learn.course.course002.log.LoggingUtil import Logger

if __name__ == '__main__':

    logger_path = "E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2" \
                  "-example\\log\\logger.log "
    logger = Logger(__name__, logger_path).Logger

    logger.info("========================== AURORA 文本推荐系统开始预测 Starting =====================================")