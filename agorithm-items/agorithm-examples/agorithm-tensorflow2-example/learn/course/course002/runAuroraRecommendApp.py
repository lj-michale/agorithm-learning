# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         AuroraRecommendApp
# Description:  极光大数据基于NLP+CNN文本消息推荐系统
# Author:       LJ.Michale
# Date:         2021/7/11
# -------------------------------------------------------------------------------

from learn.course.course002.log.LoggingUtil import Logger


if __name__ == '__main__':

    logger_path = "E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2" \
                  "-example\\log\\logger.log "
    logger = Logger(__name__, logger_path).Logger

    logger.info("========================== AURORA 文本推荐系统模型训练 Starting =====================================")

    logger.info("========================== AURORA 文本推荐系统模型训练 Ending =======================================")
