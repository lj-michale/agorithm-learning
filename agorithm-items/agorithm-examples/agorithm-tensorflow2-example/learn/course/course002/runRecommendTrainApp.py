# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         AuroraRecommendApp
# Description:  极光大数据基于NLP+CNN+BERT文本消息推荐系统
#               初步功能：1. 通过文案自动推送相关兴趣人群(文案找人)
#                        2. 通过人自动寻找感兴趣的文案(人找文案)
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
