
本项目使用的是MovieLens 1M 数据集，包含6000个用户在近4000部电影上的1亿条评论。
数据集分为三个文件：用户数据users.dat，电影数据movies.dat和评分数据ratings.dat。

用户数据
分别有用户ID、性别、年龄、职业ID和邮编等字段。
数据中的格式：UserID::Gender::Age::Occupation::Zip-code
Gender is denoted by a "M" for male and "F" for female
Age is chosen from the following ranges:
1: "Under 18"
18: "18-24"
25: "25-34"
35: "35-44"
45: "45-49"
50: "50-55"
56: "56+"
Occupation is chosen from the following choices:
0: "other" or not specified
1: "academic/educator"
2: "artist"
3: "clerical/admin"
4: "college/grad student"
5: "customer service"
6: "doctor/health care"
7: "executive/managerial"
8: "farmer"
9: "homemaker"
10: "K-12 student"
11: "lawyer"
12: "programmer"
13: "retired"
14: "sales/marketing"
15: "scientist"
16: "self-employed"
17: "technician/engineer"
18: "tradesman/craftsman"
19: "unemployed"
20: "writer"
可以看出UserID、Gender、Age和Occupation都是类别字段，其中邮编字段是我们不使用的。

电影数据
分别有电影ID、电影名和电影风格等字段。
数据中的格式：MovieID::Title::Genres
Titles are identical to titles provided by the IMDB (including year of release)
Genres are pipe-separated and are selected from the following genres:
Action
Adventure
Animation
Children's
Comedy
Crime
Documentary
Drama
Fantasy
Film-Noir
Horror
Musical
Mystery
Romance
Sci-Fi
Thriller
War
Western
MovieID是类别字段，Title是文本，Genres也是类别字段

评分数据
分别有用户ID、电影ID、评分和时间戳等字段。
数据中的格式：UserID::MovieID::Rating::Timestamp
UserIDs range between 1 and 6040
MovieIDs range between 1 and 3952
Ratings are made on a 5-star scale (whole-star ratings only)
Timestamp is represented in seconds since the epoch as returned by time(2)
Each user has at least 20 ratings
评分字段Rating就是我们要学习的targets，时间戳字段我们不使用。


数据预处理
UserID、Occupation和MovieID不用变。
Gender字段：需要将‘F’和‘M’转换成0和1。
Age字段：要转成7个连续数字0~6。
Genres字段：是分类字段，要转成数字。首先将Genres中的类别转成字符串到数字的字典，然后再将每个电影的Genres字段转成数字列表，因为有些电影是多个Genres的组合。
Title字段：处理方式跟Genres字段一样，首先创建文本到数字的字典，然后将Title中的描述转成数字的列表。另外Title中的年份也需要去掉。
Genres和Title字段需要将长度统一，这样在神经网络中方便处理。空白部分用‘< PAD >’对应的数字填充。


















































