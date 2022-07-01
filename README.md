# AI2-code
机器翻译：
1. [Seq2Seq 机器翻译, 全程手写代码](https://www.bilibili.com/video/BV1hf4y1u7ez?p=2&vd_source=14b5aa0f75150f92a422f3d1987176ce)（视频）
对应源码：[https://github.com/shouxieai/seq2seq_translation](https://github.com/shouxieai/seq2seq_translation)
(无用，可忽略)
<hr>

编码策略：
1. [BPE 算法原理及使用指南【深入浅出】](https://blog.csdn.net/a1097304791/article/details/122068153?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165667511916781432926139%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165667511916781432926139&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-122068153-null-null.142%5Ev30%5Epc_rank_34,185%5Ev2%5Etag_show&utm_term=BPE%20&spm=1018.2226.3001.4187)
<hr>

解码策略：
1. [浅谈文本生成或者文本翻译解码策略](https://blog.csdn.net/HUSTHY/article/details/115028696?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165665860016780366548699%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165665860016780366548699&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~pc_rank_34-6-115028696-null-null.142%5Ev30%5Epc_rank_34,185%5Ev2%5Etag_show&utm_term=Top-k%20Sampling&spm=1018.2226.3001.4187)（附代码）
（包含贪心搜索greedy search，beam_search集束搜索，随机sampling，Top-K Sampling和Top-p (nucleus) sampling）
2. [Beam Search快速理解及代码解析](https://blog.csdn.net/qq_41466892/article/details/121119550?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165665864516782184658848%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165665864516782184658848&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-121119550-null-null.142%5Ev30%5Epc_rank_34,185%5Ev2%5Etag_show&utm_term=Top-k%20Sampling%E4%BB%A3%E7%A0%81&spm=1018.2226.3001.4187)

<hr>

pytorch部分函数讲解：
1. [torch.argmax函数说明](https://blog.csdn.net/weixin_42494287/article/details/92797061?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165670047316782246421066%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165670047316782246421066&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-1-92797061-null-null.142%5Ev30%5Epc_rank_34,185%5Ev2%5Etag_show&utm_term=torch.argmax&spm=1018.2226.3001.4187)
2. [torch.multinomial()理解](https://blog.csdn.net/monchin/article/details/79787621?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165668782816780366580765%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165668782816780366580765&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-79787621-null-null.142%5Ev30%5Epc_rank_34,185%5Ev2%5Etag_show&utm_term=torch.multinomial&spm=1018.2226.3001.4187)
