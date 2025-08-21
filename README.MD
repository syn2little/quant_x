# QUANT X

探索和实现多种量化投资策略，数据抓取与分析、因子挖掘、模型训练、策略优化回测、自动化调仓等，实现自动化投资。

## Milestone

| 序号 | 任务 | 状态 | 特性 |
|------|------|------|------|
| 1    | 数据获取和处理 | ⏳ 迭代中 | 基本行情数据获取和分析，已储备基金、财报等能力 |
| 2    | 因子挖掘 | ⏳ 未开始 | 只跑了alpha158。因子库尚未扩展，正在规划特征筛选方法 |
| 3    | 量化策略和模型 | 🔄 进行中 | 跑通lightGBM和试验版本PPO强化学习 |
| 4    | 交易策略 | 🔄 进行中 | 简单的TopkDropout |
| 5    | 回测框架 | ⏳ 未开始 | qlib内置的回测框架|
| 6    | 接入交易 API | ⏳ 未开始 | 还没开始 |


## INSTALL

1. 克隆仓库：
   ```sh
   git clone git@github.com:syn2little/quant_x.git
   ```
2. 安装依赖：
   ```sh
   pip install -r requirements.txt
   ```
3. 运行strategy下的脚本

## 许可

Copyright (c) [2025] [Weidian Zhan]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
