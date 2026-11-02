"""读取实际实验 JSON，生成中文 PDF 和可修改的 Markdown 报告。

仅生成报告时需要 reportlab；算法及实验运行不需要第三方库。
"""
import argparse
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Flowable, KeepTogether)

ROOT = Path(__file__).resolve().parent
REPO_URL = 'https://github.com/Kallen-Dis/rengongzhinengzuoye/tree/experiment2'
WIDTH = A4[0] - 96
INK = colors.HexColor('#1F2937')
BLUE = colors.HexColor('#254E70')
LIGHT = colors.HexColor('#F1F5F9')


class MazeFigure(Flowable):
    """直接绘制矢量网格和最短路径，不依赖截图。"""
    def __init__(self, path):
        Flowable.__init__(self)
        self.path = path
        self.width, self.height = WIDTH, 180

    def draw(self):
        c, size, left, bottom = self.canv, 29, 34, 14
        grid = ['S..X.', '.X...', '.X.X.', '...XE', 'XX...']
        for r in range(5):
            for col in range(5):
                x, y = left + col * size, bottom + (4 - r) * size
                fill = colors.HexColor('#CBD5E1') if grid[r][col] == 'X' else colors.white
                if (r, col) in self.path:
                    fill = colors.HexColor('#DBEAFE')
                c.setFillColor(fill)
                c.setStrokeColor(colors.HexColor('#94A3B8'))
                c.rect(x, y, size, size, fill=1)
                c.setFillColor(INK)
                c.setFont('CN', 10)
                c.drawCentredString(x + size / 2, y + 10, grid[r][col])
        c.setStrokeColor(BLUE)
        c.setLineWidth(2)
        for a, b in zip(self.path, self.path[1:]):
            c.line(left + (a[1] + .5) * size, bottom + (4.5 - a[0]) * size,
                   left + (b[1] + .5) * size, bottom + (4.5 - b[0]) * size)
        c.setFillColor(INK)
        c.setFont('CN', 9)
        for i in range(5):
            c.drawCentredString(left + (i + .5) * size, bottom + 5 * size + 8, str(i))
            c.drawRightString(left - 9, bottom + (4.5 - i) * size - 3, str(i))
        c.setFont('CN', 10)
        for i, text in enumerate(['行列坐标从 0 开始', '蓝色连线为返回路径', 'X 为不可通行障碍', '8 个坐标对应 7 次移动']):
            c.drawString(220, 130 - i * 27, text)


class TreeFigure(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = WIDTH, 145

    def draw(self):
        c = self.canv
        levels = [[('MAX 5', 250)], [('MIN 5', 130), ('MIN 2', 370)],
                  [('MAX 5', 70), ('MAX 9', 190), ('MAX 9', 310), ('MAX 2', 430)],
                  [('3', 40), ('5', 100), ('6', 160), ('9', 220),
                   ('6', 280), ('9', 340), ('1', 400), ('2', 460)]]
        ys = [128, 92, 56, 20]
        c.setStrokeColor(colors.HexColor('#94A3B8'))
        for level in range(3):
            for i, (_, x) in enumerate(levels[level]):
                for _, xx in levels[level + 1][2 * i:2 * i + 2]:
                    c.line(x, ys[level] - 7, xx, ys[level + 1] + 12)
        for level, nodes in enumerate(levels):
            c.setFont('CN', 10)
            c.setFillColor(INK)
            for text, x in nodes:
                c.drawCentredString(x, ys[level], text)


class NodeBars(Flowable):
    def __init__(self, values):
        Flowable.__init__(self)
        self.values = values
        self.width, self.height = WIDTH, 139

    def draw(self):
        c = self.canv
        c.setFont('CN', 9)
        for tick in (0, 20, 40, 60):
            x = 140 + tick * 4
            c.setStrokeColor(colors.HexColor('#D9D9D9'))
            c.line(x, 25, x, 132)
            c.setFillColor(INK)
            c.drawCentredString(x, 11, str(tick))
        for i, (label, value) in enumerate(self.values):
            y = 118 - i * 22
            c.setFillColor(INK)
            c.drawRightString(132, y + 1, label)
            c.setFillColor(BLUE if i else colors.HexColor('#64748B'))
            c.rect(140, y - 2, value * 4, 11, fill=1, stroke=0)
            c.setFillColor(INK)
            c.drawString(146 + value * 4, y, str(value))
        c.setFont('CN', 8)
        c.drawString(410, 11, '访问节点数 个')


def build(font, bold_font, repo_url=REPO_URL):
    pdfmetrics.registerFont(TTFont('CN', font, subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNBold', bold_font, subfontIndex=0))
    data = json.loads((ROOT / 'results' / 'experiment_results.json').read_text(encoding='utf-8'))
    output = ROOT / 'report'
    output.mkdir(exist_ok=True)
    styles = {
        'title': ParagraphStyle('title', fontName='CNBold', fontSize=20, leading=27,
                                textColor=colors.black, spaceAfter=15),
        'h1': ParagraphStyle('h1', fontName='CNBold', fontSize=15, leading=22,
                             textColor=colors.black, spaceAfter=10),
        'h2': ParagraphStyle('h2', fontName='CNBold', fontSize=11.5, leading=18,
                             textColor=colors.black, spaceBefore=7, spaceAfter=5),
        'body': ParagraphStyle('body', fontName='CN', fontSize=10, leading=15.5,
                               textColor=INK, spaceAfter=7, wordWrap='CJK'),
        'small': ParagraphStyle('small', fontName='CN', fontSize=8.5, leading=13,
                                textColor=INK, spaceAfter=6, wordWrap='CJK'),
        'cell': ParagraphStyle('cell', fontName='CN', fontSize=8.6, leading=12,
                               textColor=INK, wordWrap='CJK', alignment=TA_CENTER),
        'head': ParagraphStyle('head', fontName='CNBold', fontSize=8.6, leading=12,
                               textColor=colors.white, wordWrap='CJK', alignment=TA_CENTER),
    }
    story, markdown = [], []

    def para(text, style='body'):
        story.append(Paragraph(escape(str(text)), styles[style]))
        markdown.extend([str(text), ''])

    def heading(text, level=1):
        story.append(Paragraph(escape(text), styles['h1' if level == 1 else 'h2']))
        markdown.extend([('#' * (level + 1)) + ' ' + text, ''])

    def table(headers, rows, widths=None):
        cells = [[Paragraph(escape(str(x)), styles['head']) for x in headers]]
        cells.extend([[Paragraph(escape(str(x)), styles['cell']) for x in row] for row in rows])
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
            ('GRID', (0, 0), (-1, -1), .45, colors.HexColor('#D9D9D9')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.extend([t, Spacer(1, 9)])
        markdown.append('| ' + ' | '.join(headers) + ' |')
        markdown.append('| ' + ' | '.join('---' for _ in headers) + ' |')
        markdown.extend('| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows)
        markdown.append('')

    def page():
        story.append(PageBreak())

    def find(**fields):
        return next(row for row in data['metrics'] if all(row.get(k) == v for k, v in fields.items()))

    def reduction(before, after):
        return f'{(before - after) / before * 100:.1f}%'

    # 报告固定为五个内容页，保留核心证据，详细原始数据随项目提供。
    if not repo_url.startswith('https://github.com/'):
        raise ValueError('代码仓库必须是 https://github.com/ 开头的地址')
    story.append(Paragraph(
        '代码仓库 <link href="' + escape(repo_url, quote=True) + '" color="#254E70">'
        + escape(repo_url) + '</link>', styles['small']))
    markdown.extend(['代码仓库 [' + repo_url + '](' + repo_url + ')', ''])
    story.append(Paragraph('实验二 搜索算法实验报告', styles['title']))
    markdown.extend(['# 实验二 搜索算法实验报告', ''])
    para('实验日期 2026年10月16日', 'small')
    heading('一 实验设置与路径搜索')
    meta = data['metadata']
    para(f"本实验实现 A*、Minimax、Alpha-beta 及宝藏分配。Python {meta['python']}，Windows 11；算法仅依赖标准库。每组预热 {meta['warmups']} 次，计时 {meta['repeats']} 次取中位数，单位 ms。包含统计、还原和排序，不含打印及文件写入；博弈不含距离预计算，后者单独测量。")
    para('A* 状态为坐标，g 为已付费用，h 为剩余费用下界，f=g+h。堆取最小 f；若 new_g 更小，更新 g_score 与前驱 came_from 并入堆。出堆时跳过 g 已过期的记录，抵达目标后沿前驱倒推路径。f 相同按坐标排序，保证结果可复现。')
    heading('题目一 最短路径与对照', 2)
    para('实际路径：(0,0)→(0,1)→(0,2)→(1,2)→(1,3)→(1,4)→(2,4)→(3,4)。8 个坐标对应 7 步。起点曼哈顿下界也是 7，可行路径达到下界，故最短。四方向单位费用下，曼哈顿距离可采纳且一致；h=0 等价于 Dijkstra。')
    sample_rows = []
    for case, label in [('sample_maze', '题目一'), ('sample_weighted', '题目二')]:
        for h in ('zero', 'manhattan'):
            r = find(case=case, heuristic=h)
            sample_rows.append([label, 'h=0' if h == 'zero' else ('曼哈顿' if case == 'sample_maze' else '修正曼哈顿'),
                                r['value'], r['expanded'], f"{r['median_ms']:.4f}"])
    table(['题目', '启发函数', '步数或代价', '扩展节点', '时间 ms'], sample_rows,
          [65, 135, 100, 90, WIDTH - 390])
    para('题目一扩展从 15 降至 13，减少 13.3%；题目二从 14 降至 10，减少 28.6%。扩展计数仅包含真正枚举邻居的节点，不含终点；有效出堆计数另包含终点。', 'small')
    heading('题目二 费用累计与启发函数修正', 2)
    para('路径：(0,0)→(1,0)→(2,0)→(3,0)→(3,1)→(3,2)→(4,2)→(4,3)→(4,4)。进入费用依次为 1、1、1、1、1、1、1、0；累计为 1、2、3、4、5、6、7、7。出发不支付起点费，进入 E 免费，所以 8 步只花 7。')
    para('紧邻 E 时，普通曼哈顿估计 1，真实剩余费用 0，故会高估。令 d 为曼哈顿距离，m 为除免费目标外的最小通行费用，修正 h=max(d-1,0)×m：至少 d 步中，只有末步可免费，其余每步至少 m。相邻普通格的 h 下降不超过 m，进入目标前 h=0，因此也满足一致性。若其他格有零费用，m=0，安全退化为 Dijkstra。')
    para('网格中 E=O(V)，一致启发下堆实现时间 O((V+E)log V)，空间 O(V+E)。最小费用不等于最少步数；原样例正确也不能代替对启发性质的分析。', 'small')

    page()
    heading('二 题目三 极大极小搜索与剪枝')
    para('叶子保存 MAX 视角的分数；非叶子保存 children。递归遇到叶子直接返回分数，MAX 层取最大，MIN 层取最小，根是 MAX。按 PDF 第 10 至 12 页完整输入展开如下，重复的 (6,9) 分支分别访问。')
    story.append(TreeFigure())
    para('图一 博弈树的分数回传。默认从左到右访问，逆序仅用于对照。', 'small')
    para('底层 MAX 依次返回 5、9、9、2；两个 MIN 返回 min(5,9)=5 与 min(9,2)=2；根返回 max(5,2)=5。全部分数始终从 MAX 角度解释，MIN 不是将叶子取负，而是选择较小的结果。')
    heading('剪枝依据与一次具体剪枝', 2)
    para('alpha 为 MAX 已建立的下界，beta 为 MIN 已建立的上界。alpha≥beta 时，当前节点剩余分支无法改善祖先的决定，可以停止。左 MIN 得到 5 后令 beta=5；接着它的第二个 MAX 先看到叶子 6，令 alpha=6，便跳过后面的 9。')
    rows = [r for r in data['metrics'] if r['problem'] == 3]
    table(['算法', '顺序', '值', '访问', '叶子', '剪枝', '时间 ms'], [
        ['Minimax' if r['algorithm'] == 'minimax' else 'Alpha-beta',
         '左到右' if r['order'] == 'left' else '逐层逆序', r['value'], r['visited'],
         r['leaves'], r['cutoffs'], f"{r['median_ms']:.4f}"] for r in rows],
        [90, 85, 40, 50, 50, 50, WIDTH - 365])
    para('两种顺序的 Alpha-beta 都仅省一个叶子；本树逆序没有改变总访问量。访问顺序能影响剪枝，但不是每棵树换序都会有差异。实测 Alpha-beta 比完整 Minimax 更慢，说明小树上边界维护开销可能超过少访问一个叶子的收益。')
    heading('计数口径与复杂度', 2)
    para('visited 是递归进入次数，含根与终止节点；leaves 是终止评价次数；cutoffs 只统计确实跳过剩余分支的剪枝。pruned_children 仅计跳过的直接子分支，不冒充其全部后代。A* generated 计入堆记录，可能含改进后的重复坐标；max_frontier 同样不等于去重坐标数。')
    para('分支因子 b、深度 H 下，Minimax 最坏 O(b^H)；Alpha-beta 最坏仍为 O(b^H)，理想顺序可接近 O(b^(H/2))。两者深度优先递归栈 O(H)，不计输入树存储。', 'small')

    page()
    heading('三 题目四 状态建模与深度决策')
    para('宝藏价值 T1=9、T2=6、T3=7、T4=7。A、B、T 格的进入费用均为 1，不能当免费 E。先对六个关键点逐方向调用 A* 共 30 次，得到下表；对角为 0，行是出发点，列是目的地。')
    labels = data['problem4']['labels']
    distances = data['problem4']['distances']
    table(['出发到达'] + labels, [[s] + [distances[s][t] for t in labels] for s in labels],
          [85] + [(WIDTH - 85) / 6] * 6)
    para('状态记录双方位置、剩余宝藏、双方累计收益和当前玩家。选择 Ti 后，加上 Gain=Vi-d(p,Ti)，更新当前玩家位置、移除该宝藏并换人；新建不可变状态避免分支污染。经过其他宝藏不自动收集，已取宝藏仍可通行，另一玩家不阻挡，所以全局复用距离矩阵。')
    para('U=Score_A-Score_B；A 取最大，B 取最小。深度为 0 或宝藏取完时返回当前 U。H=2 为 A、B 各一次；H=4 为 A、B、A、B 四次。固定首步后对后续应对完整搜索，结果如下。')
    first = data['problem4']['first_target_values']
    table(['A 的首步', 'H=2 最优应对后的 U', 'H=4 最优应对后的 U'],
          [[t, first['2'][t], first['4'][t]] for t in VALUES_ORDER],
          [130, 185, WIDTH - 315])
    para('H=2 选择 T2：A 得 6-4=2，B 取 T3 得 7-4=3，U=-1。H=4 选择 T3，下面列出默认顺序得到的一条双方最优行动序列。')
    table(['玩家', '移动', '费用', '本次收益', '累计 U'], [
        [s['player'], s['from'] + ' → ' + s['target'], s['cost'], s['gain'], s['utility']]
        for s in data['problem4']['traces']['4']], [50, 155, 85, 100, WIDTH - 390])
    para('A 首取 T3 虽立即亏 1，却拿走 B 容易获利的目标。在此序列中，B 最后花 12 取 T4，收益 -5；最终 A=-1、B=-4，U=3。负收益合法且不能跳过回合。最优应对可能不唯一，单条有利序列不能替代对所有应对的比较。', 'small')

    page()
    heading('四 综合题性能与访问顺序')
    para('比较自然、逆序、即时高收益优先和即时低收益优先。双方都按自己的即时收益排序；只调整访问先后，不删除动作。高即时收益不保证完整深度最优，排序的作用是较早建立强的 alpha 或 beta。')
    def order_name(order):
        return {'natural': '自然', 'reverse': '逆序', 'gain': '高收益优先', 'low_gain': '低收益优先'}[order]
    perf_rows = []
    for depth in (2, 4):
        selected = [find(case='treasure_game', depth=depth, algorithm='minimax', order='natural')]
        selected += [find(case='treasure_game', depth=depth, algorithm='alphabeta', order=o)
                     for o in ('natural', 'reverse', 'gain', 'low_gain')]
        for r in selected:
            perf_rows.append([depth, ('MM ' if r['algorithm'] == 'minimax' else 'AB ') + order_name(r['order']),
                              r['visited'], r['leaves'], r['cutoffs'], f"{r['median_ms']:.4f}"])
    table(['深度', '算法及顺序', '访问', '叶子', '剪枝', '时间 ms'], perf_rows,
          [45, 160, 60, 60, 60, WIDTH - 385])
    bar_values = [('Minimax', 65)] + [('AB ' + order_name(o), find(
        case='treasure_game', depth=4, algorithm='alphabeta', order=o)['visited'])
        for o in ('natural', 'reverse', 'gain', 'low_gain')]
    story.append(NodeBars(bar_values))
    para('图二 给定宝藏地图 H=4 的访问节点数，包含根与终止节点。MM 为 Minimax，AB 为 Alpha-beta。', 'small')
    para('完整树为 1+4+12+24+24=65 次访问。默认剪枝为 47，减少 27.7%；高收益优先为 37，减少 43.1%；低收益优先为 53。所有顺序的最优首步和值一致。剪枝次数不能代表省下的全部节点量，且排序本身有开销；耗时受系统调度影响，应结合节点数判断。')
    heading('预计算与缓存的实际效果', 2)
    zr = find(case='distance_matrix', heuristic='zero')
    hr = find(case='distance_matrix', heuristic='manhattan')
    para(f"距离矩阵的 h=0 总扩展 {zr['expanded']} 次，耗时 {zr['median_ms']:.4f} ms；收费目标曼哈顿总扩展 {hr['expanded']} 次，耗时 {hr['median_ms']:.4f} ms。收费目标使用 h=d×最小通行费用；关键点进入费相同使本题距离恰好对称，但程序仍逐方向求解。")
    c2 = find(case='treasure_game', depth=2, algorithm='minimax_cache', order='natural')
    c4 = find(case='treasure_game', depth=4, algorithm='minimax_cache', order='natural')
    para(f"精确值缓存的键含完整状态与剩余深度，H=2、4 均命中 0 次，访问仍为 17、65，耗时 {c2['median_ms']:.4f}、{c4['median_ms']:.4f} ms；没有观察到收益。缓存需要重复状态才能有效。Alpha-beta 截断结果可能仅为界，本实现未将其误存为精确值。", 'small')

    page()
    heading('五 自设实验 正确性验证与结论')
    maze_names = {'open_25': '空旷 25×25', 'wall_gap_25': '绕墙 25×25', 'random_25': '随机 25×25'}
    table(['地图', '最短步数', 'h=0 扩展', '曼哈顿扩展', '减少'], [
        [label, find(case=case, heuristic='zero')['value'],
         find(case=case, heuristic='zero')['expanded'],
         find(case=case, heuristic='manhattan')['expanded'],
         reduction(find(case=case, heuristic='zero')['expanded'],
                   find(case=case, heuristic='manhattan')['expanded'])]
        for case, label in maze_names.items()], [115, 85, 100, 100, WIDTH - 400])
    para('空旷图从 (12,0) 到 (12,24)，启发精确指向同行，扩展从 456 降到 24。绕墙图从 (0,0) 到 (0,24)，第 12 列只有底部缺口，真实距离 72 远大于下界 24，所以仍需较多探索。随机图种子 2026、障碍概率 0.22，固定顶行及右列可通行；这不是无条件随机分布，不能把单图收益推广到所有地图。')
    table(['边界输入', '结果', '验证内容'], [
        ['S 紧邻 E', '代价 0', '免费终点的启发下界'],
        ['S 9 E 下方全为 1', '4 步 代价 3', '比直接 2 步代价 9 更便宜'],
        ['S X E', '返回 -1', '不可达处理'],
        ['S 0 0 E', '代价 0', 'm=0 退化后仍正确'],
    ], [150, 100, WIDTH - 250])
    heading('独立验证与复现', 2)
    para('10 个 unittest 方法全部通过：用独立 BFS 验证迷宫；独立 Dijkstra 验证加权样例、6×6 关键点距离及 60 张种子 1431 的随机图；其中一半仅正费用、一半允许零费用，并逐点检查可采纳性、逐边检查一致性。完整 Minimax 核对给定树及 50 棵种子 2026 的随机树；枚举全部不重复行动排列并逐层归约，独立核对 H=2、H=4。另测起终点相同、零分叶子、负收益、重复收集和状态隔离。')
    para('在实验二目录执行 python run_experiments.py 生成 42 组数据；执行 python -m unittest discover -s code -p "test_*.py" -v 运行验证；执行 python build_report.py 重建不超过 5 页的报告。完整数据为 results/experiment_results.json 与 metrics.csv，详细说明见 README。', 'small')
    heading('结论与局限', 2)
    para('A* 的安全下界必须反映费用规则，尤其是免费终点和零费用地形。Alpha-beta 保持 Minimax 的值，实际收益受访问顺序和计算开销影响。宝藏题中，完整四回合的位置变化与竞争令最优首步从 T2 变为 T3，说明短期即时收益不足以决定长期策略。')
    para('距离预计算依赖地图和通行规则不变；若玩家阻路或取宝藏改变地形，应重新评估复用条件。当前测量只覆盖给定输入与有限自设场景，计时受机器影响；扩大地图、增加宝藏和多随机分布对照，可进一步检验排序、缓存与复杂度瓶颈。')
    para('参考：课程文件 实验二要求_new.pdf，第 3 至 15 页题目与样例、第 16 至 17 页报告及评分要求。代码、实测数据及复现入口随项目提供。', 'small')

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('CN', 8)
        canvas.setFillColor(colors.HexColor('#475569'))
        canvas.drawRightString(A4[0] - 48, 25, f'第 {doc.page} 页')
        canvas.restoreState()

    pdf_path = output / '实验二报告.pdf'
    document = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                 rightMargin=48, leftMargin=48,
                                 topMargin=38, bottomMargin=42,
                                 title='实验二 搜索算法实验报告', author='')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    if document.page > 5:
        raise RuntimeError(f'报告实际为 {document.page} 页，超过 5 页上限，请调整内容后重新生成')
    (output / '实验二报告.md').write_text('\n'.join(markdown), encoding='utf-8')
    print('已生成 PDF 和 Markdown 报告')


VALUES_ORDER = ('T1', 'T2', 'T3', 'T4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', default='C:/Windows/Fonts/simsun.ttc')
    parser.add_argument('--bold-font', default='C:/Windows/Fonts/simhei.ttf')
    parser.add_argument('--repo-url', default=REPO_URL,
                        help='报告页首的 GitHub 仓库或作业分支链接')
    args = parser.parse_args()
    build(args.font, args.bold_font, args.repo_url)
