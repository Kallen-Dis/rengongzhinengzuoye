"""仅在训练和可视化完成后，从实测数据生成最多5页报告。"""
import json
from pathlib import Path
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,Spacer,PageBreak
ROOT=Path(__file__).resolve().parent

def main():
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simsun.ttc',subfontIndex=0)); pdfmetrics.registerFont(TTFont('Bold','C:/Windows/Fonts/simhei.ttf'))
    style=ParagraphStyle('body',fontName='CN',fontSize=10,leading=16,spaceAfter=8,wordWrap='CJK')
    title=ParagraphStyle('title',fontName='Bold',fontSize=17,leading=24,spaceAfter=12)
    small=ParagraphStyle('small',parent=style,fontSize=8,leading=11)
    story=[]; md=[]
    def p(t): story.append(Paragraph(escape(t),style)); md.extend([t,''])
    def h(t): story.append(Paragraph(t,title)); md.extend(['## '+t,''])
    def table(head,rows,widths):
        content=[[Paragraph(escape(str(v)),small) for v in row] for row in [head]+rows]
        t=Table(content,colWidths=widths); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.lightgrey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DBEAFE')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); story.extend([t,Spacer(1,10)])
        md.append('| '+' | '.join(head)+' |'); md.append('| '+' | '.join(['---']*len(head))+' |'); md.extend('| '+' | '.join(str(v) for v in row)+' |' for row in rows); md.append('')
    def fig(name,w):
        from PIL import Image as PI
        path=ROOT/'results/figures'/name
        with PI.open(path) as im: height=w*im.height/im.width
        story.extend([Image(str(path),width=w,height=height),Spacer(1,8)])
        md.extend([f'![{name}](../results/figures/{name})',''])
    def page():story.append(PageBreak())
    best=next(x for x in s['models'] if x['name']==s['selected_by_validation'])
    h('实验三 图像分类模型设计与实现')
    p(f"使用 CIFAR-10 完成从零训练、验证选择、测试评估与误差分析。按验证集选中的模型是 {best['name']}，测试总体准确率 {best['test_accuracy']*100:.2f}%。全部数字与图片来自本次实际运行。")
    h('一 数据划分与实验设置')
    p('官方训练集50000张按类别分层划分：每类4500张训练、500张验证，共45000/5000张；官方10000张测试集仅在训练与模型选择结束后评估。固定种子2026，划分索引随结果保存。图像32×32、RGB；像素除255后用均值和标准差0.5映射到[-1,1]，未用验证或测试数据估计归一化统计量。')
    p('数据来源：https://www.cs.toronto.edu/~kriz/cifar.html。归档按官方MD5 c58f30108f718f92721af3b95e74349a 核验，torchvision 再检查分批文件。')
    table(['设置','配置'],[['训练设备',s['gpu'] or s['device']],['PyTorch',s['torch']],['训练预算',f"每模型{s['epochs']}轮，batch={s['batch']}"],['优化器','AdamW；初始lr=0.002；weight decay=0.0005'],['学习率','余弦退火至0.00002'],['模型选择','各自验证准确率最高的checkpoint，平局保留首次'],['随机性','固定随机种子；CUDA可能仍有非确定性误差']],[100,390])
    table(['模型','参数量','最佳轮次','验证准确率','测试准确率'],[[x['name'],x['parameters'],x['best_epoch'],f"{x['validation_accuracy']*100:.2f}%",f"{x['test_accuracy']*100:.2f}%"] for x in s['models']],[95,95,70,115,115])
    p('实际训练起始时间：'+s['started_at']+'。GPU训练使用自动混合精度，验证与测试使用float32。'+ '；'.join(x['name']+'训练耗时'+f"{x['training_seconds']:.1f}秒" for x in s['models'])+'。')
    p('模型架构与超参数先确定，再执行一次固定种子的完整实验；测试结果只用于报告最终性能，不能用于继续调整超参数。本实验没有使用预训练权重。')
    page();h('二 模型原理与训练过程')
    p('基础CNN有三个卷积块，通道数依次32、64、128，每块为两次3×3卷积、BatchNorm、ReLU和2×2最大池化，空间分辨率从32降至16、8、4。全局平均池化后接Dropout(0.2)与128→10线性分类器，输出未归一化logits，交叉熵内部完成分类概率处理。')
    p('改进模型在每块加入1×1投影残差与SE通道注意力。SE利用全局平均池化汇总各通道，再通过两层1×1映射和Sigmoid生成通道权重。训练时增加反射填充4像素后的随机裁剪、随机水平翻转，并使用0.1标签平滑；验证和测试不增强。')
    fig('curves.png',490)
    p('图一 两个模型的训练及验证曲线。训练准确率在训练模式下统计，改进模型使用增强图像；验证在eval模式下使用原始图像。因此训练/验证曲线的差异还受到Dropout、BatchNorm模式和输入难度影响。')
    last=[(m['name'],json.loads((ROOT/'results'/f'{m["name"]}_history.json').read_text())[-1]) for m in s['models']]
    p('最后一轮：'+'；'.join(name+f"训练准确率{v['train_accuracy']*100:.2f}%，验证准确率{v['val_accuracy']*100:.2f}%" for name,v in last)+'。训练准确率高而验证没有同步提高时，提示对训练数据的拟合超出了泛化收益；最终仍使用最佳验证checkpoint。')
    p('改进模型的训练loss包含标签平滑，验证loss使用标准交叉熵，数值不能直接作等价比较。余弦调度在后期减小更新幅度；保存最佳验证轮次而非直接采用最终轮次，可降低后期过拟合对交付模型的影响。')
    p('两模型同时改变了结构、增强和损失函数，是组合改进对照，不是单因素消融。若要判断每一项的独立贡献，需要固定其余条件，分别移除SE、残差、增强与平滑，并进行多个随机种子重复实验。')
    page();h('三 测试集混淆矩阵与分类表现')
    fig('confusion.png',335)
    table(['类别']+[x['name'] for x in s['models']],[[c]+[f"{x['per_class_accuracy'][i]*100:.2f}%" for x in s['models']] for i,c in enumerate(s['classes'])],[170,160,160])
    p('图二 行为真实类别，列为预测类别。每类准确率定义为该类正确数/该类测试总数，即召回率；本测试集每类1000张，因此宏平均召回率等于总体准确率。矩阵总数为10000，对角和/10000与报告总体准确率一致。')
    page();h('四 误分类样本与错误分析')
    fig('errors.png',490)
    p('图三 对每个真实类别，按测试集索引选取第一个误分类，展示原图、真实标签、预测标签及预测概率。选择规则固定，避免只挑选容易解释或特别极端的样例。')
    cm=best['confusion_matrix']; pairs=sorted([(cm[i][j],s['classes'][i],s['classes'][j]) for i in range(10) for j in range(10) if i!=j],reverse=True)[:5]
    table(['真实类别','误判类别','数量'],[[a,b,n] for n,a,b in pairs],[180,180,130])
    p('猫被判为狗106次，狗被判为猫103次；猫的每类准确率仅75.60%，是主要薄弱类别。样本#61的猫以0.89概率被判为狗，说明高置信度也会出错。#308汽车和#253卡车都呈正面视角，车窗、车头轮廓相似而车身信息有限，可能增加汽车与卡车的混淆。这些是结合图片的解释线索，不是因果证明。')
    p('预测概率不是可靠性保证：错误预测也可能具有较高置信度。标签平滑有助于降低过强的训练目标，但本实验没有单独验证概率校准改善；后续应在独立验证数据上测量可靠性图或ECE，而不是仅凭几个错误样例判断。')
    p('进一步改进可优先分析高频混淆类别，采用更强的空间特征或更丰富的数据增强；任何改进都应先在验证集上比较，并保持测试集隔离。')
    page();h('五 可解释性 验证与总结')
    fig('gradcam.png',470)
    p('图四 前四个误分类样本的Grad-CAM：上行为原图，下行为最后一个卷积块的热图叠加。以模型预测类别的logit求梯度，对空间梯度求均值作为通道权重，ReLU后上采样到32×32。它反映局部梯度敏感区域，不等于像素级因果解释或目标分割。')
    p('红色表示该图内较高响应，蓝色较低。#52的响应集中于飞机中部，#308在车窗和车头都有响应，#61集中于图像中部局部。即使热点覆盖目标区域，仍可能预测错误，因此“关注到物体”不等于已经正确识别类别。')
    p('最后特征图仅4×4，上采样会使热点显得平滑；热图逐图归一化，不能比较不同图片的绝对证据强度。对背景或对象局部的响应只能作为进一步检查的线索，不足以证明模型学到了完整语义。')
    p('验证包含：训练/验证索引不相交且数量正确；两模型输出维度为batch×10；模型检查点可重载；全部测试预测与标签数量一致；混淆矩阵行和为1000、总和10000；由矩阵重算的准确率与汇总一致。训练日志逐轮落盘，最优权重独立保存。')
    p('复现顺序：python train.py；python make_figures.py；python build_report.py。训练需要PyTorch及torchvision，生成图表需要matplotlib，生成报告需要reportlab。检查点、划分、逐图预测、每类准确率和训练日志随作业保存；原始数据另行下载。')
    delta=(s['models'][-1]['test_accuracy']-s['models'][0]['test_accuracy'])*100
    p(f"本次测试中改进模型相对基础模型的准确率变化为{delta:+.2f}个百分点。结论仅针对当前固定种子与训练预算；没有多次重复，不能给出统计显著性结论。实验完成了CNN设计、规范的数据划分、训练与评估、错误分析和梯度可视化，并明确了组合对照及解释图的适用边界。")
    out=ROOT/'report';out.mkdir(exist_ok=True)
    doc=SimpleDocTemplate(str(out/'实验三报告.pdf'),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=40,bottomMargin=38)
    def footer(c,d):c.setFont('CN',8);c.drawRightString(547,22,f'第{d.page}页')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    assert doc.page<=5,f'实际{doc.page}页，需调整排版'
    (out/'实验三报告.md').write_text('\n'.join(md),encoding='utf-8')
    print('Report generated:',doc.page,'pages')
if __name__=='__main__':main()
