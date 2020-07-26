import numpy as np
import matplotlib.pyplot as plt



if __name__ =='__main__':


    x=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    y=[0.5997,0.6728,0.7119,0.7219,0.7317,0.7400,0.7528,0.7618,0.7695,0.7684]
    y1=[0.5699,0.6080,0.6410,0.6672,0.6900,0.7077,0.7108,0.7217,0.7284,0.7395]
    y2=[0.5578,0.5879,0.6197,0.6454,0.6661,0.6832,0.6954,0.7007,0.7149,0.7205]
    y3=[0.6153,0.6430,0.6738,0.6977,0.7139,0.7266,0.7300,0.7434,0.7506,0.7600]
    y4=[0.7183,0.7303,0.7533,0.7639,0.7658,0.7714,0.7857,0.7876,0.7888,0.7899]
    y5=[0.73183,0.730343,0.743533,0.734639,0.763458,0.7714,0.7857,0.7876,0.7888,0.7899]
    y6=[0.713183,0.7130343,0.7423533,0.7346339,0.23763458,0.7714,0.7857,0.7876,0.7888,0.7899]
    y7=[0.7456183,0.734560343,0.745463533,0.72135334639,0.32763458,0.7714,0.7857,0.7876,0.7888,0.7899]
    plt.xlim(0,0.11)
    plt.ylim(0.55,0.8)
    plt.plot(x,y,marker='^',label=u'deepwalk')
    plt.plot(x,y1,marker='*',label=u'sdne')
    plt.plot(x,y2,marker='h',label=u'line')
    plt.plot(x,y4,marker='s',label=u'GIC2Gauss')
    plt.plot(x,y3,marker='p',label=u'G2G_oh')
    plt.plot(x,y5,marker='+',label=u'G2G_oh')
    plt.plot(x,y6,marker='<',label=u'G2G_oh')
    plt.plot(x,y7,marker='>',label=u'G2G_oh')

    plt.legend(edgecolor='black',loc=4)#让图例标签展示
    plt.xlabel(u"Percentageoflabelednodes")#X轴标签
    plt.ylabel('MicroF1')#Y轴标签
    plt.title('Cora')#标题

    plt.show()
