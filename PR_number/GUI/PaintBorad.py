from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen, QSize, QColor
from PyQt5.QtCore import Qt


class PainBoard(QWidget):

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.__initWidget__()

    def __initWidget__(self):
        self.__size__ = QSize(460, 460)  # 设置画板大小

        self.__board__ = QPixmap(self.__size__)  # 创建画板对象，白色填充
        self.__board__.fill(Qt.white)

        self.erase = False  # 默认橡皮擦为关

        self.isEmpty = True  # 默认为空画板
        self.__lastPos__ = QPoint(0, 0)  # 上一次鼠标位置
        self.__currentPos__ = QPoint(0, 0)  # 当前的鼠标位置
        self.__painter__ = QPainter()  # 新建画笔

        self.__thickness__ = 30  # 默认画笔粗细为10px
        self.__penColor__ = QColor("black")  # 设置默认画笔颜色为黑色

        self.setFixedSize(self.__size__)  # 设置界面尺寸

    def clear(self):
        self.__board__.fill(Qt.white)
        self.update()
        self.isEmpty = True

    def changeThick(self, tn=30):
        self.__thickness__ = tn

    def getImg(self):
        # 以图片形式返回画板内容
        return self.__board__.toImage()

    def paintEvent(self, paintEvent):
        # 绘图事件
        self.__painter__.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter__.drawPixmap(0, 0, self.__board__)
        self.__painter__.end()

    def mousePressEvent(self, mouseEvent):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos__ = mouseEvent.pos()
        self.__lastPos__ = self.__currentPos__

    def mouseMoveEvent(self, mouseEvent):
        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos__ = mouseEvent.pos()
        self.__painter__.begin(self.__board__)

        if self.erase == False:
            # 非橡皮擦模式
            self.__painter__.setPen(QPen(self.__penColor__, self.__thickness__))  # 设置画笔颜色，粗细
        else:
            # 橡皮擦模式下画笔为纯白色，粗细为10
            self.__painter__.setPen(QPen(Qt.white, self.__thickness__))

        # 画线
        self.__painter__.drawLine(self.__lastPos__, self.__currentPos__)
        self.__painter__.end()
        self.__lastPos__ = self.__currentPos__

        self.update()  # 更新显示

    def mouseReleaseEvent(self, mouseEvent):
        self.isEmpty = False  # 画板不再为空
