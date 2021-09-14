# 基于python的Opencv项目实战


相关代码(我都运行过！)
链接：https://pan.baidu.com/s/10p3qrBcMkjXscTAw3Xpm1g 
提取码：KKKK 
听说失效了：再来一个：
链接: https://pan.baidu.com/s/1wi9MrBvAhGxyAj1i0NAukQ 提取码: tj2h 复制这段内容后打开百度网盘手机App，操作更方便哦

* 环境配置：（代码里也有）
* python 3.6
* opencv-contrib-python=3.4.1.15
* opencv-python=3.4.1.15

## 1.图像基本操作

### 1.1 图像构成

* 一般分为有颜色图像和无颜色图像

* 像素点在[0,255]之间，值越小越黑，反之亦然

* 彩色图像：有颜色通道，一般读取的图像即为(h,w,c)
  * 如RGB，BGR：(xx,xx,xx)
* 灰度图像:即无通道，一般读取的图像即为(h,w)
### 1.2 数据读取-图像

import cv2

opencv读取的彩色图片的颜色通道格式是BGR

所以如果要用其他函数展示opencv读取的图片，最好要转变成相同的格式，如RGB，BGR等

* img = cv2.imread(filepath,cv2.IMREAD_COLOR)
	* cv2.IMREAD_COLOR：彩色图像
	* cv2.IMREAD_GRAYSCALE：灰度图像
* 一般显示图片
'''
def cv_show(name,img):
    cv2.imshow(name,img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
'''

### 1.3 数据读取-视频
* cv2.VideoCapture可以捕获摄像头，用数字来控制不同的设备，例如0,1。
* 如果是视频文件，直接指定好路径即可。
* 具体代码看文件！

### 1.4 截取部分图像数据
也即处理nparray数据，对应位置切片即可
* cat=img[0:50,0:200] 

### 1.5 颜色通道提取
* b,g,r=cv2.split(img)
or
* b1,g1,r1=img[:,:,0],img[:,:,1],img[:,:,2]

### 1.6 边界填充
* cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType)
* borderType:
    - BORDER_REPLICATE：复制法，也就是复制最边缘像素。
    - BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb   
    - BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
    - BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg  
    - BORDER_CONSTANT：常量法，常数值填充。

### 1.7 数值计算
'''
相当于% 256
防止超过255，则进行mod运算，防止溢出
(img_cat + img_cat2)[:5,:,0] 
'''

''''
防止超过255，则进行取上界运算，防止溢出
cv2.add(img_cat,img_cat2)[:5,:,0]
'''

### 1.8 图像融合
要相同大小的图片才可以进行

* img_dog = cv2.resize(img_dog, (500, 414))

‘’‘
R = a*x1+b*x2+b
融合的方式：有权重有偏置
res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)

重新改变大小，进行比值计算
x4倍，y4倍扩大
res = cv2.resize(img, (0, 0), fx=4, fy=4)
plt.imshow(res)
’‘’


## 2.图像处理
### 2.1 图像阈值
#### ret, dst = cv2.threshold(src, thresh, maxval, type)


- src： 输入图，只能输入单通道图像，通常来说为灰度图
- dst： 输出图
- ret:  返回阈值
- thresh： 阈值
- maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
- type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV

- cv2.THRESH_BINARY           超过阈值部分取maxval（最大值），否则取0
- cv2.THRESH_BINARY_INV    THRESH_BINARY的反转
- cv2.THRESH_TRUNC            大于阈值部分设为阈值，否则不变
- cv2.THRESH_TOZERO          大于阈值部分不改变，否则设为0
- cv2.THRESH_TOZERO_INV  THRESH_TOZERO的反转


eg:

* ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

### 2.2图像平滑
类似卷积操作

* 均值滤波=简单的平均卷积操作
	* blur = cv2.blur(img, (3, 3)) 
* 方框滤波=基本和均值一样，可以选择归一化，只是多了一个参数
	* -1：表示处理后的颜色通道数是一致的
	* box = cv2.boxFilter(img,-1,(3,3), normalize=True)  
	* box = cv2.boxFilter(img,-1,(3,3), normalize=False)  
	* 不选择归一化，则像素点可以超过255
* 高斯滤波=高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
	* 离中心点越接近的值是其可能性越大，充分考虑到不同像素点对中心点的影响
	* aussian = cv2.GaussianBlur(img, (5, 5), 1)  
* 中值滤波=相当于用中值代替
	* median = cv2.medianBlur(img, 5)  # 中值滤波
* 展示所有的图片，使用np的堆叠功能
	* 水平堆叠：hsres = np.hstack((blur,aussian,median))
	* 垂直堆叠：vsres = np.vstack((blur,aussian,median))

### 2.3形态学
字面意思理解即可，实在不行运行代码看效果！
#### 2.3.1 腐蚀操作
* kernel = np.ones((3,3),np.uint8) 
* erosion = cv2.erode(img,kernel,iterations = 1)
#### 2.3.2 膨胀操作
* kernel = np.ones((3,3),np.uint8) 
* dige_dilate = cv2.dilate(dige_erosion,kernel,iterations = 5)
#### 2.3.3 开运算与闭运算
* 开运算：先腐蚀，再膨胀
	* kernel = np.ones((5,5),np.uint8) 
	* opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
* 闭运算：先膨胀，再腐蚀
	* kernel = np.ones((5,5),np.uint8) 
	* closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#### 2.3.4 梯度运算
梯度=膨胀-腐蚀，也就相当于找图片不同像素点之间的差异大小，类似求导的感觉，所以叫梯度
* gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
#### 2.3.5 礼帽与黑帽
- 礼帽 = 原始输入-开运算结果 = 原始输入-(先腐蚀，再膨胀)
	-  tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

- 黑帽 = 闭运算-原始输入 = (先膨胀，再腐蚀)-原始输入
	- blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel)
### 2.4 图像梯度
* Sobel算子：计算像素点之间的差异大小，一般有从x方向和y方向
	* dst = cv2.Sobel(src, ddepth, dx, dy, ksize)
		- ddepth:图像的深度，一般为-1
		- dx和dy分别表示水平和竖直方向，判断是计算x方向还是y方向，通过指定0or1
		- ksize是Sobel算子的大小
	* sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
	* sobelxy = cv2.convertScaleAbs(sobelxy) 
	* 分别计算x和y，再求和，效果更好
	* cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3) 效果不太行
* Scharr算子：类似Sobel算子，只是数值略有不同
	*  scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
	*  scharrx = cv2.convertScaleAbs(scharrx) 
* Laplacian算子：考虑到二阶导，但对噪音点敏感 
	* laplacian = cv2.Laplacian(img,cv2.CV_64F)
	* laplacian = cv2.convertScaleAbs(la placian)  

### 2.5 Canny边缘检测
- 1)        使用高斯滤波器，以平滑图像，滤除噪声。
- 2)        计算图像中每个像素点的梯度强度和方向。
- 3)        应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应(如人脸检测的框的确定)。
- 4)        应用双阈值（Double-Threshold=minval,maxval）检测来确定真实的和潜在的边缘。
- 5)        通过抑制孤立的弱边缘最终完成边缘检测。
- v1=cv2.Canny(img,minval,maxval)

### 2.6 图像金字塔
也即对图像进行提取特征的操作，放大或者缩小图像
*  高斯金字塔
	* 向下采样方法(缩小):图像与高斯核卷积，再去除所有偶数行与列
	* 向上采样方法(放大):图像在每个方向扩大用来的两倍(右，下，右下)，用0填充，再使用高斯核卷积卷积
	* up=cv2.pyrUp(img)
	* down=cv2.pyrDown(img)
* 拉普拉斯金字塔
	* L = G-pyrUp(pyrdown(G))
	* down=cv2.pyrDown(img)
	* down_up=cv2.pyrUp(down)
	* l_1=img-down_up

### 2.7 图像轮廓
#### 2.7.1 提取轮廓
* cv2.findContours(img,mode,method)
* mode:轮廓检索模式
	- RETR_EXTERNAL ：只检索最外面的轮廓；
	- RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
	- RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
	- RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;(一般最常使用)

* method:轮廓逼近方法
	- CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
	- CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
* 为了更高的准确率，使用二值图像
* ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
* binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#### 2.7.2 绘制轮廓
* draw_img = img.copy()
* -1：画所有轮廓，如果是2，表示画第2个轮廓
* 2：绘制线条的宽度
* res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)

#### 2.7.3 轮廓特征
* cnt = contours[0] ,获得第0个轮廓
* 计算轮廓面积：cv2.contourArea(cnt)
* 周长：cv2.arcLength(cnt,True)

#### 2.7.4 轮廓近似
由于有些轮廓比较粗糙或者太详细，可以改变近似算法的阈值来调整
* cv2.approxPolyDP()近似函数
* cnt：要对哪个轮廓进行近似
* epsilon:设置近似的阈值
	* 一般为cv2.arcLength(cnt,True)轮廓长的倍数
*  approx = cv2.approxPolyDP(cnt,epsilon,True)
* res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)

#### 2.7.5 轮廓-边界矩形
通过轮廓绘制该轮廓的边界矩形
* 先通过之前的步骤获取轮廓
* cnt = contours[0]
* x,y,w,h = cv2.boundingRect(cnt)
* img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
* cv_show(img,'img')
* 也可以计算轮廓面积与边界矩形的面积比
* area = cv2.contourArea(cnt)
* rect_area = w * h

#### 2.7.6 轮廓-外接圆
类似于边界矩形的绘制方法
* 先通过之前的步骤获取轮廓
* cnt = contours[0]
* (x,y),radius = cv2.minEnclosingCircle(cnt) 
* center = (int(x),int(y)) 
* radius = int(radius) 
* img = cv2.circle(img,center,radius,(0,255,0),2)

## 3. 模板匹配
就是给定一张小图片，在另一张大图片中寻找是否存在相同的图片，所以可以通过滑动窗口+计算差别度来确定

* 模板匹配和卷积原理很像，模板在原图像上从原点开始滑动，计算模板与（图像被模板覆盖的地方）的差别程度，这个差别程度的计算方法在opencv里有6种，然后将每次计算的结果放入一个矩阵里，作为结果输出。假如原图形是AxB大小，而模板是axb大小，则输出结果的矩阵是(A-a+1)x(B-b+1)

* 最好要转变为灰度图再进行处理
* res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
* res:每个窗口左上角的值和损失大小
* - TM_SQDIFF：计算平方不同，计算出来的值越小，越相关        
	- TM_CCORR：计算相关性，计算出来的值越大，越相关
	- TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
	- TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
	- TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
	- TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关
	- 最好使用带归一化的结果
* min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
* 可视化：显示出匹配的模板的位置：画矩形
	* h, w = template.shape[:2] 
	* top_left = min_loc or top_left = max_loc看使用的是什么参数
	* bottom_right = (top_left[0] + w, top_left[1] + h)
	* cv2.rectangle(img2, top_left, bottom_right, 255, 2)

* 匹配多个对象
	* 设定一个阈值来筛选res即可
	* loc = np.where(res >= threshold)
	* for pt in zip(*loc[::-1]):  # *号表示可选参数
    	* bottom_right = (pt[0] + w, pt[1] + h)
		* cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

## 4. 直方图
统计图像中相同像素点值的个数
* cv2.calcHist(images,channels,mask,histSize,ranges)
	- images: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应 用中括号 [] 括来例如[img]
	- channels: 同样用中括号括来它会告函数我们统幅图 像的直方图。如果入图像是灰度图它的值就是 [0]如果是彩色图像 的传入的参数可以是 [0][1][2] 它们分别对应着 BGR。 
	- mask: 掩模图像。统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。
	- histSize:BIN 的数目（直方图中的横坐标的数目）。也应用中括号括来
	- ranges: 像素值范围常为 [0-256] 
 *  hist = cv2.calcHist([img],[0],None,[256],[0,256])
 * 也可以用plt.hist(img.ravel(),256)直接绘制img的直方图
 * mask操作
 	* mask = np.zeros(img.shape[:2], np.uint8)
 	* mask[100:300, 100:400] = 255 设置要进行提取直方图的图像区域
 	* 也可以通过以下代码进行查看
 		* masked_img = cv2.bitwise_and(img, img, mask=mask)#与操作
		* cv_show(masked_img,'masked_img')
### 4.1 直方图直方图均衡化
* 因为图像中的像素点的值个数分布不均，因此可以采取均衡化,使图像中的像素点个数分布均衡
* equ_img = cv2.equalizeHist(img) 
### 4.2 自适应直方图均衡化
* 之前的是整张图像进行均衡化，也可以将一张图像分割为几个部分进行均衡化
* 但是更容易受到噪声的影响
* clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
* res_img = clahe.apply(img)

## 5.傅里叶变换
https://zhuanlan.zhihu.com/p/19763358

因为先转换到频域中对图像做处理和直接对图像做处理更简单

所以傅里叶变换就是把数据转变到频域上的方法？

将图像转换为频域进行分析操作
*  傅里叶变换的作用
	- 高频：变化剧烈的灰度分量，例如边界
	- 低频：变化缓慢的灰度分量，例如一片大海

* 滤波
	* 低通滤波器：只保留低频，会使得图像模糊
	* 高通滤波器：只保留高频，会使得图像细节增强
- opencv中主要就是cv2.dft()和cv2.idft()，输入图像需要先转换成**np.float32** 格式。
- 得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现。
- cv2.dft()返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）。

* 将图片转换为np.float32格式：img_float32 = np.float32(img)
* 执行傅里叶变换
	* dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
* 将低频域值转换到中心位置
	* dft_shift = np.fft.fftshift(dft)
* 得到图像的强度频谱：越中间的像素点出现的频率越低
	* 需要转换成图像格式进行展示
	* magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
	* plt.imshow(magnitude_spectrum, cmap = 'gray')
* 通过强度频谱可以获得低通滤波or高通滤波
	* 获得低通滤波（高通相反设置mask即可np.ones...）
	* mask = np.zeros((rows, cols, 2), np.uint8)
	* 中间部分为1，其余部分为0
	* mask[crow-30:crow+30, ccol-30:ccol+30] = 1
* 对图像进行低通滤波
	* fshift = dft_shift*mask
* 将低频域值转换到原来左上角位置
	* f_ishift = np.fft.ifftshift(fshift)
* 逆傅里叶变换
    * 需要转换成图像格式进行展示
    * img_back = cv2.idft(f_ishift)
    * img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

## 6.项目一：提取银行卡数字
主要思路：
* 提取模板数字，并且与之对应(数字-图)
	* 转灰度图
	* 转二值图
	* 计算轮廓，排序轮廓，对应矩形，字典：数字-图
* 提取图像中的数字
	* 改变大小
	* 转灰度图
	* 礼帽操作：原始输入-(先腐蚀，再膨胀)=>突出更明亮的区域
	* 计算梯度：Sobel算子，绝对化，归一化
	* 闭操作：先膨胀，再腐蚀=>将数字连在一起
	* 转二值图
	* 闭操作：先膨胀，再腐蚀=>填充数字框里面的空隙
	* 计算轮廓
	* 找数字组所在的轮廓：并且找到其对应矩形，再从左到右排序
		* 在一组数字中在计算轮廓，找其对应矩形
		* 遍历矩形中的每一个数字与模板中的每一个数字进行模板匹配
	* 获得结果，在对应位置打印输出
## 7.项目二：文档扫描ORC
主要思路：
* 图像按比例变化大小
* 图像预处理
	* 转灰度图
	* 高斯去噪
* Canny边缘检测
* 再由边缘检测出的图像进行轮廓检测
	* 根据轮廓面积排序选择最大的五个轮廓
	* 遍历这些轮廓，进行轮廓近似，选择最大的矩形
	* 获取轮廓
* 根据矩形的四个坐标点进行透视变化(平移，翻转，扩大)
	* 计算变化矩阵
	* 进行透视变化
* 进行二值处理，保存图像
* 使用开源的ORC项目：tesseract
	* 进行对图像的文字提取(图片预处理很重要)
	
## 8.图像特征
### 8.1 Harris角点检测
* 边界：沿着水平or竖直区域，一个比较平稳，一个比较剧烈的变化，即称为边界
* 角点：无论沿着水平还是竖直区域，变化都比较剧烈，即称为角点。
* 因此角点的特征更为丰富！
* Harris提取角点的基本原理：
	* 对图像进行平移操作
	* 计算平移后每个像素点的自相似性c
	* 自相似的计算=>泰勒展开=>矩阵计算=>特征值
	* 可以根据特征值来判断是否为角点或边界
	* 也有人提出角点响应值R来判断
	* 均有数学公式
* 实现：
	* cv2.cornerHarris() 
		- img： 数据类型为 ﬂoat32 的入图像
			- gray = np.float32(gray)
		- blockSize： 角点检测中指定区域的大小
		- ksize： Sobel求导中使用的窗口大小 
		- k： 取值参数为 [0,04,0.06]
	* dst = cv2.cornerHarris(gray, 2, 3, 0.04)
	* 绘制图像中的角点
		* img[dst>0.1*dst.max()]=[0,0,255]
### 8.2 尺度不变特征变换-Scale Invariant Feature Transform(SIFT)
* 图像尺度空间
	* 让机器能够对物体在不同尺度下有一个统一的认知，就需要考虑图像在不同的尺度下都存在的特点。
	* 通常使用高斯模糊来获取图像的尺度空间
	* 不同σ的高斯函数决定了对图像的平滑程度，越大的σ值对应的图像越模糊。
	* 有数学公式
* 多分辨率金字塔
	* 不同尺度(大小)下的图片都要获取图像尺度空间
* 高斯差分金字塔(DOG)
	* 因为要找到图像的特征，因此要找其差异性，而不是相同的地方
		* 所以：通过相同规模(大小)的图像进行减法，获得差异性结果。
		* 有数学公式
* DOG空间极值检测
	* 找尺度空间的极值点
		* 每个像素点要和其图像域（同一尺度空间）和尺度域（相邻的尺度空间）的所有相邻点进行比较，当其大于（或者小于）所有相邻点时，该点就是极值点。
		* 也就是不仅要在一张图片上比较，还要和同层级的其他图片进行比较
		* 也即是立体的邻域比较
* 关键点精确定位
	* 因为极值点可能是DOG空间的局部极值点，而我们尽可能希望得到的是全局的极值点
	* 又因为极值点均离散
	* 因此
		* 对尺度空间DOG函数进行曲线拟合，即可计算出极值点，获得关键点的位置
			* 有数学公式
			*  泰勒展开=>矩阵计算
* 消除边界响应
	* 也即消除边缘不稳定噪声点 
	* 由于DoG函数在图像边缘有较强的边缘响应，因此需要排除边缘响应
	* 一般使用Hessian矩阵
	* 有数学公式
* 特征点的主方向
	* 每个特征点可以得到三个信息(x,y,σ,θ)，即位置、尺度和方向
	* 生成特征描述
		* 在完成关键点的梯度计算后，使用直方图统计邻域内像素的梯度和方向。
		* 为了保证特征矢量的**旋转不变性**，要以特征点为中心，在附近邻域内将坐标轴旋转θ角度
			*有坐标变换的数学公式 
		* 旋转之后的主方向为中心取8x8的窗口，求每个像素的梯度幅值和方向，箭头方向代表梯度方向，长度代表梯度幅值，然后利用高斯窗口对其进行加权运算，最后在每个4x4的小块上绘制8个方向的梯度直方图，计算每个梯度方向的累加值，即可形成一个种子点，即每个特征的由4个种子点组成，每个种子点有8个方向的向量信息。
		* 从而获得了一个32维的SIFT特征向量
* 实现：
	* 实例化sift算法
		* sift = cv2.xfeatures2d.SIFT_create()
	* 对图像进行sift检测，一般使用灰度图
		* kp = sift.detect(gray, None) 
	* 通过opencv封装好的函数绘制SIFT特征点出来
		* img = cv2.drawKeypoints(gray, kp, img) 
	* 计算SIFT特征向量
		* 计算特征，返回原来的关键点和关键点的特征向量：
		* kp, des = sift.compute(gray, kp)

### 8.3 特征匹配
能够获得图像的SIFT特征后，可以对多幅图像进行特征匹配，获取需要的信息
#### 8.3.1 Brute-Force蛮力匹配
* 获取灰度图
* 实例化sift算法
	* sift = cv2.xfeatures2d.SIFT_create()
* 检测并且计算SIFT特征向量和特征点
	* kp1, des1 = sift.detectAndCompute(img1, None)
* 进行BF匹配
	* crossCheck表示两个特征点要互相匹配
		* 例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是最近的
	* 默认使用Norm_l2：归一化数组的(欧几里德距离)，如果其他特征计算方法需要考虑不同的匹配计算方式
	* bf = cv2.BFMatcher(crossCheck=True)
* 1对1的匹配：特征点一对一的匹配：一个特征点对应一个特征点
	* matches = bf.match(des1, des2) 
	* 根据度量值进行排序
	* matches = sorted(matches, key=lambda x: x.distance)
* 显示匹配结果
	* img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,flags=2)
* k对最佳匹配：一个特征点可以对应k个特征点
	* matches = bf.knnMatch(des1, des2, k=2) 
	* 显示匹配结果
	* good = []
	* for m, n in matches:
    	* if m.distance < 0.75 * n.distance:
        * good.append([m])
    * img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

#### 8.3.2 随机抽样一致算法（Random sample consensus，RANSAC）
对比了最小二乘法和RANSAC的优劣，随机抽样一致算法可以忽略噪声的影响，拟合出较好的结果。
* 所以也可以抽取出有用的SIFT特征点(just  i understand)
* 获得匹配较好的SIFT特征点
* 可求解相关问题与估计立体摄像机的基础矩阵。
#### 8.3.3 单应性矩阵
可以描述
* 物体在世界坐标系和像素坐标系之间的位置映射关系
* 从通过透视变换实现图像从一种视图变换到另外一种视图

可以解决
* 解决拍照时候图像扭曲问题
* 用来实现图像拼接时候解决对齐问题
* 图像校正、图像拼接、相机位姿估计、视觉SLAM

有数学公式计算，有八个未知数(自由度)，因此至少需要四个点来计算

## 9.项目三：图像拼接
通过SIFT特征点和特征匹配，再使用单应性矩阵来变化多张图片，使其拼接在一起，实现全景图的效果。

主要思路：
* 图像预处理
	* 按比例改变图像大小 
* 计算每个图像的SIFT关键特征点和特征描述子(特征向量)
* 匹配两个图像中的特征点
	* KNN匹配
		* rawMatches = matcher.knnMatch(featureA, featureB, k=2)
	* 根据距离比值筛选特征点
* 获取两张图片对应的特征点后，可以进行计算变化矩阵(单应性矩阵)
	* 使两张图片在同一视角下 
* 获得单应性矩阵后，可以进行视角变换
	*  result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
* 再与另一张图片拼接(添加)
* 即可得到一张全景图

## 10.项目四：停车场车位识别
主要通过传统的图像处理方法提取出每个车位图片，再使用神经网络方法对每个车位图片进行预测，预测是否有车

主要思路：
* 图像预处理
	* 二值处理 
	* 转换为灰度图
	* 再进行Canny边缘检测
*  选择有效的停车场的区域
	*   通过：获取具体停车场图片
		*   cv2.fillPoly(mask, vertices, 255)
		*   cv2.bitwise_and(image, mask)
* 通过霍夫变换检测到直线(停车位)
* 筛选有效的直线(真正的停车位)
	* 通过坐标点之间的距离等筛选 
* 通过有效直线获得具体的停车区域(图像中的一列一列，一排一排)
	*  Step 1: 过滤部分直线
	*  Step 2: 对直线按照x1进行排序
	*  Step 3: 找到多个列，相当于每列是一排车
	*  Step 4: 得到坐标
	*  Step 5: 把列矩形画出来
* 通过停车位坐标和矩形，显示绘制停车区域，并且计算停车位数量，并且返回每个停车位的位置(字典方式即可)
* 通过停车位字典获取(根据坐标截图即可)图片中的每个停车位的图片，并保存
* 加载预训练好的神经网络模型
	* 这里使用微调后的VGG16和MobileNet等均可
*  将模型和停车位字典等传入predict_on_image()函数中，对每个停车位图片进行预测
*  如果预测结果为空则显示出来即可

## 11.项目五：答题卡识别答案

主要思路：
* 图像预处理
	* 转灰度
	* 高斯滤波去噪
	* Canny检测 
* 轮廓检测
	* 轮廓排序 
	* 轮廓近似
		* 为了找到最大的矩形(真正的答题卡的边缘四个点)
		* 为透视变换，计算变化矩阵做准备
* 透视变换
	* 四个点=>四个点的变换 
	* warped = four_point_transform(gray, docCnt.reshape(4, 2)) 
* 二值处理=>轮廓检测
* 筛选出涂题轮廓=>排序分类对应题目
* 遍历每一排(题)
	* 获得每一个选项的mask并且与原二值图做对比，判断是否被涂 
	* mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    * total = cv2.countNonZero(mask)
    * 根据索引比对正确答案，并且绘制正确答案的选项
* 统计全部结果，绘制在图片即可
	*  cv2.putText(...)


## 12.背景建模
主要目的：判断视频(图片)中哪个是目标哪个是背景
* 一般精度较差，速度较慢，现在大多建议使用深度学习方法，如：YOLO
* 帧差法
	* 因为一般目标都是在运动，所以直接对每几帧图片进行差分运行，通过阈值判断，即可实现目标检测的功能
	* 缺点
		*  会引入噪音和空洞问题
* 混合高斯模型
	*  图像中每个背景采用一个混合高斯模型进行模拟，每个背景的混合高斯的个数可以自适应。然后在测试阶段，对新来的像素进行GMM匹配，如果该像素值能够匹配其中一个高斯，则认为是背景，否则认为是前景。由于整个过程GMM模型在不断更新学习中，所以对动态背景有一定的鲁棒性。
	*  理解应是：假设像素点变化符合高斯分布，而背景是一个高斯分布，目标也符合一个高斯分布，因此计算对应的像素点是哪个高斯分布即可获得结果
	* 混合高斯模型学习方法
		- 1.首先初始化每个高斯模型矩阵参数。
		- 2.取视频中T帧数据图像用来训练高斯混合模型。来了第一个像素之后用它来当做第一个高斯分布。
		- 3.当后面来的像素值时，与前面已有的高斯的均值比较，如果该像素点的值与其模型均值差在3倍的方差内，则属于该分布，并对其进行参数更新。
		- 4.如果下一次来的像素不满足当前高斯分布，用它来创建一个新的高斯分布。
	* 混合高斯模型测试方法
		* 在测试阶段，对新来像素点的值与混合高斯模型中的每一个均值进行比较，如果其差值在2倍的方差之间的话，则认为是背景，否则认为是前景。将前景赋值为255，背景赋值为0。这样就形成了一副前景二值图。
* 具体实现
	* 获取视频
		*  cap = cv2.VideoCapture('test.avi')
	* 创建混合高斯模型
		*  fgbg = cv2.createBackgroundSubtractorMOG2()
	* While循环获取每一帧图像
		*   ret, frame = cap.read()
		*   对每一帧进行背景提取，获取掩码(二值图像)
			*  fgmask = fgbg.apply(frame)
		* 再对结果进行去噪
			*  形态学开运算去噪点
			*  fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		* 寻找视频中的轮廓
			*  im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		* 根据轮廓的周长绘制矩阵，来表示检测结果
			*  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)    
		* 显示最终结果
			* cv2.imshow('frame',frame)
    		* cv2.imshow('fgmask', fgmask)

## 13.光流估计
光流是空间运动物体在观测成像平面上的像素运动的“瞬时速度”，根据各个像素点的速度矢量特征，可以对图像进行动态分析，例如目标跟踪。

光流估计就是当给定两帧图像时，下一帧图像和上一帧图像中每一个点有什么不同，而且不同点移动到了什么位置。实现找出人眼所能看到的东西。这个过程是Lucas-Kanade发现的，简称L-K。


**前提条件：**
- 亮度恒定：同一点随着时间的变化，其亮度不会发生改变。

- 小运动：随着时间的变化不会引起位置的剧烈变化，只有小运动情况下才能用前后帧之间单位位置变化引起的灰度变化去近似灰度对位置的偏导数。

- 空间一致：一个场景上邻近的点投影到图像上也是邻近点，且邻近点速度一致。因为光流法基本方程约束只有一个，而要求x，y方向的速度，有两个未知变量。所以需要连立n多个方程求解。

* Lucas-Kanade 算法
	* 有数学公式
	* 因为不一定可逆！所以可以想想：当矩阵特征值都比较大时=>是角点=>可逆，所以做光流估计时，一般都要选择角点来进行处理。

* 具体实现
* cv2.calcOpticalFlowPyrLK():
* 参数：
	- prevImage 前一帧图像
	- nextImage 当前帧图像
	- prevPts 待跟踪的特征点向量
	- winSize 搜索窗口的大小
	- maxLevel 最大的金字塔层数
* 返回：
	- nextPts 输出跟踪特征点向量
	- status 特征点是否找到，找到的状态为1，未找到的状态为0
* 具体视频演示看代码！
* 但是精度和速度均不太行
* 最新基于深度学习的FlowNet2算法可以实时取得state-of-the-art的精度。

## 14.OpenCV中的DNN模块
如何使用OpenCv导入神经网络模型。具体可百度

如：导入Caffe模型和参数

* net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
	"bvlc_googlenet.caffemodel")
* ...

## 15.项目六：目标追踪
通过调用opencv中已经实现内置的追踪算法，对自己选定的目标进行追踪。

### 15.1传统追踪算法：遇到遮挡等情况时，追踪的效果就不太好

主要思路：
* 实例化OpenCV's multi-object tracker(多目标追踪)
	*  trackers = cv2.MultiTracker_create()
* 读取视频流文件
	* 预处理每一帧
	* 选择需要追踪的图像
	* 创建新的追踪器
	* 获得追踪结果
	* 显示追踪结果
### 15.2 使用深度学习的目标检测追踪
通过深度学习(神经网络)进行目标检测，再使用dlib内置算法对检测到的目标进行追踪

* 深度学习目标检测学习建议：
	* Faster-RCNN
	* SSD
	* YOLOv3
	* Mask-RCNN


主要思路：(也可使用多线程来加速)
* 使用SSD检测框架来检测物体
	* 读取网络模型
* 获取视频流
	* 预处理每一帧
* 获取blob数据
	* 进行归一化处理
	* 喂入神经网络中
	* 获得检测目标的结果
	* 过滤目标只取person
* 将person的坐标等传入dlib的追踪器进行追踪
* 显示

## 16.项目七：人脸关键点检测
使用dlib内置的算法和预测器对图像进行人脸检测和关键点定位

主要思路
* 加载人脸检测与关键点定位
	* detector = dlib.get_frontal_face_detector()
	* predictor = dlib.shape_predictor(args["shape_predictor"])
* 图像预处理
	* 按比例改变大小
	* 转灰度图
* 人脸检测
	*   rects = detector(gray, 1)
* 遍历检测到的人脸框
	* 进行关键点定位    
* 显示

## 17.项目八：驾驶疲劳检测
通过检测眼睛的关键点来判断是否疲劳，判断眼睛的变化可以看这篇[论文](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

主要思路：
* 加载人脸检测与关键点定位
	* detector = dlib.get_frontal_face_detector()
	* predictor = dlib.shape_predictor(args["shape_predictor"])
* 分别取两个眼睛区域
* 读取视频流
	* 图像预处理
	* 按比例改变大小
	* 转灰度图
* 人脸检测
	*   rects = detector(gray, 0)
* 遍历检测到的人脸框
	* 获取眼睛关键点    
	* 计算对应的比值
	* 绘制眼睛区域
	* 判断是否眨眼
* 显示 
