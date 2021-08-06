# NLP分词程序







使用的语料如下：

​	

```
词典	:	30wChineseSeqDic_clean.txt  为一个有三十万单词的词典

统计语言模型分词训练语料：	pku_training.utf8  接近两万行的已经分好词的语料
统计语言模型分词测试语料： pku_test.utf8   	两千行的未分词语料
统计语言模型分词黄金语料：  pku_test_gold.utf8  完美分词版本的测试语料

统计语言模型标注训练语料： pku_tag_gold.txt   接近两万行的标注好的语料
统计语言模型标注测试语料：  pku_traning.utf8	 分好词但并未标注的语料
统计语言模型标注黄金语料：	pku_tag_gold.txt  标准的标注语料
```



## 实现过程描述

## 	分词

​	分别实现词典分词和统计语言模型分词

### 		词典分词

​	词典分词的第一步是读入事先准备好的词典，第二步是设计实现给定序列与字典内容进行匹配的算法。

​	为了设计出更简洁，使用更方便的模块，使用了面向对象的设计方法，构造用于词典分词的类 DictionarySegment()

​	读入字典即将字典中所有的词读入一个列表中。

​	在第二步使用逆向最长匹配算法，匹配起点为一句话的最后位置，每次从句子开头向后扫描。

​	本次作业中词典分词源代码如下：

​	

```python
class DictionarySegment():
    '''用于词典分词的类'''
    def __init__(self):
        '''对存储分词结果的数组等初始化'''
        self.word_list=[]
        self.dictsss=[]
    
    def load_dict(self,filename):
        '''读入字典'''
        with open(filename,'r',encoding='utf-8') as fille:
            lines=fille.readlines()
            for line in lines:
                line=line.strip()
                for i in range(len(line)):
                    if line[i]==' ':     #确定空格所在位置，空格后面的内容即为单词
                        posi=i
                        break
                self.dictsss.append(line[:posi])    #将识别到的单词读入数组
    def backward_seg(self,sequence):
        '''对给定的单词序列进行逆向最长匹配分词'''
        i=len(sequence)-1
        while i>=0:     #将扫描位置作为终点
            longest_word=sequence[i]    #记录当前获取到的最长单词
            for j in range(i):
                word=sequence[j:i+1]    #从句子开头扫描到当前位置
                if word in self.dictsss:
                    if len(word)>=len(longest_word):    #更新现在所找到的最长单词
                        longest_word=word
                        break
            self.word_list.insert(0,longest_word)
            i-=len(longest_word)        #减去刚才找到的最长单词的长度
    
    def Segment(self,sentence):
        '''输出分词结果'''
        self.Output=''
        self.backward_seg(sentence)     #先用逆向最长匹配分词
        for i in range(len(self.word_list)):
            self.Output=self.Output+self.word_list[i]+'  '
        
        print(self.Output)

if __name__ == "__main__":
    #小测试    
    testi=DictionarySegment()
    testi.load_dict('30wChineseSeqDic_clean.txt')
    testi.Segment('中国是世界卫生组织的创始国和最早的成员国')

       
```



对于'中国是世界卫生组织的创始国和最早的成员国'测试结果较为准确：

![image-20201205094207644](C:\Users\Stefanny\AppData\Roaming\Typora\typora-user-images\image-20201205094207644.png)





### 隐马尔科夫模型分词

​	统计语言模型，选择了隐马尔科夫模型（Hidden Markov Model）

​	隐马尔科夫模型即为由观察状态集合、隐藏状态集合、初始概率向量，状态转移矩阵以及发射矩阵组成的五元组。

​	使用隐马尔科夫模型进行分词，应先对整个模型的初始概率向量、转移矩阵、发射矩阵三个参数进行训练。

​	之后构造维特比算法，使用训练好的三个参数进行分词。

​	对于分词模型，共有四种隐状态，即'B','M','E','S'。 分别对应一个字在词语中的开始、中间、结尾以及独立成词的字。 而观察模型则为实际输入的为分词的句子序列。

​	初始概率向量对应的是一句话以各种不同状态作为开始的概率，转移矩阵对应的是一个词语当中从一个状态到下一种不同状态的概率，而发射矩阵对应的是一种状态所对应的具体观察状态，即具体汉字的概率。

​	同样出于简洁易用的目的，将分词模型构造成类。在初始化时先构造相应的三个参数所对应的矩阵。

```python

'''用于中文分词'''
class HiddenMarkov():
    '''构造隐马尔科夫类，在类中实现HMM参数训练以及viterbi分词算法'''
    def __init__(self):
        #构造初始概率分布、状态转移矩阵以及发射矩阵'''

        self.meta_prob={}  #初始向量
        self.tran_matrix={} #状态转移矩阵
        self.emit_matrix={} #发射矩阵

        self.pmeta_prob={}  #概率初始向量
        self.ptran_matrix={}#概率转移矩阵
        self.pemit_matrix={}#概率发射矩阵

        self.state_counter={}   #统计BEMS每个状态各自的个数
        self.lable=[]       #用于存储单词标签
        self.observes=[]    #存储分析时单行的观测序列
        self.statuste=[]      #用于存储实际状态序列

        self.vito=[{}]
        self.path={}    #存储路径

        #将矩阵初始化
        self.states=['B','M','E','S']
        for state in self.states:
            self.meta_prob[state]=0             #初始向量即每个句子开头是哪个状态的概率
            self.state_counter[state]=0
            self.emit_matrix[state]={}      #发射矩阵为每个状态所对应的词的统计
            self.tran_matrix[state]={}
            for stat in self.states:
                self.tran_matrix[state][stat]=0.0     #状态转移矩阵为一个状态之后为另一种状态的概率

    def make_lable(self,word):
        #把每个词分成BMES的状态序列
        self.lable=[] 
        if len(word)==1:        #单个字为S
            self.lable.append('S')
            return self.lable
        elif len(word)==2:      #两个字为B  E
            self.lable=['B','E']
            return self.lable
        else:                   #多个字为B  M*(n-2)  E
            self.lable.append('B')
            self.lable.extend('M'*(len(word)-2))
            self.lable.append('E')
            return self.lable

    def get_corpus(self,filename):
        '''获取语料'''
        with open(filename,'r',encoding='utf-8') as corpuz :
            linez=corpuz.readlines()
            for line in linez:
                line=line.strip()   #去掉两边换行符

                #获取观测序列(即每个词对应)
                self.observes=[]
                for i in range (len(line)):
                    if line[i] != ' ':
                        self.observes.append(line[i])
                words=line.split("  ")   #获取一行中的单词   #注意pku_training.utf8里面分割单词用的是两个空格
                #获取实际状态序列
                self.statusste=[]
                for word in words:
                    self.statusste.extend(self.make_lable(word))

                if len(self.observes)>=len(self.statusste): #防止出现观测序列比状态序列长而出现list索引越界的情况
                    #对状态序列计数
                    for i in range(len(self.statusste)):
                        if i==0:            #没个句子的开头即对应初始概率
                            self.meta_prob[self.statusste[0]]+=1
                            self.state_counter[self.statusste[0]]+=1    #对应状态计数器加一
                        else:           
                            self.tran_matrix[self.statusste[i-1]][self.statusste[i]]+=1 #根据前一个状态以及现在的状态添入状态转移矩阵
                            self.state_counter[self.statusste[i]]+=1

                        if self.observes[i] in self.emit_matrix[self.statusste[i]]:         #根绝相应的状态以及字符填入发射矩阵
                            self.emit_matrix[self.statusste[i]][self.observes[i]]+=1
                        else:
                            self.emit_matrix[self.statusste[i]][self.observes[i]]=1
    def prob_calc(self):
        '''将之前的矩阵转换为概率矩阵'''
        SentBeginAmount=sum(self.meta_prob.values())#获取所有的句子开头数量
        for state in self.meta_prob:
            self.pmeta_prob[state]=float(self.meta_prob[state]/SentBeginAmount) #转换初始向量
        
        for state in self.tran_matrix:
            self.ptran_matrix[state]={}         #先对概率状态转移矩阵结构进行配置
            for nextState in self.tran_matrix[state]:
                self.ptran_matrix[state][nextState]=float(self.tran_matrix[state][nextState]/self.state_counter[state]) #转换状态转移矩阵
        
        for state in self.emit_matrix:
            self.pemit_matrix[state]={}
            for word in self.emit_matrix[state]:
                self.pemit_matrix[state][word]=float(self.emit_matrix[state][word]/self.state_counter[state])   #转换发射矩阵

    def viterbi(self,sentence):
        '''使用维特比算法寻找最大序列'''
        self.vito=[{}]
        self.path={}    #存储路径

        #根据概率初始向量初始化
        for state in self.states:
            self.vito[0][state]=self.pmeta_prob[state]*self.pemit_matrix[state].get(sentence[0],0.000001)     
            self.path[state]=[state]        #路径上的第一个值设置为
        
        #进行动态规划
        for t in range(1,len(sentence)):
            self.vito.append({})
            new_path={}
            for statei in self.states:
                possible_path=[]        #存储各种可能的状态转移的概率
                for statej in self.states:  #从i状态到j状态
                    if self.vito[t-1][statej]!= 0:
                        cur_prob=self.vito[t-1][statej]*self.ptran_matrix[statej].get(statei,0)*self.pemit_matrix[statei].get(sentence[t],0.000001) #对于此状态时的转移概率
                        possible_path.append((cur_prob,statej)) #当前的转移状态及其所对应的概率
                if(possible_path):
                    best_cur_path=max(possible_path)        #从状态i到状态j的最大可能路径

                self.vito[t][statei]=best_cur_path[0]   #记录从statei 产生观测字t的概率
                new_path[statei]=self.path[best_cur_path[1]]+[statei]
            self.path=new_path          #更新路径状态
        #print(self.path)
        #通过回溯搜索最优路径
        self.prob, state = max([(self.vito[len(sentence) - 1][state], state) for state in self.states])
        #print(self.vito)
        return state

    def Segment(self,sentence):
        '''将维特比算法求出的最大概率状态序列分词'''
        self.segOutput=''
        state=self.viterbi(sentence)
        #print(sentence)
        #print(self.path[state])
        for i in range(len(self.path[state])):  
            lable=self.path[state][i]
            if lable == 'B' or lable == 'M':      #如果是开头或中间词则直接加入到分词结果中
                self.segOutput+=sentence[i]
            else:              #如果是结束词或者独立成词则在其后面加分隔符
                self.segOutput=self.segOutput+sentence[i]+'  '
        #print(segOutput)
 
       



if __name__ == "__main__":
    test_for_get_corpus=HiddenMarkov()
    test_for_get_corpus.get_corpus('pku_training.utf8')
    test_for_get_corpus.prob_calc()

    with open('date.txt','w') as filee:
        print(test_for_get_corpus.pmeta_prob,file=filee)
        print(test_for_get_corpus.ptran_matrix,file=filee)
        print(test_for_get_corpus.pemit_matrix,file=filee)
    
    test_for_get_corpus.Segment('中国是世界卫生组织的创始国和最早的成员国')
    print(test_for_get_corpus.segOutput)

 
```

可以先查看参数训练的结果：

![image-20201205104730207](C:\Users\Stefanny\AppData\Roaming\Typora\typora-user-images\image-20201205104730207.png)

即以B状态开头的句子的概率是0.63 S开头概率为0.36，M和 E开头概率为0

状态转移矩阵中，以局部举例，M出现后，下一个词出现仍是M的概率为0.34

发射矩阵中，对应状态B的字中，’迈‘字出现的概率为0.00018.



  在维特比算法中，两个核心的矩阵为path 和 vito .其中path存储了以每一种初始状态作为起手的情况下，句子整体的路径序列。vito则是当前观测序列不同隐状态的最大概率。

对于句子'中国是世界卫生组织的创始国和最早的成员国'进行维特比算法后的path 和 vito 分别如下。

```
{'B': ['B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'B', 'E', 'B'], 'M': ['B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'B', 'M', 'M'], 'E': ['B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'B', 'M', 'E'], 'S': ['B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'B', 'E', 'S']}


[{'B': 0.0075607819167175854, 'M': 0.0, 'E': 0.0, 'S': 0.0025562626808956945}, {'B': 1.4842556079810913e-05, 'M': 4.419292363788034e-06, 'E': 0.00011301030833926061, 'S': 1.610185799964932e-06}, {'B': 1.597727525430728e-08, 'M': 4.1629098511264645e-10, 'E': 2.818180730327364e-08, 'S': 1.1013154684032325e-06}, {'B': 1.9678984423342746e-09, 'M': 2.5632269093144483e-12, 'E': 3.716082025091692e-12, 'S': 2.3679336670522635e-11}, {'B': 1.057792501021354e-15, 'M': 3.135012850868495e-13, 'E': 4.954148294560603e-12, 'S': 1.8856607502666206e-16}, {'B': 1.3475765572238024e-15, 'M': 4.389066491622755e-17, 'E': 3.1302399589396744e-17, 'S': 1.937379248352917e-17}, {'B': 7.236079758477432e-20, 'M': 5.72981970065957e-19, 'E': 4.401772779515559e-18, 'S': 2.2646182390473774e-21}, {'B': 4.6764823014141316e-21, 'M': 1.2411142837824196e-22, 'E': 3.259096461185962e-22, 'S': 1.6783320085820486e-22}, {'B': 1.8861828931761467e-27, 'M': 4.564433106737073e-25, 'E': 6.67658777317652e-24, 'S': 1.5931360641784455e-27}, {'B': 2.5392235742834897e-28, 'M': 1.0730842448597787e-28, 'E': 3.185118029897514e-28, 'S': 3.5065191635395908e-25}, {'B': 3.412787653487584e-28, 'M': 7.976406373490212e-33, 'E': 3.602949091909108e-32, 'S': 3.406670081797548e-29}, {'B': 4.759710131561864e-33, 'M': 9.571887258577382e-33, 'E': 3.160082544980113e-31, 'S': 5.425679065030526e-34}, {'B': 1.5960889802401056e-33, 'M': 1.3173229655414898e-35, 'E': 1.0972656204414927e-34, 'S': 2.443771302116103e-34}, {'B': 1.4331076625332035e-37, 'M': 4.082632093212158e-37, 'E': 4.062485253063149e-37, 'S': 2.0970663534959914e-36}, {'B': 2.0390172483376436e-39, 'M': 1.4111319282509325e-43, 'E': 4.58024325944119e-43, 'S': 1.8185869950291524e-39}, {'B': 3.716256095697199e-43, 'M': 1.1437713213382972e-44, 'E': 3.4300795318927592e-43, 'S': 2.3171176896965686e-43}, {'B': 1.3045194798518376e-47, 'M': 3.710598291545082e-47, 'E': 3.3812825976351562e-46, 'S': 1.8014650626429914e-44}, {'B': 4.845575591723288e-47, 'M': 1.7250972119304072e-50, 'E': 8.2299824414361605e-50, 'S': 6.957634516122801e-48}, {'B': 3.2403415165873857e-52, 'M': 2.9627170333463996e-50, 'E': 1.800381491931636e-49, 'S': 6.648702286370331e-53}, {'B': 9.09333543854757e-52, 'M': 4.0774144983068093e-53, 'E': 3.396286913925246e-52, 'S': 1.3922802838908736e-52}]
```

在计算完vito和path之后，通过回溯找到最大概率的初始状态，即可确定最终的初始状态，找到最终的分词序列。

最后一步是调用Segment函数，将分词序列呈现为分好词的一行话。

以下是对于“中国是世界卫生组织的创始国和最早的成员国”的分词结果

![image-20201205103705352](C:\Users\Stefanny\AppData\Roaming\Typora\typora-user-images\image-20201205103705352.png)



## 词性标注



对于词性标注问题（POS-Tagging）同样采用基于HMM的统计标注方法。

对于词性标注问题，使用HMM的参数所代表的含义略有不同，但是核心的使用方法是大体相同的。

此时，初始概率向量代表的是一句话中的第一个词词性为某种特定词性的概率，而状态转移矩阵则代表着一个词为某种词性，而下一个词为某种特定词性的概率。而发射矩阵对应的则是某种词性下，所对应的词为某一特定词语的概率。

不同于分词时的状态序列为单个字，在词性标注时的状态序列为词语，对于输入’我  喜欢  台灯‘，，观察序列长度为3.

知道了参数所代表的意义上的区别，马上就可以使用HMM对语料进行训练。

注：本次标注使用的是PKU标注集



程序代码：

```python
class hiddenMarkovTag():
    '''隐马尔科夫模型用于词性标注'''
    def __init__(self):
        '''先对各项参数初始化'''

        #以下为频数矩阵     #注意构造字典类型
        self.meta_vec={}       #原始概率向量，即各个词性作为句子开头的概率
        self.tran_matrix={}     #转移矩阵，从一个词性转移到另一个词性的概率
        self.emit_matrix={}     #发射矩阵，各个词性所对应特定词语的概率
        self.tag_count={}      #用于统计各个词性出现的次数
        self.tag_statis=[]      #用于记录出现过的tag
        self.all_tags=0

        #以下为概率矩阵
        self.pmeta_vec={}
        self.ptran_matrix={}
        self.pemit_matrix={}

    def makeMatrix(self,filename):
        '''统计词频并填充频数矩阵'''

        sentEnd=[',','。']  #标识句子的结束
        #先读出各个词性的个数
        with open(filename,'r',encoding='utf8') as fuckee:  #先读取文件
            lines=fuckee.readlines()
            count=0
            for line in lines:
                line=line.strip()   #去掉换行符
                words=line.split("  ")  #把词语分割出来
                for word in words:
                    for i in range(len(word)):  #读到斜线分隔符时计数，
                        if word[i]=='/':
                            count=i
                    tag=word[count:]        #将分隔符之后的字符计为标签
                    if tag not in self.tag_statis:
                        self.tag_statis.append(tag)     #如果标签不在list里面就加进去
            
          #  print(self.tag_statis)

            for taige in self.tag_statis:       #在统计频数之前再对每个矩阵的结构进行一下构造
                self.tag_count[taige]=0
                self.meta_vec[taige]=0
                self.tran_matrix[taige]={}
                self.emit_matrix[taige]={}      #发射矩阵再进行处理的时候再具体构造
                for ta in self.tag_statis:
                    self.tran_matrix[taige][ta]=0.0
            
            for line in lines:
                voca_list=[]    #存储本行单词的list
                voca_tag_list=[]    #存储本行词性的list
                line=line.strip()
                group=line.split('  ')  #将一个词加词性看成一个小组
                for member in group:
                    for i in range(len(member)):
                        if member[i]=='/':
                            count=i
                    voca_tag=member[count:]     #分别记录单词与词性
                    voca=member[:count]         
                    voca_list.append(voca)
                    voca_tag_list.append(voca_tag)

                    #之后对矩阵进行填充，对每行的第一个词要进行单独处理
                for i in range(len(voca_list)):   
                    try:
                        self.all_tags+=1
                        if i!=0 and voca_list[i-1] in sentEnd:                        #对句号后面的第一个词进行处理
                            self.tag_count[voca_tag_list[i]]+=1
                            self.meta_vec[voca_tag_list[i]]+=1
                            if voca_list[i] in self.emit_matrix[voca_tag]:      #处理发射矩阵
                                self.emit_matrix[voca_tag_list[i]][voca_list[i]]+=1
                            else:
                                self.emit_matrix[voca_tag_list[i]][voca_list[i]]=1
                        else:
                            self.tag_count[voca_tag_list[i]]+=1
                            self.tran_matrix[voca_tag_list[i-1]][voca_tag_list[i]]+=1
                            if voca_list[i] in self.emit_matrix[voca_tag]:
                                self.emit_matrix[voca_tag_list[i]][voca_list[i]]+=1
                            else:
                                self.emit_matrix[voca_tag_list[i]][voca_list[i]]=1
                    except KeyError:
                        pass
    
    def prob_calc(self):
        '''将频数矩阵转换为概率矩阵'''
        #处理初始向量
        meta_sun=sum(self.meta_vec.values())
        for tag in self.meta_vec:
            self.pmeta_vec[tag]=self.meta_vec[tag]/meta_sun
        #对状态转移矩阵和发射矩阵，采用加1法对二者进行平滑
        for tag in self.tran_matrix:
            self.ptran_matrix[tag]={}
            #状态转移矩阵 
            for tage in self.tag_statis:
                self.tran_matrix[tag][tage]+=1
                self.ptran_matrix[tag][tage]=float(self.tran_matrix[tag][tage]/self.tag_count[tag])

        for tag in self.emit_matrix:
            self.pemit_matrix[tag]={}   
            #发射矩阵
            for word in self.emit_matrix[tag]:
                self.pemit_matrix[tag][word]=float((self.emit_matrix[tag][word]+1)/self.tag_count[tag])    #进行除法前加1平滑


    def viterbi(self,sequence):
        '''使用维特比算法寻找最大词性序列'''
        self.vito=[{}]
        self.path={}    #存储路径
        self.obj_len=0  #观测序列的长度
        sentence=[]
        self.wordee=[]    
        #输入时以两个空格作为分割
        words=sequence.split('  ')
        for word in words:
            sentence.append(word)      #生成观测序列
            self.wordee.append(word)
            self.obj_len+=1
        #根据概率初始向量初始化
        for state in self.tag_statis:
            self.vito[0][state]=self.pmeta_vec[state]*self.pemit_matrix[state].get(sentence[0],0.000001)     
            self.path[state]=[state]        #路径上的第一个值设置为
        
        #进行动态规划
        for t in range(1,len(sentence)):
            self.vito.append({})
            new_path={}
            for statei in self.tag_statis:
                possible_path=[]        #存储各种可能的状态转移的概率
                for statej in self.tag_statis:  #从i状态到j状态
                    if self.vito[t-1][statej]!= 0:
                        cur_prob=self.vito[t-1][statej]*self.ptran_matrix[statej].get(statei,0)*self.pemit_matrix[statei].get(sentence[t],0.000001) #对于此状态时的转移概率
                        possible_path.append((cur_prob,statej)) #当前的转移状态及其所对应的概率
                if(possible_path):
                    best_cur_path=max(possible_path)        #从状态i到状态j的最大可能路径

                self.vito[t][statei]=best_cur_path[0]   #记录从statei 产生观测字t的概率
                new_path[statei]=self.path[best_cur_path[1]]+[statei]
            self.path=new_path          #更新路径状态
        #print(self.path)
        #通过回溯搜索最优路径
        self.prob, state = max([(self.vito[len(sentence) - 1][state], state) for state in self.tag_statis])
       # print(self.vito)
        return state

    def taGGing(self,sequence):
        '''对输入的序列标注结果输出'''
        self.tagOutput=''
        tagState=self.viterbi(sequence)
        # with open('caonima.txt','w',encoding='utf-8') as mmp:
        #     print(self.path,file=mmp)
        #     print(tagState,file=mmp)
        for i in range(self.obj_len):
            tage=self.path[tagState][i]
            self.tagOutput=self.tagOutput+self.wordee[i]+ tage+'  '
        print(self.tagOutput)

        

if __name__ == "__main__":
    testtt=hiddenMarkovTag()
    testtt.makeMatrix('renri.txt')
  #  print(testtt.tag_statis)
    testtt.prob_calc()
    # with open('caonimb.txt','w',encoding='utf-8') as shabi:
    #     print(testtt.all_tags,file=shabi)
    #     print(testtt.tag_count,file=shabi)
    #     print(testtt.pmeta_vec,file=shabi)
    #     print(testtt.ptran_matrix,file=shabi)
    #     print(testtt.pemit_matrix,file=shabi)
    testtt.taGGing('我  需要  台灯')
```



在进行训练时，还要注意对发射矩阵以及转移矩阵的数据平滑，此实验中采用加一法平滑，即对每一项都加1，再进行频数除以总数计算概率时，将总数加1.

在训练后我们可以看到相关参数

![image-20201205111425265](C:\Users\Stefanny\AppData\Roaming\Typora\typora-user-images\image-20201205111425265.png)



可以看到，以/n为开头词语的概率为11.5% ，而从 /m转移到/v的概率为 6.26%。

对于维特比算法以及标注结果的输出类似于在分词时所使用的维特比算法。



对于句子输入"我  需要  台灯"标注如下

![image-20201205112149257](C:\Users\Stefanny\AppData\Roaming\Typora\typora-user-images\image-20201205112149257.png)



可见结果较为准确。



## 结果指标测试

​	在实验基本完成后进行三个测试：	词典分词结果测试、HMM分词结果测试、HMM词性标注测试。

### 	对词典分词以及HMM分词结果的测试

​	对分词结果的测试共有三个指标，分别是准确率（Precision）、召回率（Recall）与F值（F-measure）。

​	准确率的计算方法是自己准确的标注数量除以自己标注的所有单词数量

​	召回率则为自己准确的标注数量除以黄金文本中词语的总数量

​	F1值为准确率与召回率乘积的2倍除以二者相加之和



​	为了进行评估，先用自己制作的标注程序读入未标注的文本并把标注结果生成到另一个文本中，再将自己标注出来的文本与标准文本（黄金文本）同时打开，逐行进行比较。

​	对于找出正确分词，可以将一句话按照每个词块分别记录其其实和结束位置的二元组。

​	如对句子： “我  喜欢  自然语言”  可以将整个句子存储为[(1,2),(2,4),(4,8)]。同时对自己的分词文本与标准分词文本进行操作，之后遍历自己文本所对应的列表，找出有多少个二元组与标准文本列表中的二元组相同即可。

​	以下为分词评测的代码：

```python
'''用于测评程序的分词指标'''

gold=open('pku_test_gold.utf8','r',encoding='utf-8')    #黄金分割样本
ste_s=open('KDA.txt','r',encoding='utf-8')      #自己分出来的词的文本

#逐行扫描进行计数
#分别要记录，黄金标准的单词总数，正确标注数量，自己分出来的总词数
gold_all=0
ste_all=0
right_seg=0     #正确标注的数量

for lineG,lineS in zip(gold,ste_s):     #同时对两个文件中的行进行处理

    gold_seg=[]     #根据位置将每个词分为一个有位置生成的二元组
    ste_seg=[]

    lineG=lineG.strip()     #去掉换行符
    lineS=lineS.strip()
    
    wordsG=lineG.split('  ')    #将每个单词都存入list中
    wordsS=lineS.split('  ')
    count=1
    for word in wordsG:         #对黄金文本和自己分的文本分别记录位置
        gold_seg.append((count,len(word)+count))
        count+=len(word)
    count=1
    for word in wordsS:
        ste_seg.append((count,len(word)+count))
        count+=len(word)

    #遍历列表进行统计正确标注数量
    for word_tuple in ste_seg:
        if word_tuple in gold_seg:  #同样位置如果出现在黄金标准中则正确数量加1
            right_seg+=1    
    
    #累加黄金标准以及自己文本的单词总数
    gold_all+=len(wordsG)
    ste_all+=len(wordsS)

#print(ste_all,gold_all,right_seg)
Percision=float(right_seg/ste_all)
Recall=float(right_seg/gold_all)
F_measure=float(2*Percision*Recall/(Percision+Recall))

print('Now shows the result about HMM segmentation:')
print('Percision: '+str(Percision))
print('Recall: '+str(Recall))
print('F1: '+str(F_measure))
```



可以看到分词的准确率为74.7%， 召回率为78.4%  F1值为76.5%

![image-20201205164306700](C:\Users\Stefanny\AppData\Roaming\Typora\typora-user-images\image-20201205164306700.png)

对词典分词，准确率83.4% 召回率 82.9%  F1值 83.2%：

![image-20201206091815218](C:\Users\Stefanny\AppData\Roaming\Typora\typora-user-images\image-20201206091815218.png)







### 	对词性标注结果进行测评

​	对词性标注效果的测评则相对简单一些，因为自己生成的文件与标准标注文件中每行的单词数是一样的，标注的结果数量也是相同的，所以可以一对一的将自己的标注结果与标准词性对比，如果相同的则可以视为正确地进行了词性标注。

​	

```python
'''用于测评词性标注的准确率，召回率以及F1'''

#打开黄金文件和自己标注的文件
gold=open('pku_tag_gold.txt','r',encoding='utf-8')
ste_S=open('self_tag_pku_test.txt','r',encoding='utf-8')

#记录黄金标准以及自己标注的总数量以及正确数量
gold_all=0
ste_all=0
tag_right=0

#获取正确标注数量及一对一的判断
for lineG,lineS in zip(gold,ste_S):
    lineG=lineG.strip()
    lineS=lineS.strip()
    #两个文件的标注记录
    tagS=[]
    tagG=[]

    count=0
    reGs=lineG.split('  ')
    reSs=lineS.split('  ')
    #先将每个词的标签都取出来
    for group in reGs:
        for i in range(len(group)):
            if group[i]=='/':
                count=i
        tagG.append(group[count:])
    for group in reSs:
        for i in range(len(group)):
            if group[i]=='/':
                count=i
        tagS.append(group[count:])

    #对标签进行逐一比对检验
    for i in range(len(tagG)):
        try:
            gold_all+=1
            tagg=tagG[i]
            tags=tagS[i]
            if tagg==tags:
                tag_right+=1
        except IndexError:
            continue
    # print('all:'+str(gold_all))
    # print('right:'+str(tag_right))
    # break
gold.close()
ste_S.close()

Precision=float(tag_right/gold_all)

print('Here shows the result of HMM POS-tagging:')
print('Precision:'+str(Precision))

```

测试指标如下：

![image-20201205165047746](C:\Users\Stefanny\AppData\Roaming\Typora\typora-user-images\image-20201205165047746.png)



即准确率为80%。由于此种测评方法中，准确率，召回率以及F1值是一样的，于是只输出了准确率。



## 实现过程中遇到的问题



​	最主要的还是设计维特比算法时候遇到的概念不太理解的问题，之后随着看文章弄明白了维特比算法即采用动态规划的方法，从一句话的初始状态出发，每次都采取概率最大的下一种状态进行记录，这样对一句话完整的计算一遍之后，便存储了从各种不同初始状态到句子结尾的最大概率分布，这时只要采取回溯的方法，确定这个句子最大概率的初始状态，便可以成功得出最佳序列路径。

​	还有在制作初始概率，转移矩阵以及发射矩阵的python表示时，遇到了报错KeyError: ,后来发现是在构造的时候没有先在大词典中插入一个空的内嵌词典项，便导致找不到相应的键。

​	然后在最后测评指标的时候，对于分词正好有对应的测试文本和黄金文本，但是词性标注却要单独制作。

​	于是便对训练语料进行处理，删掉其前面的日期项使其与已有的分词时的训练文本格式一样。简而言之，即为了测试词性标注的准确率，我处理了训练文本，并将分词时使用的训练文本作为词性标注的测试文本使用。

​	

## 程序使用说明 

语料

```


​```
词典	:	30wChineseSeqDic_clean.txt  为一个有三十万单词的词典

统计语言模型分词训练语料：	pku_training.utf8  接近两万行的已经分好词的语料
统计语言模型分词测试语料： pku_test.utf8   	两千行的未分词语料
统计语言模型分词黄金语料：  pku_test_gold.utf8  完美分词版本的测试语料

统计语言模型标注训练语料： pku_tag_gold.txt   接近两万行的标注好的语料
统计语言模型标注测试语料：  pku_traning.utf8	 分好词但并未标注的语料
统计语言模型标注黄金语料：	pku_tag_gold.txt  标准的标注语料
​```


```

​		



​	用于词典分词的类即为DictionarySegment, 先创建一个类，之后使用其中的load_dict()方法加载字典，之后使用Segment() 方法进行分词并输出结果

​			用于HMM分词的类为HiddenMarkovTrain文件中的 HiddenMarkov,先创建一个类，之后调用get_corpus()方法读取训练语料加载频数，之后使用prob_calc()方法将频数矩阵转换为概率矩阵，在训练之后再调用Segment（）方法进行分词并输出结果

​			用于词性标注的类为hiddenMarkovTagging文件中的hiddenMarkovtag,先创建此初始化类，之后调用makeMatrix(filename) 方法，加载训练语料并且生成频数矩阵， 之后同上使用prob_calc()方法转换为概率矩阵，最后使用taGGing()方法对所选取的句子进行分词。

​		Note：以上所有类中的分割以及用于标注的方法都内嵌了维特比算法。

​	

 	

