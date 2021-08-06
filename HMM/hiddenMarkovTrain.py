
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
       # print(self.path)
        #通过回溯搜索最优路径
        self.prob, state = max([(self.vito[len(sentence) - 1][state], state) for state in self.states])
       # print(self.vito)
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

 