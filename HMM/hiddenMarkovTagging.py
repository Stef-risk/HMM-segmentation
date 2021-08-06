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

    testtt.taGGing('我  需要  台灯')





        