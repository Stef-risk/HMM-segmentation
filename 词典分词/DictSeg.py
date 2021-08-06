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
                    if line[i]==' ':                #确定空格所在位置，空格后面的内容即为单词
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
        
        #print(self.Output)

if __name__ == "__main__":
    #小测试    
    testi=DictionarySegment()
    testi.load_dict('30wChineseSeqDic_clean.txt')
    testi.Segment('中国是世界卫生组织的创始国和最早的成员国')

        








