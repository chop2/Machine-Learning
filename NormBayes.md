
朴素贝叶斯分类器  
1.基本公式：  
    $P(A|B)P(B) = P(B|A)P(A) $    (1)  
  设输入样本数据为$D=\{(x_{0},y_{0}),(x_{1},y_{1}),..,(x_{n},y_{n}) \}$,其中$x\in X,y \in Y，x_{i}^{(j)}$表示第i个样本中的第j个特征。  
  $x^{j}$可能的取值$x^{(j)} = \{ a_{j1},a_{j2},..,a_{jS_{j}}\},j=1,2,..,S_{j}$,y可能的取值为$y_{i} \in \{c_{0},c_{1},..,c_{k}\}$。   
  公式（1）在样本D空间中的描述为：  
        $P(y_{i}|x) = \frac{P(x|y_{i})P(y_{i})}{P(x)}$ (2)  
  即根据输入的样本x输出对应的y中各个分类的概率值。取概率值最高的分类作为最终的预测结果。由于公式2中的$P(x)$对所有的$y$分类的贡献都一样，那么公式2可以化简为：  
        $P(y_{i}|x) = P(x|y_{i})P(y_{i})$  
  最终的预测结果为：  
        $arg max(P(y_{i} = c_{k}|x)),c_{k} \in {c_{0},c_{1},..,c_{K}}$
    
    
  算法步骤：  
  1).计算先验概率和条件概率：  
  $P(y_{i}) = \frac{\sum_{i=1}^{N}I(y_{i} = c_{k})}{N},k = 1,2,..,K$  
  $P(x^{(j)}=a_{jl}|y_{i} = c_{k}) = \frac{\sum_{i=1}^{N}I(x^{(j)} = a_{jl},y_{i}=c_{k})}{\sum_{i=1}^{N}I(y_{i}=c_{k})}$  
  2).对于给定的实例$x=(x_{(1)},x_{(2)},..,x_{(n)})^T$ 计算：  
  $P(Y=c_{k}) \prod_{j=1}^{n}P(X^{(j)} = x^{(j)}|Y=c_{k}),k=1,2,..,K$  
  3).确定实例$x$的分类：  
  $y = arg max \{P(Y=c_{k}) \prod_{j=1}^{n}P(X^{(j)} = x^{(j)}|P(Y = c_{k}))\}$  


```python
import numpy as np

class NormBayes:
    def __init__(self):
        self.__label = []
        self.__Prob_yi=[]
        self.__Prob_xi=[]
        self.__lamda = 1
    def fit(self,X,Y):
        '''
        @X - input numpy array as features
        @Y - input label
        '''
        self.__calc_prob_yi(Y)
        self.__calc_prob_xi(X,Y)        

    def __calc_prob_yi(self,Y):
        #clac  priori probability
        self.__label = list(set(Y))
        N = Y.shape[0];k = 0
        self.__Prob_yi = np.zeros((len(self.__label)))
        for l in self.__label:
            count = 0
            for n in range(N):
                if(Y[n] == l):
                    count += 1
            self.__Prob_yi[k] = float(count) / N
            #print "(yi = ",l,")= ",self.__Prob_yi[k]
            k += 1
    def __calc_prob_xi(self,X,Y):
        #conditional probability
        y = list(set(Y))
        num_cls = len(y);
        feat_dim = X.shape[1]
        self.__Prob_xi = np.zeros((num_cls,feat_dim))
        for c in range(num_cls): 
            count_yi = self.__count_label(Y,y[c])
            #print "count_yi=",count_yi
            yi_idx = self.__get_data_idx(Y,y[c])
            subX = self.__get_sub_data(X,yi_idx)
            for f in range(feat_dim):
                count_xi = np.count_nonzero(subX[:,f])                
                #print "count_x",f,"= ",count_xi
                self.__Prob_xi[c][f] = float(count_xi) / count_yi
                #print "(ck=",c,"xi=",f,")= ",self.__Prob_xi[c][f]

    def __count_label(self,Y,y):     
        return list(Y).count(y)

    def __get_data_idx(self,Y,y):
        return [i for i,a in enumerate(Y) if a == y]

    def __get_sub_data(self,X,idx=[]):
        data = np.zeros((len(idx),X.shape[1]))
        for i in range(len(idx)):
            data[i] = X[idx[i]]
        return data
    def predict(self,X):
        '''
        @X - single-predict if you input one sample,
            multi-predict if you input serval samples
        @return index of label
        '''
        rows,cols = X.shape
        num_cls = len(self.__label)
        rsp = []
        for r in range(rows):            
            prob_y = np.zeros((num_cls))
            for n in range(num_cls):
                prod = 1.
                for c in range(cols):
                    if(X[r][c] != 0):
                        prod *= self.__Prob_xi[n][c]
                prob_y[n] = prod * self.__Prob_yi[n]
            maxIdx = prob_y.argmax()
            rsp.append((self.__label[maxIdx],prob_y[maxIdx]))
        return rsp
```


例1.练一个贝叶斯分类器并确定$x=(2,S)^T$的类标记y,表中$X^{(1)},X^{(2)}$为特征，取值的集合分别为：$A_{1} \in \{1,2,3\},A_{2} \in \{S,M,L\}$  
$Y$为类标记，$Y \in \{-1,1\}$  

|       | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|  ----   |---|---|---|---|---|---|---|---|---| ---|--- |--- |--- |--- |--- |
|$X^{(1)}$ | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2  | 3  | 3  | 3  | 3  | 3  |
|$X^{(2)}$ | S | M | M | S | S | S | M | M | L | L  | L  | M  | M  | L  | L  |
|$Y$     | -1| -1| 1 | 1 | -1| -1| -1| 1 | 1 | 1  | 1  | 1  | 1  | 1  | -1 |


```python
X = np.array([[1,0,0,1,0,0],[1,0,0,0,1,0],[1,0,0,0,1,0],
             [1,0,0,1,0,0],[1,0,0,1,0,0],[0,1,0,1,0,0],
              [0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,0,1],
              [0,1,0,0,0,1],[0,0,1,0,0,1],[0,0,1,0,1,0],
              [0,0,1,0,1,0],[0,0,1,0,0,1],[0,0,1,0,0,1]])
Y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
#Y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])

#single-predict
#x = np.array([[0,1,0,1,0,0]])
#multi-predict
x = np.array([[0,1,0,1,0,0],[1,0,0,0,1,0],[0,0,1,0,1,0]])

clf = NormBayes()
clf.fit(X,Y)
print clf.predict(x)
```

    [(-1, 0.066666666666666666), (-1, 0.066666666666666666), (1, 0.11851851851851851)]
    

例2.贷款申请样本数据表  

   |  ID  |   年龄  |   有工作   |   有自己的房子  |   信贷情况   |   类别   |
   | ---- | -------- |  ----------  |  --------------  |  ------------  |  --------  |
   |  1  |  青年  |   否   |   否   |   一般   |   否   |
   |  2  |  青年  |   否   |   否   |   好   |   否   |
   |  3  |  青年  |   是   |   否   |   好   |   是   |
   |  4  |  青年  |   是   |   是   |   一般   |   是   |
   |  5  |  青年  |   否   |   否   |   一般   |   否   |
   |  6  |  中年  |   否   |   否   |   一般   |   否   |
   |  7  |  中年  |   否   |   否   |   好   |   否   |
   |  8  |  中年  |   是   |   是   |   好   |   否   |
   |  9  |  中年  |   否   |   是   |   非常好  |   是   |
   |  10  |  中年  |   否   |   是   |   非常好   |   是   |
   |  11  |  老年  |   否   |   是   |   非常好   |   是   |
   |  12  |  老年  |   否   |   是   |   好   |   是   |
   |  13  |  老年  |   是   |   否   |   好   |   是   |
   |  14  |  老年  |   是   |   否   |   非常好   |   是   |
   |  15  |  老年  |   否   |   否   |   一般   |   否   |
   
  试预测：$x = (老年，是，否，一般，是)$ 是否发放贷款


```python
X = np.array([[1,0,0,0,0,1,0,0,0],[1,0,0,0,0,0,1,0,0],[1,0,0,1,0,0,1,0,1],
             [1,0,0,1,1,1,0,0,1],[1,0,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0,0],
             [0,1,0,0,0,0,1,0,0],[0,1,0,1,1,0,1,0,0],[0,1,0,0,1,0,0,1,1],
             [0,1,0,0,1,0,0,1,1],[0,0,1,0,1,0,0,1,1],[0,0,1,0,1,0,1,0,1],
             [0,0,1,1,0,0,1,0,1],[0,0,1,1,0,0,0,1,1],[0,0,1,0,0,1,0,0,0]])
Y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
x = np.array([[0,0,1,1,0,1,0,0,1]])

clf = NormBayes()
clf.fit(X,Y)
print clf.predict(x)
```

    [(1, 0.014631915866483762)]
    
