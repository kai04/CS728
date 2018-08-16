
import numpy as np
import tensorflow as tf

global num_features,num_tags,batch_size,glove_file_name,alpha
global dev_epoch_step,test_epoch_step,total_epochs
global trueDevFile,trueTestFile,predDevFile,predTestFile
global glove_dict,glove,unk

num_features = 200 #should be equal to glove file dimension
num_tags = 8
batch_size = 64
alpha = 0.01
glove_file_name = 'glove.6B.200d.txt' #'glove.840B.300d.txt' #'glove.6B.200d.txt' #'glove.6B.50d.txt'
dev_epoch_step = 5
test_epoch_step = 10
total_epochs = 50

global trainfile_name,devfile_name,testfile_name

trueDevFile = 'true.dev.txt'
predDevFile = 'pred.dev.txt'
trueTestFile = 'true.test.txt'
predTestFile = 'pred.test.txt'

def max_words(data):
    temp_list = [len(i) for i in data]
    return max(temp_list)
    
def func(glove,glove_dict,line,unk,num_words):
    l2 = list()
    count = num_words
    for word in line:
        if word.lower() in glove:
            l2.append(glove_dict[word.lower()])
        else:
            l2.append(glove_dict['unk'])
        count -= 1
    while count > 0:
        l2.append(glove_dict['unk'])
        count -= 1        
    return l2
            
def func2(ner_set,line,num_words):
    l2 = list()
    count = num_words
    for word in line:
        l2.append(ner_set[word])
        count -= 1
    while count > 0:
        l2.append(ner_set['O'])
        count -= 1
    return l2
    
def do_glove():
            
    unk = np.random.rand(1, num_features).astype(np.float32)
    unk = unk.reshape(num_features)
    unk = list(unk)
    
    glove = set()
    
    glove_vectors = open(glove_file_name,'r')
    #glove_vectors = open('glove_anand.txt','r')
    vectors = glove_vectors.readlines()    
    glove_dict = dict()
    
    print("Number of lines in GloVe: ",len(vectors))
    for i in range(0,len(vectors),1):
        vector = vectors[i]
        vector = vector.strip()
        ls = vector.split(' ')
        word = ls[0]
        word = word.lower()
        ls = ls[1:]
        
        for z in range(0,len(ls),1):
            ls[z] = float(ls[z])
            
        glove_dict[word] = ls    
        glove.add(word)
        
        if i % 100000 == 0:
            print i
            
    glove_dict['unk'] = unk
    return glove_dict,glove,unk

glove_dict,glove,unk = do_glove()
print(len(glove_dict.keys()),len(glove))

def do_processing(file_path):
    
    f = open(file_path,'r')
    lines = f.readlines()
    print (file_path,len(lines))
    
    vocab = set()
    data = list()
    sentence = list()
    sentence_ner = list()
    count = 0
    ner_set = dict()
    ner_list = list()
    pos_set = set()
    chunk_set = set()
    
    for i in range(0,len(lines),1):
        line = lines[i]
        
        if len(line.strip()) :        
            word, pos, chunk_tag, ner = line.strip().split(' ')
        #if word == '-DOCSTART-' :
         #   continue
        
        if len(line.strip()) == 0 : #blank line
            data.append(sentence)
            ner_list.append(sentence_ner)
            sentence = list()
            sentence_ner = list()  
            continue
        
        sentence.append(word.lower())
        sentence_ner.append(ner)
        if ner not in ner_set.keys():
            ner_set[ner] = count
            count += 1
        
        pos_set.add(pos)
        vocab.add(word.lower())
        chunk_set.add(chunk_tag)
    
    return data,vocab,ner_set,ner_list
    
#def make_data(glove,glove_dict,data,ner_list,ner_set,unk):
def make_data(data,ner_list,ner_set,low,high):

    #num_examples = len(data)
    #num_tags = 8 # 0 to 5
    #num_features = 300
    num_words = max_words(data)
            
    l3 = list()
    l2 = list()
    
    for i in range(low,high,1):
        line = data[i]
        l2 = func(glove,glove_dict,line,unk,num_words)
        l3.append(l2)
    
    X = np.array(l3).astype(np.float32)
        
    l3 = list()
    l2 = list()
    for i in range(low,high,1):
        line = ner_list[i]
        l2 = func2(ner_set,line,num_words)
        l3.append(l2)
            
    y = np.array(l3).astype(np.int32)
    
    l1 = list()
    for i in range(low,high,1):
        l1.append(len(data[i]))
    
    sequence_length = np.array(l1).astype(np.int32)

    #print (X.shape,y.shape,sequence_length.shape)
    return X,y,sequence_length

'''def do_train(train_X,train_y,train_sequence_length,train_num_words,train_data,
             dev_X,dev_y,dev_sequence_length,dev_num_words,dev_data,
             test_X,test_y,test_sequence_length,test_num_words,test_data):'''
def do_train(train_num_words,train_data,train_ner_list,train_ner_set,
             dev_num_words,dev_data,dev_ner_list,dev_ner_set,
             test_num_words,test_data,test_ner_list,test_ner_set):

    print("In training")  
    
    #x_size = train_X.shape
    #y_size = train_y.shape
    #seq_size = train_sequence_length.shape
    #print(x_size,y_size,seq_size)
    
    X = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.int32)
    init = tf.placeholder(tf.int32, shape=(), name="init")
    init_max_words = tf.placeholder(tf.int32, shape=(), name="init2")
    #sequence_lengths = np.zeros(batch_size,dtype=np.int32) #train_sequence_length #tf.placeholder(tf.int32)
    sequence_lengths = tf.placeholder(tf.int32)
            
            
    weights = tf.get_variable("weights", [num_features, num_tags])     
    # Compute unary scores from a linear layer.       
    matricized_x_t = tf.reshape(X, [-1, num_features])
    matricized_unary_scores = tf.matmul(matricized_x_t, weights)
    
    unary_scores = tf.reshape(matricized_unary_scores, [init, init_max_words, num_tags])

    # Compute the log-likelihood of the gold sequences and keep the transition
    # params for inference at test time.
    print("**************1")
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        unary_scores, y, sequence_lengths)

    # Compute the viterbi sequence and score.
    print("**************2")
    viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode( unary_scores, transition_params, sequence_lengths)

    # Add a training op to tune the parameters.
    loss = tf.reduce_mean(-log_likelihood)
    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)       
    
    print("Begin Session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    print("Begin Epochs")
    
    # Train for a fixed number of iterations.
    train_acc = list()
    dev_acc = list()
    test_acc = list()

    for epoch in range(total_epochs):
        print("Running Epoch: ",epoch)
        num_of_batches = int(len(train_data)/batch_size)
        correct = 0
        total = 0
        for i in range(0,num_of_batches+1,1):
            
            if i == num_of_batches:
                last = len(train_data) % batch_size
                low = len(train_data) - last
                high = len(train_data)
            else:
                last = batch_size
                low = last * i
                high = min((i+1)*last,len(train_data))
             
            
            next_x,next_y,next_seq_length = make_data(train_data,train_ner_list,train_ner_set,low,high)            
            #next_x = train_X[low:high,:]
            #next_y = train_y[low:high,:]
            #next_seq_length = train_sequence_length[low:high]
            #print(i,low,high,next_x.shape,next_y.shape,next_seq_length.shape)
            tf_viterbi_sequence, _ = sess.run([viterbi_sequence, train_op], 
                                              feed_dict={X: next_x, 
                                                         y: next_y,
                                                         sequence_lengths:next_seq_length,
                                                         init:last,
                                                         init_max_words:train_num_words})
            
            
            for z in range(0,last,1):
                sentence_length = next_seq_length[z]
                true_labels = next_y[z][:sentence_length]
                predicted_labels = tf_viterbi_sequence[z][:sentence_length]
                correct += np.sum(true_labels == predicted_labels)
                total += sentence_length
                #print(total)
        Acc = 100.0 * correct/float(total)
        train_acc.append((epoch,Acc))
        print(epoch,correct,total,Acc)
        print("*********************")
        
        #Test of Dev Set
        if epoch % dev_epoch_step == 0:
            
            print("Testing On Dev data: ",epoch)
            num_of_batches = int(len(dev_data)/batch_size)
            correct = 0
            total = 0
            for i in range(0,num_of_batches+1,1):
                
                if i == num_of_batches:
                    last = len(dev_data) % batch_size
                    low = len(dev_data) - last
                    high = len(dev_data)
                else:
                    last = batch_size
                    low = last * i
                    high = min((i+1)*last,len(dev_data))
                 
                #next_x = dev_X[low:high,:]
                #next_y = dev_y[low:high,:]
                #next_seq_length = dev_sequence_length[low:high]
                next_x,next_y,next_seq_length = make_data(dev_data,dev_ner_list,train_ner_set,low,high)            
                tf_viterbi_sequence, _ = sess.run([viterbi_sequence, train_op], 
                                                  feed_dict={X: next_x, 
                                                             y: next_y,
                                                             sequence_lengths:next_seq_length,
                                                             init:last,
                                                             init_max_words:dev_num_words})
                
                
                for z in range(0,last,1):
                    sentence_length = next_seq_length[z]
                    true_labels = next_y[z][:sentence_length]
                    predicted_labels = tf_viterbi_sequence[z][:sentence_length]
                    correct += np.sum(true_labels == predicted_labels)
                    total += sentence_length
                    #print(total)
            DevAcc = 100.0 * correct/float(total)
            dev_acc.append((epoch,DevAcc))
            print(epoch,correct,total,DevAcc)
            print("*********************")
        
        #Test on Test Set
        if epoch % test_epoch_step == 0:
            
            print("Testing On Test data: ",epoch)
            num_of_batches = int(len(test_data)/batch_size)
            correct = 0
            total = 0
            for i in range(0,num_of_batches+1,1):
                
                if i == num_of_batches:
                    last = len(test_data) % batch_size
                    low = len(test_data) - last
                    high = len(test_data)
                else:
                    last = batch_size
                    low = last * i
                    high = min((i+1)*last,len(test_data))
                 
                
                #next_x = test_X[low:high,:]
                #next_y = test_y[low:high,:]
                #next_seq_length = test_sequence_length[low:high]
                #print(i,low,high,next_x.shape,next_y.shape,next_seq_length.shape)
                next_x,next_y,next_seq_length = make_data(test_data,test_ner_list,train_ner_set,low,high)            
                tf_viterbi_sequence, _ = sess.run([viterbi_sequence, train_op], 
                                                  feed_dict={X: next_x, 
                                                             y: next_y,
                                                             sequence_lengths:next_seq_length,
                                                             init:last,
                                                             init_max_words:test_num_words})
                
                
                for z in range(0,last,1):
                    sentence_length = next_seq_length[z]
                    true_labels = next_y[z][:sentence_length]
                    predicted_labels = tf_viterbi_sequence[z][:sentence_length]
                    correct += np.sum(true_labels == predicted_labels)
                    total += sentence_length
                    #print(total)
            TestAcc = 100.0 * correct/float(total)
            test_acc.append((epoch,TestAcc))
            print(epoch,correct,total,TestAcc)
            print("*********************")
            
    #dev_f = open('devfile_name','w')
    #test_f = open('testfile_name','w')
        
    print("Final Testing On Dev data: ")
    num_of_batches = int(len(dev_data)/batch_size)
    correct = 0
    total = 0
    f_true_dev = open(trueDevFile, 'w+')
    f_pred_dev = open(predDevFile,'w+')
    for i in range(0,num_of_batches+1,1):
        
        if i == num_of_batches:
            last = len(dev_data) % batch_size
            low = len(dev_data) - last
            high = len(dev_data)
        else:
            last = batch_size
            low = last * i
            high = min((i+1)*last,len(dev_data))
         
        #next_x = dev_X[low:high,:]
        #next_y = dev_y[low:high,:]
        #next_seq_length = dev_sequence_length[low:high]
        next_x,next_y,next_seq_length = make_data(dev_data,dev_ner_list,train_ner_set,low,high)
        tf_viterbi_sequence, _ = sess.run([viterbi_sequence, train_op], 
                                          feed_dict={X: next_x, 
                                                     y: next_y,
                                                     sequence_lengths:next_seq_length,
                                                     init:last,
                                                     init_max_words:dev_num_words})
        
        
        for z in range(0,last,1):
            sentence_length = next_seq_length[z]
            true_labels = next_y[z][:sentence_length]
            true_labels_ls = list(true_labels)
            temp_true_labels_ls = [str(k) for k in true_labels_ls]                        
            f_true_dev.write(' '.join(temp_true_labels_ls) + '\n')
                
            predicted_labels = tf_viterbi_sequence[z][:sentence_length]
            predicted_labels_ls = list(predicted_labels)
            temp_predicted_labels_ls = [str(k) for k in predicted_labels_ls]
            f_pred_dev.write(' '.join(temp_predicted_labels_ls) + '\n')
            
            correct += np.sum(true_labels == predicted_labels)
            total += sentence_length
            #print(total)
    DevAcc = 100.0 * correct/float(total)
    dev_acc.append((total_epochs,DevAcc))
    print(correct,total,DevAcc)
    print("*********************")
    
    print("Final Testing On Test data: ")
    num_of_batches = int(len(test_data)/batch_size)
    correct = 0
    total = 0
    f_true_test = open(trueTestFile, 'w+')
    f_pred_test = open(predTestFile,'w+')
    for i in range(0,num_of_batches+1,1):
        
        if i == num_of_batches:
            last = len(test_data) % batch_size
            low = len(test_data) - last
            high = len(test_data)
        else:
            last = batch_size
            low = last * i
            high = min((i+1)*last,len(test_data))
         
        
        #next_x = test_X[low:high,:]
        #next_y = test_y[low:high,:]
        #next_seq_length = test_sequence_length[low:high]
        #print(i,low,high,next_x.shape,next_y.shape,next_seq_length.shape)
        next_x,next_y,next_seq_length = make_data(test_data,test_ner_list,train_ner_set,low,high)            
        tf_viterbi_sequence, _ = sess.run([viterbi_sequence, train_op], 
                                          feed_dict={X: next_x, 
                                                     y: next_y,
                                                     sequence_lengths:next_seq_length,
                                                     init:last,
                                                     init_max_words:test_num_words})
        
        
        for z in range(0,last,1):
            sentence_length = next_seq_length[z]
            true_labels = next_y[z][:sentence_length]
            true_labels_ls = list(true_labels)
            temp_true_labels_ls = [str(k) for k in true_labels_ls]                        
            f_true_test.write(' '.join(temp_true_labels_ls) + '\n')
                
            predicted_labels = tf_viterbi_sequence[z][:sentence_length]
            predicted_labels_ls = list(predicted_labels)
            temp_predicted_labels_ls = [str(k) for k in predicted_labels_ls]
            f_pred_test.write(' '.join(temp_predicted_labels_ls) + '\n')
                
            correct += np.sum(true_labels == predicted_labels)
            total += sentence_length
            #print(total)
    TestAcc = 100.0 * correct/float(total)
    test_acc.append((total_epochs,TestAcc))
    print(correct,total,TestAcc)
    print("*********************")
    print(train_acc)
    print("*********************")
    print(dev_acc)
    print("*********************")
    print(test_acc)
    print("*********************")
    
    
    
def process(train_file,dev_file,test_file):    
    
    train_data,train_vocab,train_ner_set,train_ner_list = do_processing(train_file)
    dev_data,dev_vocab,dev_ner_set,dev_ner_list = do_processing(dev_file)
    test_data,test_vocab,test_ner_set,test_ner_list = do_processing(test_file)
    
    train_num_words = max_words(train_data)
    dev_num_words = max_words(dev_data)
    test_num_words = max_words(test_data)
        
    #train_X,train_y,train_sequence_length = make_data(glove,glove_dict,train_data,train_ner_list,train_ner_set,unk)
    #dev_X,dev_y,dev_sequence_length = make_data(glove,glove_dict,dev_data,dev_ner_list,train_ner_set,unk)
    #test_X,test_y,test_sequence_length = make_data(glove,glove_dict,test_data,test_ner_list,train_ner_set,unk)

    
    '''do_train(train_X,train_y,train_sequence_length,train_num_words,train_data,
             dev_X,dev_y,dev_sequence_length,dev_num_words,dev_data,
             test_X,test_y,test_sequence_length,test_num_words,test_data)'''
    
    do_train(train_num_words,train_data,train_ner_list,train_ner_set,
             dev_num_words,dev_data,dev_ner_list,dev_ner_set,
             test_num_words,test_data,test_ner_list,test_ner_set)
    
        
         

if __name__ == '__main__':
    
    train_file = 'data/train.txt'
    dev_file = 'data/dev.txt'
    test_file = 'data/test.txt'
    tf.device('/gpu:2')
    
    vocab = process(train_file,dev_file,test_file)
    #print file_path,file_name