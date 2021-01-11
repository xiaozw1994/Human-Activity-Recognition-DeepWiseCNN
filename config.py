import numpy as np
import os 

save_dir = './data/'
###  save dataset that is processed where is decedied by our needs
save_train_90_size =os.path.join(save_dir, 'train_90_size.npz.npy')
save_train_label_90 = os.path.join(save_dir,'train_label_90.npz.npy')
save_test_90_size =os.path.join(save_dir, 'test_90_size.npz.npy')
save_test_label_90 = os.path.join(save_dir,'test_label_90.npz.npy')
####  90 
save_train_96_size = os.path.join(save_dir, 'train_96_size.npz.npy')
save_train_label_96 = os.path.join(save_dir,'train_label_96.npz.npy')
save_test_96_size = os.path.join(save_dir,'test_96_size.npz.npy')
save_test_label_96 = os.path.join(save_dir,'test_label_96.npz.npy')
####  96
save_train_64_size = os.path.join(save_dir,'train_64_size.npz.npy')
save_train_label_64 = os.path.join(save_dir,'train_label_64.npz.npy')
save_test_64_size = os.path.join(save_dir,'test_64_size.npz.npy')
save_test_label_64 = os.path.join(save_dir,'test_label_64.npz.npy')
#### 64
save_train_48_size = os.path.join(save_dir,'train_48_size.npz.npy')
save_train_label_48 = os.path.join(save_dir,'train_label_48.npz.npy')
save_test_48_size = os.path.join(save_dir,'test_48_size.npz.npy')
save_test_label_48 = os.path.join(save_dir,'test_label_48.npz.npy')
#########48
'''
    np.save("data/margin_loss.npy",local_loss_list)
    np.save("data/squared_error_loss.npy",margin_loss_list)
    np.save("data/total_loss.npy",total_loss_list)
    np.save("data/acc.npy",acc_list)
'''
save_margin_loss = "data/margin_loss.npy"
save_reconstruct_loss = "data/squared_error_loss.npy"
save_total_loss = "data/total_loss.npy"
save_acc = "data/acc.npy"
############################## SAVING TRAINING FILES
class Config90(object):
    def __init__(self):
        self.windows = 90
        self.init_shape = [None,self.windows,3,1]
        self.batch_size = 120
        self.num_label = 6
        self.stddev = 0.01
        self.decay = 0.0005 * self.windows*3
        self.new_decay = 0.0005 * 48 * 3
        ###
        self.init_seq = 0.9
        self.init_sub = 1 - self.init_seq
        ##
        self.lamdaset = 0.5