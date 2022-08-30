`

#TINY IMAGENET EXAMPLE
#create tinyimagenet sequence
make_tinyImagenet_dataset.sh     


##Train with a continual learning method on the tinyimagenet  sequence
python Run_Tiny_ImageNet.py --reg_lambda 0  --regularization_method "LwF" --lr_decay_rate 0.1 --dropr 0 --arch ResNet

##Evaluate and estimate the CL performance
python test_tinyImagenet.py --reg_lambda 0 --regularization_method "LwF" --lr_decay_rate 0.1 --dropr 0 --arch ResNet

##Create the 3 sets for this setting
python construct_inout_sets.py --reg_lambda 0  --regularization_method "LwF" --lr_decay_rate 0.1 --dropr 0  --arch ResNet

##evaluate softmax baseline 
python evaluate_cl_novelty_tinyImagenet.py   --reg_lambda 0.0 --arch ResNet --device cuda:0  --regularization_method LwF --batch_size 64 --lr_decay_rate 0.1 --dropr 0.0 --lr_scheduler plt  --novelty_method Max_Softmax 



#8 task sequence example
## create 8 tasks sequence
python prepare_8datasets.py

##Train with a continual learning method on the 8 task sequence
python Run_8_tasks_seq.py --dropr 0 --no_bias --reg_lambda 0 --regularization_method "LwF" --lr_decay_rate 0.1 --lr 0.0001

##Evaluate and estimate the CL performance
python test_8tasks_seq.py  --dropr 0 --no_bias --reg_lambda 0 --regularization_method "LwF" --lr_decay_rate 0.1 --lr 0.0001

##Create the 3 sets for this setting
python construct_inout_sets_8tasks.py  --dropr 0 --no_bias --reg_lambda 0 --regularization_method "LwF" --lr_decay_rate 0.1 --lr 0.0001

##evaluate softmax baseline 
python evaluate_cl_novelty_8tasks.py  --arch Alex --device cuda:0  --batch_size 200 --lr_decay_rate 0.1 --dropr 0. --lr_scheduler plt --buffer_size 0 --regularization_method LwF  --reg_lambda 0 --novelty_method Max_Softmax 

