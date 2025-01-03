cd /home/zhu.3723/code/project/PP_SVRG

python run_svrg.py \
--exp_name SVRG_lr_search \
--optimizer SVRG \
--nn_model MNIST_one_layer \
--dataset MNIST \
--n_epoch 100 \
--batch_size 128 \
--weight_decay 0.0001 \
--lr 0.01
