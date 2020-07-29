1. 人脸识别侧脸识别数据集制作  
1.1 侧脸数据集原始数据集亚洲名人数据集  
1.2 侧脸偏航角计算算法:FaceYaw(FSANET)demo.py(newest)生成trainImage.txt,trainLabel.txt,valImage.txt,valLabel.txt  
1.3 切分数据集为训练集和验证集splitDataset.py(注意shutil.copytree()的用法细节)  
2. 合成dream算法和ArcFace算法组建新的算法  
2.1 新的算法ArcProfileFace算法   
3. 训练技巧  
3.1 先用小数据集验证超参,具体使用验证集试探  
3.2 torch版本1.1支持,1.2不支持  
3.3 训练方法问题:采用pytorch分布式训练方法,调用  
    sampler = torch.utils.data.distributed.DistributedSampler(...)  
    train_loader = torch.utils.data.DataLoader(...sampler=sampler, pin_memory=True, shuffle=False...)    
    数据集总共9万+ID,280万张图片,单卡训练太慢,多卡分布式训练使用pytorch这个接口函数非常方便,实际训练中epoch/0.5h  
3.4 81/100 epoch Loss 0.088 (mean:0.090)	Prec@1 91.964 (mean:90.692)	Prec@5 95.982 (mean:95.497)
4. 验证集问题  
4.1 evoLve_val_dataset.zip该文件是压缩文件以及有对应的验证集处理代码  
5. 问题  
5.1 亚洲名人数据集存在数据分布不均的问题,即:有的ID数量极少只有几个,有的ID数量特别多上千  
6. 小结  
6.1 人脸识别目前采用ArcFace算法做,猜测可能原因:人脸是刚性的变化小,但是在行人重识别中使用三元组损失函数  

人脸识别正脸识别算法  
1. 在侧脸识别算法的基础上去掉偏航角  
1.1 训练代码: train_ori_apex.py, 配置文件: config_ori.py  
2. 侧脸识别算法  
2.1 训练代码: train_profile_ddp.py, 配置文件:config_ori_ddp.py  
3. 超参使用  
3.1 正脸侧脸在亚洲名人数据集上可通用,取亚洲名人数据集子集验证通过  
4. 侧脸识别算法训练集制作  
4.1 在另一个工程profileFace中,详见readme文件

  
warning  
yaw = torch.tensor(yaw[idx:idx + batch_size])
/home/zhengxiangzhong/work/occlusionFace/util/utils.py:263: UserWarning:   
To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or   
sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).