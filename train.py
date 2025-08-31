import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import itertools  # 用于处理不同长度的数据加载器

import my_datasets
from model import mymodel
import common

def compute_accuracy(outputs, targets):
    """计算准确率"""
    # 获取每个字符位置的预测结果
    _, predicted = torch.max(outputs, dim=2)  # [batch, captcha_size]
    
    # 检查整个验证码是否完全正确
    correct = (predicted == targets).all(dim=1).float()
    return correct.mean().item()

# 梯度反转层实现域适应
class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# 域分类器模块
class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 2)  # 二分类：源域 vs 目标域
        )
    
    def forward(self, x, alpha=1.0):
        x = GradientReverse.apply(x, alpha)
        return self.classifier(x)

if __name__ == '__main__':
    # 设置随机种子确保可复现性
    torch.manual_seed(20250603)
    np.random.seed(20250603)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集
    train_dataset = my_datasets.mydatasets("./dataset/train")
    test_dataset = my_datasets.mydatasets("./dataset/test")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 初始化模型
    model = mymodel().to(device)
    print(f"Model architecture:\n{model}")
    
    # 获取特征维度（根据您的模型结构调整）
    feature_dim = 512  # 请根据您的模型实际特征维度修改
    
    # 初始化域分类器
    domain_classifier = DomainClassifier(feature_dim).to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    domain_criterion = nn.CrossEntropyLoss().to(device)
    
    # 优化器配置
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': domain_classifier.parameters(), 'lr': 0.01}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 日志记录
    writer = SummaryWriter("logs")
    
    # 训练参数
    num_epochs = 80
    best_acc = 0.0
    start_time = time.time()
    
    # 域适应参数
    max_alpha = 1.0  # 梯度反转的最大强度
    min_alpha = 0.0  # 梯度反转的最小强度
    alpha = min_alpha  # 当前梯度反转强度
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        domain_classifier.train()
        
        running_loss = 0.0
        train_correct = 0
        total_samples = 0
        domain_loss_total = 0.0
        domain_acc_total = 0.0
        domain_count = 0
        
        # 逐步增加域适应的强度
        alpha = min_alpha + (max_alpha - min_alpha) * epoch / (num_epochs - 1)
        
        # 创建迭代器来处理不同长度的数据加载器
        train_iter = iter(train_loader)
        test_iter = iter(test_loader)
        
        # 使用较长的数据加载器长度
        max_steps = max(len(train_loader), len(test_loader))
        
        for step in range(max_steps):
            # 获取源域数据（训练集）
            try:
                src_images, src_labels = next(train_iter)
                src_images = src_images.to(device)
                src_labels = src_labels.to(device)
            except StopIteration:
                train_iter = iter(train_loader)
                src_images, src_labels = next(train_iter)
                src_images = src_images.to(device)
                src_labels = src_labels.to(device)
            
            # 获取目标域数据（测试集）
            try:
                tgt_images, tgt_labels = next(test_iter)
                tgt_images = tgt_images.to(device)
                # 目标域标签仅用于评估，不用于域分类器训练
            except StopIteration:
                test_iter = iter(test_loader)
                tgt_images, tgt_labels = next(test_iter)
                tgt_images = tgt_images.to(device)
            
            # 跳过无效样本
            if src_images.size(0) == 0 or tgt_images.size(0) == 0:
                continue
            
            # ===== 1. 提取特征 =====
            src_features = model.extract_features(src_images)  # 假设您的模型有extract_features方法
            tgt_features = model.extract_features(tgt_images)
            
            # ===== 2. 主任务训练（源域） =====
            src_outputs = model.classify(src_features)  # 假设您的模型有classify方法
            
            # 计算主任务损失
            cls_loss = criterion(
                src_outputs.view(-1, src_outputs.size(-1)), 
                src_labels.view(-1)
            )
            
            # ===== 3. 域分类任务 =====
            # 合并源域和目标域特征
            combined_features = torch.cat([src_features, tgt_features], dim=0)
            
            # 创建域标签：0表示源域，1表示目标域
            domain_labels = torch.cat([
                torch.zeros(src_features.size(0)), 
                torch.ones(tgt_features.size(0))
            ], dim=0).long().to(device)
            
            # 域分类预测
            domain_preds = domain_classifier(combined_features, alpha)
            domain_loss = domain_criterion(domain_preds, domain_labels)
            
            # ===== 4. 计算总损失并反向传播 =====
            # 总损失 = 主任务损失 + 域分类损失
            total_loss = cls_loss + domain_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(domain_classifier.parameters(), 5.0)
            optimizer.step()
            
            # 统计信息
            running_loss += cls_loss.item()
            domain_loss_total += domain_loss.item()
            
            # 计算训练准确率
            with torch.no_grad():
                batch_acc = compute_accuracy(src_outputs, src_labels)
                train_correct += batch_acc * src_images.size(0)
                total_samples += src_images.size(0)
            
            # 计算域分类准确率
            with torch.no_grad():
                _, domain_predicted = torch.max(domain_preds, 1)
                domain_acc = (domain_predicted == domain_labels).float().mean().item()
                domain_acc_total += domain_acc
                domain_count += 1
            
            # 每100批次打印一次信息
            if (step + 1) % 100 == 0:
                avg_loss = running_loss / (step + 1)
                train_acc = train_correct / total_samples if total_samples > 0 else 0.0
                avg_domain_loss = domain_loss_total / domain_count
                avg_domain_acc = domain_acc_total / domain_count
                
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{max_steps}], "
                      f"Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, "
                      f"Domain Loss: {avg_domain_loss:.4f}, Domain Acc: {avg_domain_acc:.4f}, "
                      f"Alpha: {alpha:.4f}")
        
        # 计算平均训练损失和准确率
        avg_train_loss = running_loss / max_steps if max_steps > 0 else 0.0
        train_acc = train_correct / total_samples if total_samples > 0 else 0.0
        avg_domain_loss = domain_loss_total / domain_count if domain_count > 0 else 0.0
        avg_domain_acc = domain_acc_total / domain_count if domain_count > 0 else 0.0
        
        # 记录域适应指标
        writer.add_scalar('Domain/Loss', avg_domain_loss, epoch)
        writer.add_scalar('Domain/Accuracy', avg_domain_acc, epoch)
        writer.add_scalar('Domain/Alpha', alpha, epoch)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                if images.size(0) == 0:
                    continue
                
                outputs = model(images)
                
                # 计算损失
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    labels.view(-1)
                )
                val_loss += loss.item()
                
                # 计算准确率（忽略大小写）
                predicted_labels = torch.argmax(outputs, dim=2)
                correct = (predicted_labels == labels).all(dim=1).float()
                val_correct += correct.sum().item()
                total_val += images.size(0)
        
        # 计算验证指标
        avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        val_acc = val_correct / total_val if total_val > 0 else 0.0
        
        # 打印epoch结果
        print(f"Epoch [{epoch+1}/{num_epochs}] => "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 记录到TensorBoard
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr} to {new_lr}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'domain_classifier_state_dict': domain_classifier.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch
            }, "best_model.pth")
            print(f"Saved new best model with accuracy: {best_acc:.4f}")
        
        # 早停机制
        if epoch > 10 and best_acc < 0.1:
            print("Early stopping due to low accuracy")
            break
    
    # 计算总训练时间
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'domain_classifier_state_dict': domain_classifier.state_dict(),
        'best_acc': best_acc,
        'epoch': epoch
    }, "final_model.pth")
    writer.close()
    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")