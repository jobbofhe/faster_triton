import matplotlib.pyplot as plt
import numpy as np
import os

# 读取epoch loss日志文件
def read_epoch_loss_log(log_path):
    epochs = []
    train_losses = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 解析epoch和平均训练loss
        if 'Epoch' in line and 'Average Train Loss:' in line:
            # 提取epoch号
            epoch_part = line.split('Epoch')[1].split(',')[0].strip()
            epoch = int(epoch_part)
            
            # 提取loss值
            loss_part = line.split('Average Train Loss:')[1].strip()
            loss = float(loss_part)
            
            epochs.append(epoch)
            train_losses.append(loss)
    
    return epochs, train_losses

# 绘制epoch loss曲线图
def plot_epoch_loss_curve(log_path):
    epochs, train_losses = read_epoch_loss_log(log_path)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制训练loss曲线
    plt.plot(epochs, train_losses, marker='o', markersize=6, linestyle='-', color='b', linewidth=2, label='Average Train Loss')
    
    # 在每个数据点上显示loss值
    for i, (epoch, loss) in enumerate(zip(epochs, train_losses)):
        plt.annotate(f'{loss:.4f}', (epoch, loss), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, ha='center', va='bottom')
    
    # 设置图表标题和标签
    plt.title('Epoch Average Train Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 设置x轴刻度
    plt.xticks(epochs)
    
    # 设置y轴范围（稍微扩大一点以便更好地显示数据点）
    y_min = min(train_losses) * 0.99
    y_max = max(train_losses) * 1.01
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.dirname(log_path)
    plt.savefig(os.path.join(output_dir, 'epoch_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'epoch_loss_curve.svg'), format='svg', bbox_inches='tight')
    
    print(f"Epoch loss curve saved to: {os.path.join(output_dir, 'epoch_loss_curve.png')}")
    print(f"Epoch loss curve (SVG) saved to: {os.path.join(output_dir, 'epoch_loss_curve.svg')}")

# 主函数
if __name__ == "__main__":
    log_file_path = '/home/xuxing/dev/ds_acc_tools/training_with_triton/output/epoch_loss.txt'
    plot_epoch_loss_curve(log_file_path)
