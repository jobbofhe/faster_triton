import matplotlib.pyplot as plt
import numpy as np
import os

# 读取loss日志文件
def read_loss_log(log_path):
    steps = []
    train_losses = []
    val_loss = None
    avg_train_loss = None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 解析训练步骤的loss
        if 'Epoch' in line and 'Step' in line and 'Loss:' in line:
            # 使用更灵活的解析方式
            step_part = line.split('Step')[1].split(',')[0].strip()
            loss_part = line.split('Loss:')[1].strip()
            
            if ':' in step_part:
                step = int(step_part.split(':')[1].strip())
            else:
                step = int(step_part)
            
            loss = float(loss_part)
            steps.append(step)
            train_losses.append(loss)
        # 解析平均训练loss
        elif 'Average Train Loss:' in line:
            avg_train_loss = float(line.split(':')[1].strip())
        # 解析验证loss
        elif 'Validation Loss:' in line:
            val_loss = float(line.split(':')[1].strip())
    
    return steps, train_losses, avg_train_loss, val_loss

# 绘制loss曲线图
def plot_loss_curve(log_path):
    steps, train_losses, avg_train_loss, val_loss = read_loss_log(log_path)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制训练loss曲线
    plt.plot(steps, train_losses, marker='o', markersize=4, linestyle='-', color='b', label='Training Loss')
    
    # 添加平均训练loss和验证loss的参考线
    if avg_train_loss is not None:
        plt.axhline(y=avg_train_loss, color='r', linestyle='--', label=f'Avg Train Loss: {avg_train_loss:.4f}')
    if val_loss is not None:
        plt.axhline(y=val_loss, color='g', linestyle='--', label=f'Validation Loss: {val_loss:.4f}')
    
    # 设置图表标题和标签
    plt.title('Training Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.dirname(log_path)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'loss_curve.svg'), format='svg', bbox_inches='tight')
    
    print(f"Loss curve saved to: {os.path.join(output_dir, 'loss_curve.png')}")
    print(f"Loss curve (SVG) saved to: {os.path.join(output_dir, 'loss_curve.svg')}")

# 主函数
if __name__ == "__main__":
    # log_file_path = '/home/xuxing/dev/ds_acc_tools/training_with_triton/output/training_loss.txt'
    log_file_path = '/home/xuxing/dev/ds_acc_tools/training_with_triton/output/epoch_loss.txt'
    plot_loss_curve(log_file_path)
