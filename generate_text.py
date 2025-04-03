import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from novel_gpt import (
    custom_standardization,
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    create_model
)
from tensorflow.keras.layers import TextVectorization

def load_model_and_vectorizer():
    """加载模型和向量化层"""
    try:
        print("正在加载模型...")
        # 创建向量化层
        vectorize_layer = TextVectorization(
            max_tokens=50000,
            output_mode='int',
            output_sequence_length=512,
            standardize=custom_standardization
        )
        
        # 加载词汇表
        print("正在加载词汇表...")
        try:
            with open('vocabulary.txt', 'r', encoding='utf-8') as f:
                vocabulary = [line.strip() for line in f]
            vectorize_layer.set_vocabulary(vocabulary)
            print(f"词汇表大小：{len(vocabulary)}")
            
            # 创建模型
            model = create_model(
                vocab_size=len(vocabulary),
                maxlen=512,
                d_model=256,
                num_heads=8,
                dff=512,
                num_layers=6
            )
            
            # 编译模型
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )
            
            # 加载权重
            model.load_weights('novel_gpt.weights.h5')
            
            return model, vectorize_layer
            
        except FileNotFoundError:
            print("错误：找不到词汇表文件 vocabulary.txt")
            return None, None
            
    except Exception as e:
        print(f"加载模型时出错：{str(e)}")
        return None, None

def generate_text(prompt, model, vectorize_layer, max_length=500, temperature=1.0):
    """生成文本"""
    # 标准化输入文本
    prompt = custom_standardization(prompt).numpy().decode('utf-8')
    
    # 向量化输入文本
    input_tokens = vectorize_layer([prompt])
    input_tokens = input_tokens[:, :-1]  # 确保输入形状为(None, 511)
    
    # 生成文本
    generated_text = prompt
    last_words = []  # 用于检测重复
    consecutive_repeats = 0  # 连续重复次数
    
    while len(generated_text) < max_length:
        # 预测下一个token
        predictions = model.predict(input_tokens, verbose=0)
        
        # 使用温度采样
        predictions = predictions[0][-1]  # 获取最后一个预测
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        # 采样下一个token
        next_token = np.random.choice(len(predictions), p=predictions)
        
        # 获取对应的词
        next_word = vectorize_layer.get_vocabulary()[next_token]
        
        # 检查是否重复
        if next_word in last_words:
            consecutive_repeats += 1
            if consecutive_repeats > 2:
                temperature *= 1.5  # 增加温度以减少重复
        else:
            consecutive_repeats = 0
            temperature = max(1.0, temperature / 1.5)  # 恢复正常温度
        
        # 更新最近使用的词
        last_words.append(next_word)
        if len(last_words) > 10:
            last_words.pop(0)
        
        # 检查是否结束
        if next_word == '[UNK]' or next_word == '[END]':
            break
            
        # 添加新词到生成文本
        generated_text += next_word
        
        # 更新输入
        input_tokens = vectorize_layer([generated_text])
        input_tokens = input_tokens[:, :-1]  # 确保输入形状为(None, 511)
        
    return generated_text

def main():
    # 加载模型和向量化层
    model, vectorize_layer = load_model_and_vectorizer()
    if model is None or vectorize_layer is None:
        print("初始化失败，程序退出")
        return
        
    while True:
        print("\n" + "="*50)
        print("文本生成系统")
        print("="*50)
        
        # 获取用户输入
        prompt = input("\n请输入提示词（直接按回车退出）：")
        if not prompt:
            print("\n感谢使用！再见！")
            break
            
        try:
            max_length = int(input("请输入要生成的文本长度（建议：100-1000）："))
            temperature = float(input("请输入温度值（0.1-2.0，值越大生成越随机）："))
        except ValueError:
            print("输入无效，将使用默认值：长度=300，温度=0.8")
            max_length = 300
            temperature = 0.8
        
        print(f"\n开始生成文本...")
        print(f"提示词: {prompt}")
        print(f"文本长度: {max_length}")
        print(f"温度值: {temperature}")
        
        generated_text = generate_text(
            prompt,
            model,
            vectorize_layer,
            max_length=max_length,
            temperature=temperature
        )
        
        print("\n生成的文本：")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        print(f"生成文本长度：{len(generated_text)}字")
        
        # 询问是否继续
        choice = input("\n是否继续生成文本？(y/n): ")
        if choice.lower() != 'y':
            print("\n感谢使用！再见！")
            break

if __name__ == "__main__":
    main()

    # 示例：生成故事
    prompt = "在一个阳光明媚的早晨"
    print(f"\n使用提示词: {prompt}")
    info = generate_text(
        prompt,
        model,
        vectorize_layer,
        max_length=100,  # 生成更长的文本
        temperature=0.8  # 降低随机性，使输出更保守
    )
    
    # 打印生成的故事
    print("\n生成的故事:")
    print(info[-1]["prompt"])

    # 获取生成的故事
    story = info[-1]["prompt"]

    # 获取最后一个词的预测概率
    word_probs = info[-1]["word_probs"]

    # 获取注意力分数
    attention_scores = info[-1]["atts"] 