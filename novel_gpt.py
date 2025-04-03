import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import re
from datasets import load_dataset
import os
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len
        })
        return config
        
    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pos_encoding = np.zeros((self.max_len, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
        
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

@register_keras_serializable()
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, inputs):
        q, k, v = inputs
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention = tf.matmul(q, k, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        
        return output

@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff
        })
        return config
        
    def build(self, input_shape):
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.dff, activation='relu'),
            layers.Dense(self.d_model)
        ])
        self.built = True
        
    def call(self, inputs, training=False):
        attn_output = self.mha([inputs, inputs, inputs])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_model(vocab_size, sequence_length=512, embedding_dim=256, num_heads=8, num_layers=12):
    """创建Transformer模型"""
    inputs = layers.Input(shape=(sequence_length-1,))  # 输入序列长度为511
    
    # 词嵌入层
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    
    # 位置编码
    x = PositionalEncoding(embedding_dim, sequence_length-1)(x)
    
    # 增加dropout
    x = layers.Dropout(0.1)(x)
    
    # Transformer编码器层
    for _ in range(num_layers):
        # 多头自注意力层
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads
        )(x, x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # 前馈网络
        ffn_output = layers.Dense(embedding_dim * 4, activation='relu')(x)
        ffn_output = layers.Dense(embedding_dim)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # 增加dropout
        x = layers.Dropout(0.1)(x)
    
    # 输出层（移除softmax激活函数）
    outputs = layers.Dense(vocab_size)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def custom_standardization(input_string):
    """自定义文本标准化函数"""
    # 转换为小写
    input_string = tf.strings.lower(input_string)
    # 替换标点符号
    input_string = tf.strings.regex_replace(input_string, '[！，。？、~@#￥%&*（）]', ' ')
    # 删除多余空格
    input_string = tf.strings.regex_replace(input_string, '\\s+', ' ')
    return input_string

def prepare_dataset():
    """准备数据集"""
    print("正在加载数据集...")
    # 在Colab上直接下载数据集
    dataset = load_dataset("wdndev/webnovel-chinese", split="train[:50]")
    
    # 创建向量化层
    print("正在构建词汇表...")
    vectorize_layer = layers.TextVectorization(
        max_tokens=50000,  # 增加词汇表大小
        output_mode='int',
        output_sequence_length=512,  # 修改序列长度
        standardize=custom_standardization
    )
    
    # 提取文本内容
    texts = []
    print("开始处理文本，共", len(dataset), "本小说...")
    for i, novel in enumerate(dataset, 1):
        print(f"正在处理第 {i} 本小说...")
        # 处理前50000个字符
        text = novel['text'][:50000]
        # 标准化文本
        text = custom_standardization(text).numpy().decode('utf-8')
        # 按段落分割
        paragraphs = re.split(r'\n+', text)
        # 处理每个段落
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) > 50:  # 增加最小段落长度
                # 将长段落分割成更小的片段
                if len(paragraph) > 500:  # 增加最大段落长度
                    # 每500个字符分割一次
                    for j in range(0, len(paragraph), 500):
                        segment = paragraph[j:j+500]
                        if len(segment) > 50:
                            texts.append(segment)
                else:
                    texts.append(paragraph)
        print(f"第 {i} 本小说处理完成，当前共有 {len(texts)} 个文本片段")
    
    print(f"总共处理了 {len(texts)} 个文本片段")
    
    # 训练向量化层
    vectorize_layer.adapt(texts)
    
    # 保存词汇表
    vocabulary = vectorize_layer.get_vocabulary()
    with open('vocabulary.txt', 'w', encoding='utf-8') as f:
        for word in vocabulary:
            f.write(word + '\n')
    print(f"词汇表大小：{len(vocabulary)}")
    
    # 创建训练数据集
    print("正在创建训练数据集...")
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    dataset = dataset.map(lambda x: vectorize_layer(x))
    
    # 创建输入-目标对
    def create_input_target(sequence):
        input_text = sequence[:-1]  # 去掉最后一个token
        target_text = sequence[1:]  # 从第二个token开始
        return input_text, target_text
    
    dataset = dataset.map(create_input_target)
    dataset = dataset.cache()  # 缓存数据集
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32, drop_remainder=True)  # 减小batch size
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 预取数据
    
    return dataset, vectorize_layer, len(vocabulary)

def train_model():
    """训练模型"""
    # 准备数据集
    dataset, vectorize_layer, vocab_size = prepare_dataset()
    
    # 创建模型
    model = create_model(
        vocab_size=vocab_size,
        sequence_length=512,
        embedding_dim=256,
        num_heads=8,
        num_layers=12
    )
    
    # 编译模型
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=CustomSchedule(256),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
        weight_decay=0.01
    )
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)
    
    # 创建检查点
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
        save_best_only=True,
        monitor='loss'
    )
    
    # 早停策略
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )
    
    # 训练模型
    history = model.fit(
        dataset,
        epochs=50,  # 增加训练轮数
        steps_per_epoch=1000,
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    return model, vectorize_layer, history

def generate_text(model, vectorize_layer, prompt, max_length=1000, temperature=0.7, top_p=0.9):
    """生成文本"""
    # 将提示文本转换为序列
    input_sequence = vectorize_layer(prompt)
    
    # 获取词汇表
    vocabulary = vectorize_layer.get_vocabulary()
    
    # 生成文本
    generated_text = prompt
    current_sequence = input_sequence
    
    # 添加标点符号列表
    punctuation = ['。', '！', '？', '，', '；', '：', '"', '"', ''', ''', '（', '）']
    
    for _ in range(max_length):
        # 获取预测
        predictions = model.predict(current_sequence[np.newaxis, :], verbose=0)
        predictions = predictions[0, -1, :]
        
        # 应用温度
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        # 使用nucleus sampling（top-p采样）
        sorted_indices = np.argsort(predictions)[::-1]
        cumulative_probs = np.cumsum(predictions[sorted_indices])
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        predictions[indices_to_remove] = 0
        predictions = predictions / np.sum(predictions)
        
        # 采样下一个token
        next_token = np.random.choice(len(predictions), p=predictions)
        next_word = vocabulary[next_token]
        
        # 如果生成了标点符号，增加换行的概率
        if next_word in punctuation and np.random.random() < 0.3:
            generated_text += next_word + '\n'
        else:
            generated_text += next_word
        
        # 更新序列
        current_sequence = np.append(current_sequence[1:], next_token)
        
        # 如果生成了结束标记或达到最大长度，就停止生成
        if next_word == '[END]' or len(generated_text) >= max_length:
            break
    
    return generated_text

def main():
    # 准备数据集和向量化层
    print("正在准备数据集...")
    train_ds, vectorize_layer, vocab_size = prepare_dataset()
    
    # 构建模型
    print("\n构建模型...")
    model = create_model(
        vocab_size=vocab_size,
        sequence_length=512,  # 使用默认的序列长度
        embedding_dim=256,
        num_heads=8,
        num_layers=12
    )
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 保持from_logits=True
        metrics=['accuracy']
    )
    
    # 设置回调
    callbacks = [
        ModelCheckpoint(
            'novel_gpt.weights.h5',
            save_best_only=True,
            monitor='loss',
            mode='min',
            save_weights_only=True
        ),
        EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        train_ds,
        epochs=5,
        callbacks=callbacks,
        steps_per_epoch=200
    )
    
    print("\n模型训练完成！")

if __name__ == "__main__":
    main() 