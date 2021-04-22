from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,LayerNormalization
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# 接embedding层，位置编码
class PositionalEncoding(Layer):
    def __init__(self,model_dim,**kwargs):
        self.model_dim = model_dim
        super(PositionalEncoding, self).__init__(**kwargs)
    def get_angles(self,pos,i,d_model):
        return pos/(np.power(10000, (2 * (i//2)) / np.float32(d_model)))
    def call(self,embedding):
        # 输入的是embedding，所以embedding的行就是当前这句话
        # embedding.shape[0]=数据量
        # embedding.shape[1]=句子长度
        # embedding.shape[2]=词嵌入维度
        sentence_length=embedding.shape[1]
        positional_encoding = np.zeros(shape=(sentence_length,self.model_dim))
        # 计算sin/cos位置编码(论文里有公式，懒得备注了)
        for pos in range(sentence_length):
            for i in range(self.model_dim):
                positional_encoding[pos, i] = self.get_angles(pos,i,self.model_dim)
        positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])  # 用于偶数索引2i
        positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])  # 用于奇数索引2i+1
        return K.cast(positional_encoding, 'float32')
    def get_config(self):
        config = super().get_config()
        config.update({
            'mmodel_dim': self.model_dim,
        })
        return config
    def compute_output_shape(self,input_shape):
        return input_shape

# 将embedding和positional encoding相加
class Add(Layer):
    def __init__(self,**kwargs):
        super(Add, self).__init__(**kwargs)
    # 这里的inputs指embedding+positional encoding
    def call(self, inputs):
        input_a, input_b = inputs
        res = input_a+input_b
        return res
    def compute_output_shape(self, input_shape):
        return input_shape[0]

class ScaledDotProductAttention(Layer):
    def __init__(self,mode,**kwargs):
        assert mode == "encoder" or mode == "decoder", "The parameter 'mode' can only receive two values, 'encoder' and 'decoder'."
        self.masking_num = -2**32
        self.mode = mode
        super(ScaledDotProductAttention, self).__init__(**kwargs)
    # padding mask
    # 将0值位置置为一个极小的负数，使得softmax时该值接近0
    def padding_mask(self, QK):
        padding = tf.cast(tf.equal(QK,0),tf.float32)
        padding *= self.masking_num
        return QK+padding
    # sequence mask(传说中的下三角)
    def sequence_mask(self,QK):
        # 初始化下三角矩阵
        seq_mask = 1-tf.linalg.band_part(tf.ones_like(QK), -1, 0)
        seq_mask *= self.masking_num
        return QK+seq_mask
    # 输入为qkv三个矩阵和一个mask矩阵
    def call(self, inputs):
        assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
        queries, keys, values = inputs
        # 转换为32位
        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')
        # Qk计算
        matmul = tf.matmul(queries,keys,transpose_b=True)
        dk = tf.cast(tf.shape(keys)[-1],tf.float32)
        matmul = matmul / tf.sqrt(dk) # QxK后缩放dk**(0.5)
        # mask层,区别encoder和decoder部分
        if self.mode == "encoder":
            matmul = self.padding_mask(matmul)
        else:
            matmul = self.sequence_mask(matmul)
        softmax_out = K.softmax(matmul)  # SoftMax层
        return K.batch_dot(softmax_out, values) # 最后乘V
    def get_config(self):
        config = super().get_config()
        config.update({
            'masking_num': self.masking_num,
            "mode" : self.mode
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape

class MultiHeadAttention(Layer):
    def __init__(self, heads=8,model_dim=512,mode="encoder",trainable=True,**kwargs):
        self.heads = heads
        self.head_dim = model_dim//heads
        self.mode = mode
        self.trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)
    # 随机初始化Q K V矩阵权重
    def build(self,input_shape):
        self.weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self.heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_queries')
        self.weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self.heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_keys')
        self.weights_values = self.add_weight(
            shape=(input_shape[2][-1], self.heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_values')
        self.shape= input_shape
        super(MultiHeadAttention, self).build(input_shape)
    def call(self, inputs):
        assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
        # 注意，这里传入的qkv并不是真正的qkv，而是上一层的embedding(3个),之后乘权重才是真正的qkv
        queries, keys, values = inputs
        # 初始化
        queries_linear = K.dot(queries, self.weights_queries)
        keys_linear = K.dot(keys, self.weights_keys)
        values_linear = K.dot(values, self.weights_values)
        # 多头切割
        queries_multi_heads = tf.concat(tf.split(queries_linear, self.heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self.heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self.heads, axis=2), axis=0)

        att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]
        attention = ScaledDotProductAttention(mode=self.mode)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self.heads, axis=0), axis=2)
        return outputs
    def get_config(self):
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'heads': self.heads,
            "mode" : self.mode,
            "trainable" : self.trainable
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape

# encoder和decoder都要用到的前向传播
def FeedForwardNetwork(units_dim,model_dim):
    return Sequential([Dense(units_dim, activation='relu'),Dense(model_dim)])

class EncoderLayer(Layer):
    def __init__(self,heads=8,model_dim=512,units_dim=512,epsilon=0.001,drop_rate=0.2,**kwargs):
        self.heads = heads
        self.model_dim = model_dim
        self.multi_head_attention = MultiHeadAttention(self.heads,model_dim=model_dim,mode="encoder")
        self.ff_netword = FeedForwardNetwork(units_dim,model_dim)
        self.layer_norm1 = LayerNormalization(epsilon=epsilon)
        self.layer_norm2 = LayerNormalization(epsilon=epsilon)
        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)
        self.dropout3 = Dropout(drop_rate)
        super(EncoderLayer, self).__init__(**kwargs)
    # traning是个bool
    def call(self,encodings,training=True):
        attn_output = self.multi_head_attention([encodings,encodings,encodings])
        attn_output = self.dropout1(attn_output,training=training)
        out1 = self.layer_norm1(encodings + attn_output)

        ffn_output = self.ff_netword(out1)
        ffn_output = self.dropout2(ffn_output,training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        out3 = self.dropout3(out2)
        return out3
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'model_dim': self.model_dim,
            'heads': self.heads,
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape


class Encoder(Layer):
    def __init__(self,num_layers,heads=8,model_dim=512,drop_rate=0.2,units_dim=512,epsilon=0.001,**kwargs):
        self.model_dim = model_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = Dropout(drop_rate)
        self.encodings_layer = PositionalEncoding(model_dim=self.model_dim)
        self.enc_layers = [EncoderLayer(model_dim=model_dim,heads=heads,units_dim=units_dim,drop_rate=drop_rate,epsilon=epsilon)
                           for _ in range(num_layers)]
        super(Encoder,self).__init__(**kwargs)
    def call(self,inputs,training=True):
        encodings = self.encodings_layer(inputs)
        encodings = Add()([inputs,encodings])
        outputs = self.dropout(encodings,training=training)
        for i in range(self.num_layers):
            outputs = self.enc_layers[i](outputs,training=training)
        return outputs
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'model_dim': self.model_dim,
            'heads': self.heads,
        })
        return config

