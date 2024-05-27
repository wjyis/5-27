import torch
import torch.nn as nn


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config_path', type= str, default= './config', 
                        help= "config文件的地址")
    parser.add_argument('--expname', type=str, default= 'exp', 
                        help= '实验名称')
    parser.add_argument('--logdir', type= str, default= './log', 
                        help='保存log和ckpt的位置')
    parser.add_argument('--datadir', type= str, default= './dataset', 
                        help= '要训练数据的地址')

    #trainging options

    parser.add_argument('-N_rand', type= int, default= 32*32*4, 
                        help= 'batch size (number of random rays per gradient step) 光束的数量')
    parser.add_argument('--lrate', type= float, default= '5e-4', 
                        help= '学习率')
    parser.add_argument('--lrate_decay', type= int, default= 250, 
                        help= '训练多少轮之后减少学习率')
    parser.add_argument('--chunk', type= int, default= 1024*32,
                        help='并行处理的光线数量，如果溢出则减少')
    parser.add_argument('--netchunk', type= int, default= 1024*64,
                        help= '并行发送的点数')
    parser.add_argument('--no_batching', action= 'store_true',
                        help= '一次只从一张图片中获取随机光线')
    parser.add_argument('--load_pretrain', type=bool, default=False,
                        help = '是否加载之前训练模型')
    parser.add_argument('--model_path', type=str, default='./log/ckpt/',
                        help= '加载的模型地址')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help= '为粗网络重新加载特定权重')
    
    return parser

def train():
    parser = config_parser()
    args = parser.parse_args()

    #load_data

    

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    
    def create_embedding_fn(self):
        embed_fns = []
        #输入维度
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            #保留原始输入
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']   

        # 生成频率带
        if self.kwargs['log_sampling']:
            #对数采样
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            #线性采样
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)     
        
        # 对每个频率和周期函数生成嵌入函数
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def embed(self, inputs):
        #沿着最后一个维度的张量连接
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

#multires限制了最大频率,比如取10,则最大频率为2^(9)
def get_embedder(multires, i= 0):
    if i == -1:
        #不做任何变换返回原值和维度3
        return nn.Identity(),3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos]

    }

    embeder_obj = Embedder(**embed_kwargs)
    embed = lambda x,eo = embeder_obj : eo.embed(x)
    return embed, embeder_obj.out_dim

if __name__ == "__main__":
    embed_fn, out_dim = get_embedder(10)
    inputs = torch.rand(5, 3)  # 5 个 3 维输入
    embedded_inputs = embed_fn(inputs)
    print(f'Output dimension: {out_dim}')
    print(embedded_inputs.shape)



    
    

