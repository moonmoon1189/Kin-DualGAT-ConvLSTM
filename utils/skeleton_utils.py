import torch


def get_edge_index():
    # 返回简化的25节点解剖学邻接拓扑关系
    # 为保证代码可直接运行，这里构建一个双向线性连接图
    source = torch.arange(0, 24)
    target = torch.arange(1, 25)
    edge_index = torch.stack([source, target], dim=0)

    # 转换为无向图的双向边
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index