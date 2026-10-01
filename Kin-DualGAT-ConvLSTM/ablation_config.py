class AblationConfig:
    def __init__(self, use_dif=True, use_pa=True, use_kcl=True):
        self.use_dynamic_interaction_field = use_dif
        self.use_postural_attention = use_pa
        self.use_kinematic_constraint_loss = use_kcl

# 配置字典用于自动化消融实验运行
ablation_variants = {
    "Backbone": AblationConfig(False, False, False),
    "+ DIF only": AblationConfig(True, False, False),
    "+ PA only": AblationConfig(False, True, False),
    "+ KCL only": AblationConfig(False, False, True),
    "+ DIF + PA": AblationConfig(True, True, False),
    "+ DIF + KCL": AblationConfig(True, False, True),
    "+ PA + KCL": AblationConfig(False, True, True),
    "Full Model": AblationConfig(True, True, True),
}

def train_ablation_model(variant_name, config):
    # 此处接入主干训练循环
    # 例如：如果 config.use_dynamic_interaction_field 为 False，
    # 则图网络边的权重退化为无物理信息调制的标准常数归一化。
    print(f"Running training for: {variant_name}")
    # trainer.fit(model(config))