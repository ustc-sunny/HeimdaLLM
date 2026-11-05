# 分类别解释更正

2026-11-05 GitHub发布前核对发现，AG News H5中的标签映射为1→0、2→2、3→3、4→1，混淆矩阵内部顺序为World、Sci/Tech、Sports、Business。先前说明误按World、Sports、Business、Sci/Tech解释分类别召回；已更正文字和机器指标的类别名称。全部总体accuracy、macro-F1、交叉熵、训练曲线、原始混淆矩阵和训练工件不变，不涉及重新训练。

依据：REAL_MATCHED pack_manifest.json的label_vocab，以及text_classification_preprocessor.py按label_vocab映射标签的实现。旧解释仅在本地superseded_label_interpretation/保留，未作为当前GitHub结果发布。
