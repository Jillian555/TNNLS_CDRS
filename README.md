# Collaborative Decision-Reinforced Self-Supervision for Attributed Graph Clustering
PyTorch implementation of the paper "[Collaborative Decision-Reinforced Self-Supervision for Attributed Graph Clustering](https://ieeexplore.ieee.org/abstract/document/9777842)".


# Get started
```Shell
cd ARVGA
python cdrs_arvga.py --ds='cora' --e=152 --e1=180 --e2=200 --e3=300 --epochs=900 --pseudo_num=36 --lr_dec=0.15 --func='col' --max_num=700 --w_sup=1.0
```

# Citation

```BibTeX
@ARTICLE{cdrs,
  author={Pengfei Zhu, Jialu Li, Yu Wang, Bin Xiao, Shuai Zhao, Qinghua Hu},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Collaborative Decision-Reinforced Self-Supervision for Attributed Graph Clustering}, 
  year={2022},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TNNLS.2022.3171583}}
```

