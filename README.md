# Collaborative Decision-Reinforced Self-Supervision for Attributed Graph Clustering
PyTorch implementation of the paper "[Collaborative Decision-Reinforced Self-Supervision for Attributed Graph Clustering](https://ieeexplore.ieee.org/abstract/document/9777842)".


# Get started
```Shell
cd ARVGA
python cdrs_arvga.py --ds='cora' --e=152 --e1=180 --e2=200 --e3=300 --epochs=900 --pseudo_num=36 --lr_dec=0.15 --func='col' --max_num=700 --w_sup=1.0
```

# Citation

```BibTeX
@article{cdrs,
  author       = {Pengfei Zhu and
                  Jialu Li and
                  Yu Wang and
                  Bin Xiao and
                  Shuai Zhao and
                  Qinghua Hu},
  title        = {Collaborative Decision-Reinforced Self-Supervision for Attributed
                  Graph Clustering},
  journal      = {{IEEE} Trans. Neural Networks Learn. Syst.},
  volume       = {34},
  number       = {12},
  pages        = {10851--10863},
  year         = {2023},
}
```

