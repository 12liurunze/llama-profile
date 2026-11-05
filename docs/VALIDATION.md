# 发布前验证范围

2026-11-05，本地CPU检查结果：

- 70项测试通过，覆盖数据、配置、输出计数、自然EOS、真实满BS窗口、八rank对齐、嵌套计时、kernel窗口、进程所有权、失败保留、恢复运行及报告封存校验。
- 4000条原始重复负载可离线逐字节重建；SHA256：`b79ae34f47010c0eb1d8929f785011a9c278279c90eb181010d75e598093cb4c`。
- 冻结源码归档4886个文件全部与manifest一致；SHA256：`55d9c884269a1e5fb948931e34eedbc16132a0334d03eb02894dc1e84657eb2d`。
- 在真实归档的四个待插桩源文件上执行转换，剥离计时包装后AST一致；原文件保持原hash。
- 核对冻结源码的Engine API、Graph配置解析、TP8头维度和输出计数路径。
- 上传清单不含模型权重、SSH凭据、实验服务器地址、个人实验结果或安装后的运行目录。归档中的上游本地服务示例含公开短占位API字符串，未当作真实凭据使用。

没有在本机运行70B模型、CUDA/NCCL、八卡Graph或真实性能实验。下列事项必须在目标机器验证：

1. 实际checkpoint完整且与配置一致；头部审计不能检查tensor数值是否被损坏或词表映射数值是否正确。
2. 软件包、驱动、A100互联与TP8兼容，完整模型能加载。
3. BS512的KV、激活、Graph内存足够，运行期间没有retraction/OOM。
4. 选定长响应负载能产生连续至少10步真实满BS且无Prefill的窗口。
5. Graph内部GPU kernel实际可见，八个rank捕获同一组诊断step。

代码通过只表示这些检查已经实现，不能预先宣称上述硬件条件全部成立。正式结果为空时应保留失败证据并诊断。

复查指令：

```bash
python -m unittest discover -s tests -v
python scripts/build_dataset.py --check
python tools/check_release.py
```
