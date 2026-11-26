MiniMax-H3 activation offload 缓存原生 MUSA Flash SDPA 的 output 和 LSE，减少 checkpoint 重算。默认关闭，需要 eager、普通 MiniMaxH3DiT 和 `--use_gradient_checkpointing`。启用时 `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch` 在 MUSA 上使用原生 Flash SDPA。

已有 Ref2VA 多机启动命令可以增加以下环境变量。层数和 pinned 主存预算需按实际模型、序列长度和机器容量选择。

```bash
export MINIMAX_H3_ACTIVATION_OFFLOAD_LAYERS=24
export MINIMAX_H3_ACTIVATION_GPU_LAYERS=12
export MINIMAX_H3_ACTIVATION_OFFLOAD_GIB=24
export TRAIN_STEPS=50
# 随后执行原有 start_ref2va_multinode.sh 启动命令。
```

对应训练参数仍为 `--activation_offload_layers`、`--activation_gpu_layers`、`--activation_offload_gib`，没有新增调度参数。前 N 层把 FA 结果复制到 pinned CPU 内存，末 K 层保留 GPU 结果，中间层重算。CPU payload 超过 host 预算的 segment 也会重算。

D2H 使用独立流和异步 DMA。每个 block 持有其 FA 源存储引用，在该 block 的前向计算全部提交后，让源分配流等待精确的复制完成事件，再释放引用。每个 block 的异常出口也执行同样的收尾。因此显式保留的 D2H 源存储局限于当前 block 的 segment，不累计到整个前向结束。

H2D 的恢复 buffer 分配在最终计算流上，复制仍在独立 H2D 流上运行。H2D 流先等待分配流先前的内存使用以及 D2H 完成事件，然后复制。预取持有 buffer 引用；`restore()` 将 H2D 完成事件接回计算流后，把张量交给 autograd。这些张量随后由 autograd 持有到最后一次实际使用，分配、消费和复用归属同一计算流，因而不需要 `record_stream(consumer)`。

这里有一个关键边界：checkpoint replay 返回或 early-stop 后，原始图的 FA backward 还会读取恢复的 O/LSE，不能在 replay 结束时回收或覆盖它们。上述分配归属方案直接沿用 autograd 的实际生命周期，不需要用 replay 结束事件猜测最后使用点。

预取目标取原始 FA 前向的计算流，与正常 autograd backward 的流一致。如果实际 restore 使用另一个流，则先按精确事件释放旧预取，在实际消费流上重新分配并复制。只预取而不消费的 buffer 在 replay 收尾、backward task 完成/异常清理或图销毁时安全释放。重复 backward 和交错 graph 各自恢复所需结果，不在不同 replay 之间提前覆盖存储。

双向搬运都不再向 allocator 登记额外使用流；完成事件仅建立设备流依赖，不增加 CPU/device synchronize。预取深度仍为相邻一层。如果最后一个 DiT block 也启用 CPU offload，其 backward 入口先预取本层，再预取前一层，避免反向的第一份恢复等到 FA replay 才启动。FSDP CE 配置不受此修复影响。完整四机训练的长期吞吐与内存趋势需要在相同配置下配对验证；allocator 的 `free_requested → free_completed` 接近零，仅表示可按流顺序安全复用，不表示 DMA 已瞬间完成。

相关回归测试可在仓库根目录执行。

```bash
python -m pytest tests/test_minimax_h3_activation_offload.py \
  tests/test_minimax_h3_musa_patch.py tests/test_minimax_h3_training.py \
  tests/test_minimax_h3_compile.py tests/test_fsdp2_options.py -q
```
