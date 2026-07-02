# Self-test Summary

测试命令：

```bash
python run_pipeline.py --left /mnt/data/left.bmp --right /mnt/data/right.bmp --lens-xlsx /mnt/data/C48501B模组镜头-光学资料-20260417.xlsx --output /mnt/data/fisheye_sphere_pipeline_selftest --pano-width 1024 --pano-height 512
```

测试结果：通过。

关键输出：

- `selftest_output/final_multiband.png`
- `selftest_output/overlap_components_overlay.png`
- `selftest_output/global_seam_overlay.png`
- `selftest_output/pipeline_summary.json`
- `selftest_output/circle_report.json`
- `selftest_output/projection_report.json`
- `selftest_output/pose_overlap_report.json`
- `selftest_output/seam_report.json`

注意：这是低分辨率 smoke test，目的是验证模块链路和输出结构。正式调参建议使用 `--pano-width 4096 --pano-height 2048` 或更高。
