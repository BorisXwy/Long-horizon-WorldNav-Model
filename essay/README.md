# NAV 论文写作目录

本目录用于撰写 NAV 项目的论文。与 `doc/`（工程知识库）和 `result/`（实验
产物）分工不同，本目录只保存论文相关的 LaTeX 源文件、草稿和子文件。

## 目录结构

```text
essay/
├── README.md            本说明
├── nav_paper_draft.tex  单文件论文草稿：abstract / intro / related work /
│                        method / experiments / references
├── nav_paper_draft.pdf  由 nav_paper_draft.tex 编译得到的 PDF 草稿
├── intro/               引言章节
├── relatedworks/        相关工作章节
│   └── related_works_survey.md   调研汇总（链接、bibtex、问题、创新点、
│                                  核心方法、训练/推理测试数据、重要实验）
├── method/              方法章节
│   └── method.md        Method 草稿：quasi-infinite spatial memory as
│                        embodied observation history
├── writingref/          写法参考：相近会议论文/tech report 的组织方式与章节结构
└── experiment/          实验章节
```

各章节目录存放对应的 `.tex` 主文件与子文件；`relatedworks/` 额外维护
`related_works_survey.md` 作为调研工作表，后续写作时直接引用。

## 目录职责

- 核心论文 `.tex` 主文件；
- 各章节子文件、图表源码、bibliography；
- 写作草稿、笔记和投稿版本归档；
- `writingref/` 中保存相近论文的写法分析，只记录章节组织、论证主线和
  可借鉴/应避免的写法，不作为算法事实唯一来源；
- 不存放原始日志、checkpoint、生成视频或评测 JSON。

## 与其他目录的边界

| 目录 | 内容 |
| --- | --- |
| `NAV/essay/` | 论文写作源文件（本目录） |
| `NAV/doc/` | 工程设计、数据、训练、评测的可复现知识库 |
| `NAV/result/` | 评测指标、生成视频、可视化产物 |
| `NAV/log/` | 原始训练日志与 checkpoint |

论文中引用的实验数字、图表和结论应回溯到 `doc/` 与 `result/` 中的稳定
记录，不要在 `essay/` 内重复维护易变的运行数字。

## 约定

- 文件名使用 `lower_snake_case.tex` 或论文惯例命名；
- 主体语言随投稿目标；模型、数据集、指标名保留英文；
- 大型图表源码可建子目录（如 `figures/`、`tables/`）；
- 不提交大型二进制；图片优先使用矢量格式或可重新生成的脚本。
