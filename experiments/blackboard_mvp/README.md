# 本地多黑板视觉证据原型

本目录独立于生产 Pipeline。只读用户指定的本地短视频，使用真实 FFprobe/FFmpeg 探测与提帧，再逐块比较手工指定的矩形区域。不导入生产模块、不连接学校账号、不读取凭据、不使用数据库、语音分析、公式识别、AI API、邮件或 PDF。

位置标签只用于辨认；板数可配置，没有“四块板必须按某顺序书写”的假设。输出是**视觉变化候选证据**，不是已确认的书写、讲解或完整证明顺序。

## 环境与工具

已验证 Python 3.12、NumPy 2.3.5、OpenCV 5.0.0.93，以及 FFmpeg/FFprobe 9.0.2。

本机本轮准备了仓库外的专用环境：

```sh
PY="$HOME/.local/share/icourse-blackboard-tools/python-env/bin/python"
```

FFmpeg/FFprobe 优先读取 `FFMPEG_BINARY` / `FFPROBE_BINARY`（仅工具路径），其次 PATH，最后本机固定工具缓存：

```text
~/.local/share/icourse-blackboard-tools/ffmpeg-9.0.2/bin/
```

本机缓存来自 FFmpeg 官方 9.0.2 签名源码的 arm64 构建，来源、签名验证与二进制哈希记录在工具缓存目录。该构建禁用网络，仅支持 file/pipe；没有安装系统软件或 Homebrew，也没有采用未知第三方二进制。其他机器可以使用已有可信 FFmpeg 5+ / FFprobe，通过上述路径变量指定。

新环境安装只需本目录 `requirements.txt`；不要改生产 `requirements.txt`。所有输入录像和运行结果必须在 Git 仓库外。运行入口也会拒绝仓库内输出、远程 URL、已有输出目录、重复 board_id 和越界 ROI。

## 运行合成演示

在仓库根目录执行，结果目录必须尚不存在：

```sh
cd "$HOME/Developer/iCourse_Subscriber-phase1"
PY="$HOME/.local/share/icourse-blackboard-tools/python-env/bin/python"
"$PY" -B -m experiments.blackboard_mvp demo \
  --output "$HOME/Movies/iCourse-Phase3-Runs/demo-new"
open "$HOME/Movies/iCourse-Phase3-Runs/demo-new/review/review.html"
```

合成视频为 640×400、4 fps、8 秒；四板配置故意用 D/A/C/B 顺序排列。实际变化为：

- 0 秒：四块板在片段起始时已有内容，不推断何时写成。
- 1/2/3/4 秒：C→A→D→B 变化，顺序来自观察时间而非位置。
- 5 秒：返回 C 板增加只有 10 像素的小负号，保留原版本及前后采样证据。
- 6 秒：A 板出现遮挡候选，其他板不因此创建新内容版本。
- 7 秒：B、D 同区间变化，保留共同采样区间和“先后未知”。

素材和 ground_truth 均为人工生成，不是课堂录像。它们验证归属、时间和证据保留，不证明真实黑板检测精度。

## 使用真实本地短视频

仅在你提供并允许测试的视频上运行；无需 UIS、Cookie、API Key 或网络。

1. 先探测并打开首帧。此页支持人工拖拽任意数量矩形，显示源像素坐标并生成可复制的配置 JSON；不自动检测黑板：

```sh
"$PY" -B -m experiments.blackboard_mvp probe \
  --video "/absolute/path/allowed-short-video.mp4" \
  --output "$HOME/Movies/iCourse-Phase3-Runs/real-probe-new"
open "$HOME/Movies/iCourse-Phase3-Runs/real-probe-new/source-preview.html"
```

2. 将页面生成的 JSON 保存为仓库外的 `boards.json`，人工设置稳定的 board_id 和位置标签。`roi` 是 `[x,y,width,height]`，不是 `[x1,y1,x2,y2]`。本目录 `example.boards.json` 仅适用于合成视频，不能照搬到真实录像。

3. 分析并直接打开离线预览：

```sh
"$PY" -B -m experiments.blackboard_mvp analyze \
  --video "/absolute/path/allowed-short-video.mp4" \
  --config "/absolute/path/boards.json" \
  --output "$HOME/Movies/iCourse-Phase3-Runs/real-review-new"
open "$HOME/Movies/iCourse-Phase3-Runs/real-review-new/review.html"
```

输出目录不能复用；换新名称重新运行，不会自动删除原结果。默认只接受容器时长可确认且不超过 10 分钟的片段；更长实验必须显式调整 `max_duration_ms`。工具有执行超时，不能承诺处理任意大或异常媒体。

## 时间、版本与顺序语义

- FFprobe 读取选定视频流原始 `pts` 和 `time_base`，FFmpeg 按相同源帧索引抽取 PNG；不使用“帧号/fps”编造时间。
- `native_pts/native_time_base/pts_origin` 保留源值；`pts_time_ms` 为源 PTS 换算毫秒；`timestamp_ms` 为相对首个视频帧的观察时间。`source.timeline_origin_ms` 记录偏移。
- 若只有 FFprobe `best_effort_timestamp`，显式标为 `best_effort_pts`；完全缺失或非单调时间戳时停止。毫秒显示会舍入，原始分数时间基准仍保留。
- FFmpeg 禁用自动旋转，ROI 坐标始终以实际提取 PNG 的像素为准；手机旋转元数据不会悄悄改变坐标。必要时先另存一个方向正确的本地副本，再重新手工框选。
- 各板独立比较局部差异，不用整张图的平均分压掉小符号；连通变化和累计基线差异都可产生候选。阈值可调，但低对比/极细笔画仍可能漏检。
- 普通事件前帧指上一采样；累计微变事件前帧指累计基线，时间窗从该基线开始，并另存 `previous_sample_frame_id`。不会假装知道符号的精确出现瞬间。
- 同次采样中多板有事件时，共享 simultaneous_group，标记 `undetermined_within_interval`。同组的 JSON/列表排列仅为显示，不表示内部先后。
- 连续候选段只是同板相邻视觉活动的引用集，可能与其他板交错，不假定老师写完一板才换板。
- `initial_existing`：起始时已有；无变化：持续引用旧版本；后续变化：追加新版本；遮挡也只作未确认观察，不覆盖此前状态。
- 不生成跨时间拼接黑板图。预览中的原图和各板裁剪属于同一实际采样时刻；事件前后证据有明确各自时间。

## ROI 身份失效

分辨率变化、可检测的全局位移/场景变化或板内一致位移会锁存 `needs_reconfirmation`。之后仍保存采样帧与裁剪，但不生成该板的可信连续内容版本。相似画面恢复不自动解锁。

这些检查是启发式警告，不是物理黑板身份认证，不能保证检测所有缓慢移动或复杂镜头变化。必须在预览中核对 ROI；也可在配置 `reconfirm_at_ms` 中人工加入需要失效的片段时刻。重新人工确认后，应以新的结果目录和更新配置处理新的稳定片段，而非自动恢复旧身份。

老师移动或遮挡只标 `possible_interference`。本轮没有人物动作识别，不会认定“检测到书写”。

## 输出与数据边界

```text
manifest.json       媒体信息、原始时间、全部采样、逐板事件、版本与候选段
review.html         内联CSS/JS，可file://直接打开，不依赖服务器/CDN/API
frames/fNNNN.png    全部采样原图，包括未入选帧
crops/board_id/     各采样时刻的板裁剪图
```

预览显示原图ROI/board_id、全局时间轴、逐板筛选、前后证据、变化框、版本历史和失效状态。筛选只改变引用，不删图。生成失败时可能保留本轮部分结果，需换新目录重跑；不要将缺少完整 manifest 的目录当成功结果。

## 测试

从仓库根目录：

```sh
PYTHONPATH=experiments "$PY" -B -m unittest discover \
  -s experiments/blackboard_mvp/tests -v
```

测试包含真实 FFmpeg/FFprobe 调用，核对 native PTS、帧内容对应事件、逐板归属和时间区间，不只统计截图数；另覆盖累计微变、遮挡、同时变化、返回旧板、相机/单板位移、分辨率变化锁存、未知时间戳、非法路径、拒绝覆盖与离线HTML安全。

生产测试仍使用原来的 `python -m unittest discover -s tests -v`，本原型不修改其代码或工作流。

## 滑动机械黑板：物理身份模式（本轮新增）

本模式仍在同一个原型中，使用配置中的 `assemblies` 显式启用；没有该字段的旧配置保持固定屏幕 ROI 模式，**不能宣称固定位置就是滑动板身份**。现有真实录屏的 `boards.json` 原样保留，未自动改名或转换，也未执行真实分析。

组件数可配置，每个组件恰有一个 `outer` 与一个 `inner`；本次演示为左右两个组件、四块物理板。`board_id` 稳定标识实体，不代表空间排序或教学顺序。`assembly_id` 表示机械关联；`layer` 表示前后遮挡关系，不能随着板升降交换。

- `assemblies[].viewport`：人工确认的组件屏幕窗口 `[x,y,w,h]`。
- `boards[].roi`：初始完整物理板的屏幕矩形（含暂时不可见部分），不是仅可见条带。尺寸/x 固定，首版只允许上下平移；y 可以越出画面。无法人工确定完整尺寸或身份时，不能用可见条带冒充完整物理板。
- `position_keyframes`：可选的精确采样时刻人工确认点 `{timestamp_ms, positions:{board_id:[x,y,w,h]}}`。含义为人工确认几何与身份，不是自动识别。不会自动把时间点挪到最近一帧，不接受未知 ID、尺寸变化或横向移动。
- 不提供后续确认点时，以历史可见纹理 ORB 特征、独立目标点、空间分布和一致平移估计位置，再核对可见像素。无法获得直接外观证据时标 `identity_uncertain`。无特征且有像素变化时，不以大量绿色背景相似来认定静止。
- `delta_y_inner ≈ -delta_y_outer` 只作近似运动先验。保留残差与容差（至少 3 像素、或外板位移的 15%）；不一致会提示复核，不强行修正已确认位置。单靠该先验补出的内板位置仍是未确认估计，不能发布内容变化。

逐板每次观察记录：`screen_roi`、`crop_roi`、`visible_regions`、`visible_fraction`、`visibility`、`identity_status`、`tracking_source`、`motion_status`、`version_id`。`position_history` 追加保存所有位置与遮挡状态，不替换以前的版本。

可见遮罩为：完整板面与组件窗口/画面交集，再从内板扣掉前方外板投影。`visible / partially_occluded / occluded` 是这个**机械几何模型**的状态。`1-visible_fraction` 是几何不可见比例，包含出框，不是纯机械遮挡率。若位置尚未确认，区域与比例也只是估计。没有人体分割，不能声称已剔除老师所有遮挡。

完全遮挡：保留历史内容和版本，暂停内容比较。重新出现：只有直接历史纹理证据或人工确认才能恢复可靠身份；反向移动、屏幕位置相同或重新可见均不够。首次显露且没有旧参考的像素只标首次观察，不能推断它们刚写成。已见过的区域在重现后若改变，候选时间窗从实际历史参考开始，不能把变化时间指认为重现瞬间。

`board_motion_event` 与 `board_content_change_event` 独立。整体位移先变换至物理板局部坐标，仅比较当前可见与历史已确认可见像素的交集。移动中暂缓内容判断，即使采样中继续书写也保留所有帧，等可靠静止帧再比较残差；不保证当时笔画与位移已自动分离。内容事件仍是视觉候选，老师干扰、真实书写或擦除需要人看图确认。

参考像素保留各自历史源帧；`comparison_references` 可追溯变化区域实际使用的前证据。没有把不同时间的板书合成一张仿佛同时存在的黑板图。相机/画面几何失效会锁存，人工位置确认点也不默默消除失效；应在新的输出目录重新配置稳定片段。

### 新机械合成演示

```sh
cd "$HOME/Developer/iCourse_Subscriber-phase1"
PY="$HOME/.local/share/icourse-blackboard-tools/python-env/bin/python"
"$PY" -B -m experiments.blackboard_mvp mechanical-demo \
  --output "$HOME/Movies/iCourse-Phase3-Runs/sliding-demo-new"
open "$HOME/Movies/iCourse-Phase3-Runs/sliding-demo-new/review/review.html"
```

12 秒、800×480、4 fps 合成视频，1 秒采样；包含外板下移/上推、内板反向移动、大部分遮挡、完全遮挡、重新出现、保留旧身份/内容、移动中增加小负号、左右组件独立运动、纹理匹配失败和局部可见比较。演示几何真值明确标为 `synthetic_ground_truth_manual_geometry`；9 秒故意不提供人工确认并遮住外观纹理。**它验证状态机和证据保留，不代表真实课堂自动追踪问题已经解决**。另有不提供人工确认的自动外观跟踪单元测试。

新增输出 `masks/board_id/fNNNN.png`（白为机械模型可见区、黑为几何不可见区；位置待确认时仅为估计）。裁剪图始终是该时间/屏幕位置的证据，部分遮挡时可能包含外板，不是重建的完整内板内容。`review.html` 按组件分组，展示动态位置/可见性、身份状态、机械/内容事件分类和每块板从遮挡到重新出现的完整历史；完全遮挡时不展示外板截图冒充内板内容。

`example.mechanical.boards.json` 仅对应这段合成视频的初始几何，不能直接用于真实录屏。真实数据下一次测试前，需要人工将原四个 ROI 映射到物理板、确认组件窗口/完整板尺寸，必要时提供位置确认点。本轮没有修改原 ROI、真实样本和旧运行目录，也不接入生产 Pipeline。


## 长采样列表提帧修复与全高度窗口候选

`runner.run_video()` 将 FFprobe 选中的源帧索引完整传入 `media.extract_frames()`。旧实现平铺拼接 `eq(n\,i)+eq(n\,j)+...`；本机 FFmpeg 9.0.2 的 `libavutil/eval.c` 将连续加法构成左深表达式树，超过 `MAX_DEPTH=100` 后返回错误，因此 197 项会失败。这不是 197 张图片的容量限制，也不是应该减少采样。

新实现保留索引顺序，将选择表达式构造成平衡树；每批至多512项，限制树深度与命令参数长度。197项只需一次解码；超过512项时分批从同一源流的帧0重新解码，输出编号延续到全局 `fNNNN.png`，不使用 `-ss` 改变源帧索引。每批检查全部累计输出文件，缺帧即停止，已经生成的证据保留，不自动重试或覆盖。多批的代价是重复解码，不减少任何请求的帧。

采样策略、PTS、time_base、相对时间和源帧索引映射保持不变。本修复不自动补入此前未选中的末帧：原先独立预检查的197张包括额外末帧，而正式CLI按原1000ms策略对本次视频选择196张；两者的策略差异不是修复时删帧，原197/297张历史预览也不会被删除。真正传入提帧器的每一张都会验证保留。

仅接受递增、唯一、非负、可被FFmpeg数值精确表示的整数源索引，拒绝空列表、重复、乱序和字符串表达式。FFmpeg通过参数数组调用，不使用shell；输出目录中的字面 `%` 另转义为 `%%`，避免 image2 把目录名也当图片编号。`source_frames()` 的间隔/时长参数在调用工具前验证，防止零间隔循环。

manifest 新增 `frame_extraction`，记录批次、请求数量及首尾索引。runner 验证解码PNG尺寸与该帧FFprobe元数据一致，再由同一张PNG产生四板裁剪与可见遮罩；尺寸不符则保留原图并停止，不生成伪装成功的manifest。

新增测试实际调用FFmpeg/FFprobe，覆盖197项、512项边界、513项多批稀疏选择、特殊路径、错误输入/输出、197帧四板逐像素裁剪和遮罩、全高度下遮挡/重现及源时间映射。所有测试使用合成素材，不执行真实课堂analyze。

全高度候选如 `left=[40,0,870,1008]`、`right=[912,0,874,1008]` 仅适用于已探测为1836×1008的本地录屏，横向范围和四个初始ROI不变。它覆盖这两条横向带内的全部**屏幕可见高度**，不能证明机械极限、隐藏位置或左内板身份。当前viewport同时用于搜索和几何可见遮罩；扩大后墙面/轨道/老师/水印可成为干扰，画面外或真实组件窗口外的板面也不能自动视为可见。相机背景区域减少，横向位移±2px门槛仍在。此配置必须标为实验候选，未知范围保持unknown；不是已完成真实遮挡识别的证明。

## Phase 3.2 S0–S2：自动几何与恢复

新增状态契约见 [STATE_CONTRACT.md](STATE_CONTRACT.md)。机械CLI默认进入新自动路径，保留旧tracker仅作明确的legacy/oracle测试对照。自动模式拒绝未来位置真值，内容接纳与writing_runs关闭；追踪成功不会生成板书变化。

在仓库根目录，使用现有工具环境回放已经保存的样本（新输出必须不存在）：

```sh
PYTHONPATH=experiments "$HOME/.local/share/icourse-blackboard-tools/python-env/bin/python" -B -m blackboard_mvp replay \
  --input-manifest /absolute/path/to/original/manifest.json \
  --output /absolute/path/to/new-output
```

replay不访问源视频或FFmpeg，不改原manifest或图片；复制全部采样并保持原PTS。可用`--config`提供独立配置，但不得夹带position_keyframes。组件`search_region`替代viewport表示搜索范围；`physical_aperture`未知留null，不自动拿整个搜索窗替代。相机`anchor_regions`可人工指定固定墙面/导轨；默认仅以组件搜索区外的像素作候选锚点，仍须经过几何证据校验。

本批追踪使用SIFT参考与多位移候选，没有实现板面LK、人体分割或内容识别。review显示实测/预测/最后可信值及camera状态；黑色未知mask不代表擦除。真实片的coverage是接纳比例，缺失独立真值时不输出伪造的身份准确率、位置误差或真实恢复延迟。

测试分为原有legacy/oracle几何与安全回归、新自动CV序列、真实FFmpeg/FFprobe自动合成视频、指标契约和生产回归；逐帧提供真值的测试不计入自动追踪性能。
