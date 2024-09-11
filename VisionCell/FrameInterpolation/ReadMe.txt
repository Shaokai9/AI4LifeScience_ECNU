这个脚本的目标是从输入的 GIF 中读取 10 帧，并通过插值技术在每两帧之间生成额外的 9 帧，从而得到一个总共 100 帧的输出 GIF。

函数 load_frames_from_gif:

这个函数的目的是从指定的 GIF 文件中加载所有帧。 使用 OpenCV 的 VideoCapture 读取 GIF 文件。 通过循环读取每一帧，并将其转换为灰度图像。 所有帧被添加到一个列表中并返回。

函数 interpolate:

这个函数用于插值两个图像帧。它接收两个图像帧和一个 alpha 值（介于0和1之间），表示插值的权重。 使用 OpenCV 的 addWeighted 函数进行权重插值。

函数 save_gif:

这个函数的目的是将一系列帧保存为一个 GIF 文件。 使用 OpenCV 的 VideoWriter 将帧写入文件。 这需要指定一个编解码器和帧的分辨率。

函数 main:

首先，从输入的 GIF 中加载帧。 然后，对每两个连续的帧进行插值，生成额外的 9 帧。 最后，保存所有帧为一个新的 GIF 文件。

脚本的执行:

设置输入 GIF 的路径。 根据输入 GIF 的名称生成输出 GIF 的名称，并在名称中加入 "_100"。 运行 main 函数来完成整个插值和保存过程。


确保已安装 OpenCV：

pip install opencv-python