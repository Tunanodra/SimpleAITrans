# 超长文本汉化工具使用说明

这是一个简单可靠的超长文本汉化工具，专为处理大型文本文件而设计，能够保持上下文连贯性并有效管理内存。

## 主要特点

- **简单易用的界面**：直观的GUI界面，操作简单明了
- **AI驱动翻译**：使用OpenAI API进行高质量翻译
- **自定义API设置**：支持自定义API密钥、Base URL和模型
- **上下文感知**：智能分块处理，保持翻译的上下文连贯性
- **内存优化**：有效管理内存，能够处理超长文本
- **多线程翻译**：可选的多线程处理，提高翻译速度
- **速率限制控制**：支持RPM（每分钟请求数）和TPM（每分钟令牌数）限制
- **实时进度显示**：详细显示翻译进度和状态信息
- **对话历史**：可选显示与AI的交互历史

## 安装

### 依赖项

- Python 3.8+
- OpenAI Python库
- CustomTkinter

### 安装步骤

1. 确保已安装Python 3.8或更高版本
2. 安装依赖库：

```bash
pip install openai customtkinter
```

3. 运行程序：

```bash
python simple_translator.py
```

## 使用方法

### 配置API设置

首次使用时，需要配置API设置：

1. 点击界面上的"API设置"按钮
2. 输入您的OpenAI API密钥
3. 如果使用第三方OpenAI兼容API，可以修改Base URL
4. 选择预设模型（gpt-3.5-turbo、gpt-4、gpt-4-turbo）或勾选"自定义模型"并输入模型名称
5. 设置线程数（1-10）
6. 可选：设置RPM和TPM限制
7. 可选：调整高级设置（温度、块大小、行数等）
8. 点击"测试API"按钮验证设置是否正确
9. 点击"保存设置"按钮保存配置

### 翻译文本

1. 在左侧文本框中输入需要翻译的文本，或者使用"加载文件"按钮从文件中加载文本
2. 选择是否使用多线程（对于大文件，多线程可以提高速度）
3. 选择是否显示对话历史（可以查看与AI的交互过程）
4. 点击"开始翻译"按钮开始翻译过程
5. 翻译进度会在状态栏和进度条中显示
6. 翻译结果会实时显示在右侧文本框中
7. 翻译完成后，可以使用"保存翻译"按钮将结果保存到文件

### 停止翻译

如果需要中断翻译过程，可以点击"停止翻译"按钮。程序会尽快停止翻译过程，已翻译的部分会保留在右侧文本框中。

## 技术细节

### 文本分块

程序使用混合分块策略，同时考虑行数和字符数：

- 每个块最多包含指定行数（默认50行）
- 每个块最多包含指定字符数（默认2000字符）
- 块之间有重叠区域（默认200字符），确保上下文连贯性

### 多线程处理

多线程模式下，程序会：

1. 将文本分成多个块
2. 创建指定数量的工作线程（默认2个）
3. 将块分配给工作线程进行翻译
4. 按原始顺序合并翻译结果

### 速率限制

速率限制功能可以控制API请求频率：

- RPM（每分钟请求数）：限制每分钟发送的请求数量
- TPM（每分钟令牌数）：限制每分钟使用的令牌数量

如果达到限制，程序会自动等待适当的时间再继续翻译。

### 内存管理

程序采用以下策略优化内存使用：

1. 分块处理文本，避免一次性加载全部内容
2. 定期执行垃圾回收，释放不再需要的内存
3. 使用翻译缓存，避免重复翻译相同内容

## 常见问题

### Q: 程序支持哪些文件格式？
A: 程序支持任何文本格式的文件，包括但不限于TXT、JSON、MD、PY等。

### Q: 如何处理超大文件？
A: 对于超大文件，建议启用多线程功能，并适当调整块大小和行数。程序设计为能够处理超长文本，只要有足够的时间和API配额。

### Q: 如何提高翻译速度？
A: 可以在API设置中增加线程数（最高10个），但请注意这会增加API请求频率，可能需要设置适当的RPM和TPM限制。

### Q: 如何保证翻译质量？
A: 程序使用上下文感知的分块策略，确保翻译的连贯性。您也可以通过调整温度参数（较低的值产生更确定性的输出）来影响翻译风格。

### Q: 为什么需要API密钥？
A: 程序使用OpenAI API进行翻译，需要API密钥进行身份验证。您可以在OpenAI官网获取API密钥，或使用兼容的第三方API服务。

## 故障排除

### API错误
- 检查API密钥是否正确
- 验证Base URL是否正确（对于第三方API）
- 使用"测试API"按钮验证连接
- 检查API账户余额和限制

### 内存错误
- 减小块大小和最大行数
- 关闭其他内存密集型应用程序

### 翻译质量问题
- 调整温度参数（较低的值产生更确定性的输出）
- 增加重叠大小，提高上下文连贯性

## 许可证

本程序仅供学习和研究使用，使用前请确保遵守OpenAI的使用条款和相关法律法规。
