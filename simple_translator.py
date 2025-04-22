"""
简单可靠的超长文本翻译工具

这个程序提供了一个简单的GUI界面，用于翻译超长文本内容。
它使用OpenAI API进行翻译，支持自定义base URL和模型。
程序采用分块处理方式，能够保持上下文连续性，并有效管理内存。
"""

import os
import sys
import time
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import queue
import gc
from typing import List, Dict, Optional, Tuple, Any

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI库未安装，请使用 pip install openai 安装")
    sys.exit(1)

try:
    import customtkinter as ctk
except ImportError:
    print("CustomTkinter库未安装，请使用 pip install customtkinter 安装")
    sys.exit(1)

# 设置主题
ctk.set_appearance_mode("System")  # 系统主题
ctk.set_default_color_theme("blue")  # 蓝色主题

# 常量定义
DEFAULT_MAX_CHUNK_SIZE = 2000  # 默认最大块大小
DEFAULT_MAX_LINES = 50  # 默认最大行数
DEFAULT_OVERLAP_SIZE = 200  # 默认重叠大小
DEFAULT_TEMPERATURE = 0.3  # 默认温度
DEFAULT_MODEL = "gpt-3.5-turbo"  # 默认模型
DEFAULT_THREAD_COUNT = 2  # 默认线程数
CONFIG_FILE = "translator_config.json"  # 配置文件

class TranslationChunk:
    """翻译块类，表示待翻译的文本块"""
    
    def __init__(self, 
                 index: int, 
                 text: str, 
                 previous_context: str = "", 
                 next_context: str = ""):
        """
        初始化翻译块
        
        Args:
            index: 块索引
            text: 块文本
            previous_context: 前文上下文
            next_context: 后文上下文
        """
        self.index = index
        self.text = text
        self.previous_context = previous_context
        self.next_context = next_context
        self.translated_text = ""
        self.is_translated = False
        self.error = None

class TextChunker:
    """文本分块器，将长文本分割成小块"""
    
    def __init__(self, 
                 max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE, 
                 max_lines: int = DEFAULT_MAX_LINES,
                 overlap_size: int = DEFAULT_OVERLAP_SIZE):
        """
        初始化文本分块器
        
        Args:
            max_chunk_size: 最大块大小（字符数）
            max_lines: 最大行数
            overlap_size: 重叠大小（字符数）
        """
        self.max_chunk_size = max_chunk_size
        self.max_lines = max_lines
        self.overlap_size = overlap_size
    
    def chunk_text(self, text: str) -> List[TranslationChunk]:
        """
        将文本分块
        
        Args:
            text: 待分块的文本
            
        Returns:
            chunks: 文本块列表
        """
        if not text:
            return []
        
        # 按行分割文本
        lines = text.splitlines(True)  # 保留换行符
        
        chunks = []
        current_chunk_lines = []
        current_chunk_size = 0
        chunk_index = 0
        
        for i, line in enumerate(lines):
            # 如果当前行加入后会超过最大块大小或最大行数，则创建新块
            if (current_chunk_size + len(line) > self.max_chunk_size or 
                len(current_chunk_lines) >= self.max_lines) and current_chunk_lines:
                
                # 创建当前块的文本
                chunk_text = "".join(current_chunk_lines)
                
                # 获取前文上下文
                previous_context = ""
                if chunk_index > 0:
                    # 使用上一个块的最后部分作为上下文
                    prev_chunk = chunks[chunk_index - 1]
                    prev_text = prev_chunk.text
                    if len(prev_text) > self.overlap_size:
                        previous_context = prev_text[-self.overlap_size:]
                    else:
                        previous_context = prev_text
                
                # 获取后文上下文
                next_context = ""
                next_lines = lines[i:i+10]  # 最多取后面10行作为上下文
                if next_lines:
                    next_text = "".join(next_lines)
                    if len(next_text) > self.overlap_size:
                        next_context = next_text[:self.overlap_size]
                    else:
                        next_context = next_text
                
                # 创建并添加块
                chunk = TranslationChunk(
                    index=chunk_index,
                    text=chunk_text,
                    previous_context=previous_context,
                    next_context=next_context
                )
                chunks.append(chunk)
                
                # 重置当前块
                current_chunk_lines = []
                current_chunk_size = 0
                chunk_index += 1
            
            # 添加当前行到当前块
            current_chunk_lines.append(line)
            current_chunk_size += len(line)
        
        # 处理最后一个块
        if current_chunk_lines:
            chunk_text = "".join(current_chunk_lines)
            
            # 获取前文上下文
            previous_context = ""
            if chunk_index > 0:
                prev_chunk = chunks[chunk_index - 1]
                prev_text = prev_chunk.text
                if len(prev_text) > self.overlap_size:
                    previous_context = prev_text[-self.overlap_size:]
                else:
                    previous_context = prev_text
            
            # 创建并添加块
            chunk = TranslationChunk(
                index=chunk_index,
                text=chunk_text,
                previous_context=previous_context,
                next_context=""
            )
            chunks.append(chunk)
        
        return chunks

class RateLimiter:
    """速率限制器，控制API请求频率"""
    
    def __init__(self, rpm_limit: Optional[int] = None, tpm_limit: Optional[int] = None):
        """
        初始化速率限制器
        
        Args:
            rpm_limit: 每分钟请求数限制
            tpm_limit: 每分钟令牌数限制
        """
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_timestamps = []
        self.token_usage = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self, tokens: int = 0) -> float:
        """
        如果需要，等待以符合速率限制
        
        Args:
            tokens: 本次请求使用的令牌数
            
        Returns:
            wait_time: 等待时间（秒）
        """
        with self.lock:
            now = time.time()
            one_minute_ago = now - 60
            
            # 清理旧记录
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > one_minute_ago]
            self.token_usage = [(ts, usage) for ts, usage in self.token_usage if ts > one_minute_ago]
            
            wait_time = 0.0
            
            # 检查RPM限制
            if self.rpm_limit is not None and self.rpm_limit > 0:
                current_rpm = len(self.request_timestamps)
                if current_rpm >= self.rpm_limit:
                    # 计算需要等待的时间
                    oldest_timestamp = self.request_timestamps[0] if self.request_timestamps else now
                    wait_time_rpm = max(0, 60 - (now - oldest_timestamp))
                    wait_time = max(wait_time, wait_time_rpm)
            
            # 检查TPM限制
            if self.tpm_limit is not None and self.tpm_limit > 0:
                current_tpm = sum(usage for _, usage in self.token_usage)
                if current_tpm + tokens >= self.tpm_limit:
                    # 计算需要等待的时间
                    oldest_timestamp = self.token_usage[0][0] if self.token_usage else now
                    wait_time_tpm = max(0, 60 - (now - oldest_timestamp))
                    wait_time = max(wait_time, wait_time_tpm)
            
            # 如果需要等待，返回等待时间
            if wait_time > 0:
                return wait_time
            
            # 记录本次请求
            self.request_timestamps.append(now)
            if tokens > 0:
                self.token_usage.append((now, tokens))
            
            return 0.0
    
    def record_usage(self, tokens: int):
        """
        记录令牌使用情况
        
        Args:
            tokens: 使用的令牌数
        """
        with self.lock:
            now = time.time()
            self.token_usage.append((now, tokens))

class Translator:
    """翻译器，使用OpenAI API翻译文本"""
    
    def __init__(self, 
                 api_key: str, 
                 base_url: str = "https://api.openai.com/v1",
                 model: str = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 thread_count: int = DEFAULT_THREAD_COUNT,
                 rpm_limit: Optional[int] = None,
                 tpm_limit: Optional[int] = None):
        """
        初始化翻译器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 模型名称
            temperature: 温度参数
            thread_count: 线程数
            rpm_limit: 每分钟请求数限制
            tpm_limit: 每分钟令牌数限制
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.thread_count = max(1, min(thread_count, 10))  # 限制线程数在1-10之间
        
        # 创建速率限制器
        self.rate_limiter = RateLimiter(rpm_limit, tpm_limit)
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 停止标志
        self.should_stop = False
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 翻译缓存
        self.translation_cache = {}
        
        # 进度回调
        self.progress_callback = None
        
        # 对话回调
        self.conversation_callback = None
    
    def set_progress_callback(self, callback):
        """
        设置进度回调
        
        Args:
            callback: 进度回调函数，参数为(进度, 状态, 详情)
        """
        self.progress_callback = callback
    
    def set_conversation_callback(self, callback):
        """
        设置对话回调
        
        Args:
            callback: 对话回调函数，参数为(角色, 内容)
        """
        self.conversation_callback = callback
    
    def update_progress(self, progress: float, status: str, detail: str = ""):
        """
        更新进度
        
        Args:
            progress: 进度（0-1）
            status: 状态信息
            detail: 详细信息
        """
        if self.progress_callback:
            self.progress_callback(progress, status, detail)
    
    def add_to_conversation(self, role: str, content: str):
        """
        添加对话记录
        
        Args:
            role: 角色（"user"或"assistant"）
            content: 内容
        """
        if self.conversation_callback:
            self.conversation_callback(role, content)
    
    def translate(self, text: str, use_threading: bool = True) -> str:
        """
        翻译文本
        
        Args:
            text: 待翻译的文本
            use_threading: 是否使用多线程
            
        Returns:
            translated_text: 翻译后的文本
        """
        if not text.strip():
            return ""
        
        # 更新进度
        self.update_progress(0.1, "正在分割文本...", f"文本长度: {len(text)} 字符")
        
        # 创建分块器
        chunker = TextChunker()
        
        # 分块
        chunks = chunker.chunk_text(text)
        
        # 更新进度
        self.update_progress(0.2, f"文本已分割为 {len(chunks)} 个块", f"开始翻译")
        
        # 重置停止标志
        self.should_stop = False
        
        # 翻译结果
        translated_chunks = []
        
        if use_threading and self.thread_count > 1:
            # 多线程翻译
            translated_chunks = self._translate_with_threading(chunks)
        else:
            # 单线程翻译
            translated_chunks = self._translate_sequential(chunks)
        
        # 更新进度
        self.update_progress(0.9, "正在合并翻译结果...", f"合并 {len(translated_chunks)} 个块")
        
        # 合并翻译结果
        translated_text = ""
        for chunk in translated_chunks:
            if chunk.is_translated:
                translated_text += chunk.translated_text
            else:
                # 如果块未翻译（可能是因为错误或停止），使用原文
                translated_text += chunk.text
        
        # 更新进度
        self.update_progress(1.0, "翻译完成", f"总字符数: {len(translated_text)}")
        
        return translated_text
    
    def _translate_sequential(self, chunks: List[TranslationChunk]) -> List[TranslationChunk]:
        """
        顺序翻译块
        
        Args:
            chunks: 待翻译的块列表
            
        Returns:
            translated_chunks: 翻译后的块列表
        """
        for i, chunk in enumerate(chunks):
            if self.should_stop:
                break
            
            # 更新进度
            progress = 0.2 + (i / len(chunks)) * 0.7
            self.update_progress(
                progress, 
                f"正在翻译第 {i+1}/{len(chunks)} 块...", 
                f"块大小: {len(chunk.text)} 字符"
            )
            
            try:
                # 翻译块
                translated_text = self._translate_chunk(
                    chunk.text,
                    chunk.previous_context,
                    chunk.next_context
                )
                
                # 保存翻译结果
                chunk.translated_text = translated_text
                chunk.is_translated = True
                
                # 更新进度
                self.update_progress(
                    progress, 
                    f"已完成第 {i+1}/{len(chunks)} 块", 
                    f"翻译结果: {len(translated_text)} 字符"
                )
                
                # 清理内存
                if i % 10 == 0:
                    gc.collect()
            
            except Exception as e:
                # 记录错误
                chunk.error = str(e)
                print(f"翻译块 {i} 出错: {str(e)}")
        
        return chunks
    
    def _translate_with_threading(self, chunks: List[TranslationChunk]) -> List[TranslationChunk]:
        """
        使用多线程翻译块
        
        Args:
            chunks: 待翻译的块列表
            
        Returns:
            translated_chunks: 翻译后的块列表
        """
        # 创建工作队列
        work_queue = queue.Queue()
        
        # 添加任务到队列
        for chunk in chunks:
            work_queue.put(chunk)
        
        # 创建结果列表
        results = [None] * len(chunks)
        
        # 创建线程锁
        lock = threading.Lock()
        
        # 已完成的任务数
        completed_tasks = 0
        
        # 工作线程函数
        def worker():
            nonlocal completed_tasks
            
            while not self.should_stop:
                try:
                    # 获取任务，最多等待1秒
                    chunk = work_queue.get(timeout=1)
                    
                    # 更新进度
                    with lock:
                        progress = 0.2 + (completed_tasks / len(chunks)) * 0.7
                        self.update_progress(
                            progress, 
                            f"正在翻译第 {chunk.index+1}/{len(chunks)} 块...", 
                            f"块大小: {len(chunk.text)} 字符"
                        )
                    
                    try:
                        # 翻译块
                        translated_text = self._translate_chunk(
                            chunk.text,
                            chunk.previous_context,
                            chunk.next_context
                        )
                        
                        # 保存翻译结果
                        chunk.translated_text = translated_text
                        chunk.is_translated = True
                        
                        # 保存结果
                        results[chunk.index] = chunk
                        
                        # 更新进度
                        with lock:
                            completed_tasks += 1
                            progress = 0.2 + (completed_tasks / len(chunks)) * 0.7
                            self.update_progress(
                                progress, 
                                f"已完成第 {chunk.index+1}/{len(chunks)} 块", 
                                f"翻译结果: {len(translated_text)} 字符"
                            )
                    
                    except Exception as e:
                        # 记录错误
                        chunk.error = str(e)
                        results[chunk.index] = chunk
                        print(f"翻译块 {chunk.index} 出错: {str(e)}")
                        
                        # 更新进度
                        with lock:
                            completed_tasks += 1
                    
                    finally:
                        # 标记任务完成
                        work_queue.task_done()
                
                except queue.Empty:
                    # 队列为空，退出线程
                    break
        
        # 创建工作线程
        threads = []
        for _ in range(self.thread_count):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # 等待所有任务完成或停止信号
        while not work_queue.empty() and not self.should_stop:
            time.sleep(0.1)
        
        # 如果收到停止信号，清空队列
        if self.should_stop:
            while not work_queue.empty():
                try:
                    work_queue.get_nowait()
                    work_queue.task_done()
                except queue.Empty:
                    break
        
        # 等待所有线程结束
        for thread in threads:
            thread.join(timeout=1)
        
        # 过滤掉未翻译的块
        return [chunk for chunk in results if chunk is not None]
    
    def _translate_chunk(self, 
                        text: str, 
                        previous_context: str = "", 
                        next_context: str = "") -> str:
        """
        翻译块
        
        Args:
            text: 待翻译的文本
            previous_context: 前文上下文
            next_context: 后文上下文
            
        Returns:
            translated_text: 翻译后的文本
        """
        if not text.strip():
            return ""
        
        # 检查缓存
        cache_key = text
        with self.lock:
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]
        
        # 构建系统提示词
        system_prompt = """你是一个专业的中文翻译，负责将文本翻译成流畅、自然的中文。请遵循以下规则：
1. 保持原文的格式，包括段落、换行、缩进等
2. 保留原文中的代码块、标记和控制字符
3. 如果无法判断文件格式, 那就只有引号中的句子需要翻译, 任何人名都保持原状就行
4. 对于JSON、XML等结构化数据，保持其结构不变，只翻译内容部分
5. 对于Markdown、HTML等标记语言，保持标记不变，只翻译内容部分
6. 翻译时考虑上下文，确保术语一致性, 但不许输出上下文
7. 输出纯翻译结果，不要添加解释或注释
8. 如果无法翻译某些内容，保留原文, 同时不要进行任何处理
"""
        
        # 构建用户提示词
        user_prompt = ""
        
        # 添加前文上下文
        if previous_context:
            user_prompt += f"这是前文的上下文（仅供参考，不需要翻译, 同时禁止输出到结果中）:\n{previous_context}\n\n"
        
        # 添加待翻译文本
        user_prompt += f"待翻译文本:\n{text}"
        
        # 添加后文上下文
        if next_context:
            user_prompt += f"\n\n后文上下文（仅供参考，不需要翻译, 同时禁止输出到结果中）:\n{next_context}"
        
        # 添加到对话历史
        self.add_to_conversation("user", user_prompt)
        
        try:
            # 等待速率限制
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                self.update_progress(
                    0.5, 
                    f"等待API速率限制...", 
                    f"等待时间: {wait_time:.1f} 秒"
                )
                time.sleep(wait_time)
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            
            # 获取响应
            translated_text = response.choices[0].message.content
            
            # 记录令牌使用情况
            tokens_used = response.usage.total_tokens
            self.rate_limiter.record_usage(tokens_used)
            
            # 添加到对话历史
            self.add_to_conversation("assistant", translated_text)
            
            # 添加到缓存
            with self.lock:
                self.translation_cache[cache_key] = translated_text
            
            return translated_text
        
        except Exception as e:
            error_message = f"翻译出错: {str(e)}"
            print(error_message)
            
            # 添加到对话历史
            self.add_to_conversation("assistant", error_message)
            
            # 抛出异常
            raise
    
    def test_api(self) -> Tuple[bool, str]:
        """
        测试API连接
        
        Returns:
            success: 是否成功
            message: 成功或错误消息
        """
        try:
            # 简单的测试文本
            test_text = "Hello, world! This is a test."
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a translator. Translate the following text to Chinese."},
                    {"role": "user", "content": test_text}
                ],
                temperature=self.temperature
            )
            
            # 获取响应
            translated_text = response.choices[0].message.content
            
            # 记录令牌使用情况
            tokens_used = response.usage.total_tokens
            
            return True, f"API测试成功！翻译结果: {translated_text}\n使用令牌: {tokens_used}"
        
        except Exception as e:
            error_message = f"API测试失败: {str(e)}"
            return False, error_message
    
    def stop(self):
        """停止翻译"""
        self.should_stop = True

class TranslatorApp:
    """翻译器应用程序"""
    
    def __init__(self):
        """初始化应用程序"""
        # 创建主窗口
        self.root = ctk.CTk()
        self.root.title("超长文本汉化工具")
        self.root.geometry("1200x800")
        
        # 加载配置
        self.config = self._load_config()
        
        # 创建翻译器
        self.translator = None
        
        # 翻译线程
        self.translation_thread = None
        
        # 创建UI
        self._create_ui()
        
        # 更新API设置
        self._update_api_settings_from_config()
    
    def _load_config(self) -> Dict:
        """
        加载配置
        
        Returns:
            config: 配置字典
        """
        default_config = {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model": DEFAULT_MODEL,
            "custom_model": "",
            "use_custom_model": False,
            "temperature": DEFAULT_TEMPERATURE,
            "thread_count": DEFAULT_THREAD_COUNT,
            "rpm_limit": None,
            "tpm_limit": None,
            "max_chunk_size": DEFAULT_MAX_CHUNK_SIZE,
            "max_lines": DEFAULT_MAX_LINES,
            "overlap_size": DEFAULT_OVERLAP_SIZE
        }
        
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # 确保所有必要的键都存在
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                
                return config
        except Exception as e:
            print(f"加载配置出错: {str(e)}")
        
        return default_config
    
    def _save_config(self):
        """保存配置"""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置出错: {str(e)}")
    
    def _create_ui(self):
        """创建UI"""
        # 创建主框架
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建顶部框架
        self.top_frame = ctk.CTkFrame(self.main_frame)
        self.top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建API设置按钮
        self.api_settings_button = ctk.CTkButton(
            self.top_frame, 
            text="API设置", 
            command=self._show_api_settings
        )
        self.api_settings_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 创建加载文件按钮
        self.load_file_button = ctk.CTkButton(
            self.top_frame, 
            text="加载文件", 
            command=self._load_file
        )
        self.load_file_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 创建保存翻译按钮
        self.save_translation_button = ctk.CTkButton(
            self.top_frame, 
            text="保存翻译", 
            command=self._save_translation
        )
        self.save_translation_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 创建多线程复选框
        self.use_threading_var = tk.BooleanVar(value=True)
        self.use_threading_checkbox = ctk.CTkCheckBox(
            self.top_frame, 
            text="使用多线程", 
            variable=self.use_threading_var
        )
        self.use_threading_checkbox.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 创建显示对话复选框
        self.show_conversation_var = tk.BooleanVar(value=False)
        self.show_conversation_checkbox = ctk.CTkCheckBox(
            self.top_frame, 
            text="显示对话", 
            variable=self.show_conversation_var,
            command=self._toggle_conversation_panel
        )
        self.show_conversation_checkbox.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 创建开始翻译按钮
        self.start_translation_button = ctk.CTkButton(
            self.top_frame, 
            text="开始翻译", 
            command=self._start_translation
        )
        self.start_translation_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 创建停止翻译按钮
        self.stop_translation_button = ctk.CTkButton(
            self.top_frame, 
            text="停止翻译", 
            command=self._stop_translation,
            state=tk.DISABLED
        )
        self.stop_translation_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 创建中间框架
        self.middle_frame = ctk.CTkFrame(self.main_frame)
        self.middle_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建文本框架
        self.text_frame = ctk.CTkFrame(self.middle_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # 创建对话框架
        self.conversation_frame = ctk.CTkFrame(self.middle_frame)
        self.conversation_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0))
        self.conversation_frame.pack_forget()  # 默认隐藏
        
        # 创建文本框架的左右分割
        self.text_paned = ttk.PanedWindow(self.text_frame, orient=tk.HORIZONTAL)
        self.text_paned.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧文本框架
        self.left_text_frame = ctk.CTkFrame(self.text_paned)
        self.text_paned.add(self.left_text_frame, weight=1)
        
        # 创建右侧文本框架
        self.right_text_frame = ctk.CTkFrame(self.text_paned)
        self.text_paned.add(self.right_text_frame, weight=1)
        
        # 创建左侧标签
        self.left_label = ctk.CTkLabel(self.left_text_frame, text="原文")
        self.left_label.pack(padx=5, pady=5, anchor=tk.W)
        
        # 创建左侧文本框
        self.left_text = tk.Text(self.left_text_frame, wrap=tk.WORD)
        self.left_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建左侧滚动条
        self.left_scrollbar = ttk.Scrollbar(self.left_text, command=self.left_text.yview)
        self.left_text.configure(yscrollcommand=self.left_scrollbar.set)
        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建右侧标签
        self.right_label = ctk.CTkLabel(self.right_text_frame, text="译文")
        self.right_label.pack(padx=5, pady=5, anchor=tk.W)
        
        # 创建右侧文本框
        self.right_text = tk.Text(self.right_text_frame, wrap=tk.WORD)
        self.right_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建右侧滚动条
        self.right_scrollbar = ttk.Scrollbar(self.right_text, command=self.right_text.yview)
        self.right_text.configure(yscrollcommand=self.right_scrollbar.set)
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建对话标签
        self.conversation_label = ctk.CTkLabel(self.conversation_frame, text="对话历史")
        self.conversation_label.pack(padx=5, pady=5, anchor=tk.W)
        
        # 创建对话文本框
        self.conversation_text = tk.Text(self.conversation_frame, wrap=tk.WORD)
        self.conversation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建对话滚动条
        self.conversation_scrollbar = ttk.Scrollbar(self.conversation_text, command=self.conversation_text.yview)
        self.conversation_text.configure(yscrollcommand=self.conversation_scrollbar.set)
        self.conversation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 设置对话文本框标签
        self.conversation_text.tag_configure("user", foreground="blue")
        self.conversation_text.tag_configure("assistant", foreground="green")
        
        # 创建底部框架
        self.bottom_frame = ctk.CTkFrame(self.main_frame)
        self.bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建进度条
        self.progress_bar = ttk.Progressbar(self.bottom_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建状态标签
        self.status_label = ctk.CTkLabel(self.bottom_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 创建详细进度标签
        self.detail_label = ctk.CTkLabel(self.bottom_frame, text="")
        self.detail_label.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def _toggle_conversation_panel(self):
        """切换对话面板显示状态"""
        if self.show_conversation_var.get():
            self.conversation_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0))
        else:
            self.conversation_frame.pack_forget()
    
    def _show_api_settings(self):
        """显示API设置对话框"""
        # 创建设置窗口
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("API设置")
        settings_window.geometry("500x600")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # 创建设置框架
        settings_frame = ctk.CTkFrame(settings_window)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建API密钥标签和输入框
        api_key_label = ctk.CTkLabel(settings_frame, text="API密钥:")
        api_key_label.pack(anchor=tk.W, padx=5, pady=5)
        
        api_key_entry = ctk.CTkEntry(settings_frame, width=400, show="*")
        api_key_entry.pack(fill=tk.X, padx=5, pady=5)
        api_key_entry.insert(0, self.config["api_key"])
        
        # 创建Base URL标签和输入框
        base_url_label = ctk.CTkLabel(settings_frame, text="Base URL:")
        base_url_label.pack(anchor=tk.W, padx=5, pady=5)
        
        base_url_entry = ctk.CTkEntry(settings_frame, width=400)
        base_url_entry.pack(fill=tk.X, padx=5, pady=5)
        base_url_entry.insert(0, self.config["base_url"])
        
        # 创建模型选择框架
        model_frame = ctk.CTkFrame(settings_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建模型标签
        model_label = ctk.CTkLabel(model_frame, text="模型:")
        model_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # 创建模型选择变量
        model_var = tk.StringVar(value=self.config["model"])
        
        # 创建模型选择按钮
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        model_buttons = []
        
        for model in models:
            button = ctk.CTkRadioButton(
                model_frame, 
                text=model, 
                variable=model_var, 
                value=model
            )
            button.pack(anchor=tk.W, padx=20, pady=2)
            model_buttons.append(button)
        
        # 创建自定义模型复选框
        use_custom_model_var = tk.BooleanVar(value=self.config["use_custom_model"])
        use_custom_model_checkbox = ctk.CTkCheckBox(
            model_frame, 
            text="自定义模型", 
            variable=use_custom_model_var,
            command=lambda: self._toggle_custom_model(
                use_custom_model_var, 
                custom_model_entry, 
                model_buttons
            )
        )
        use_custom_model_checkbox.pack(anchor=tk.W, padx=5, pady=5)
        
        # 创建自定义模型输入框
        custom_model_entry = ctk.CTkEntry(model_frame, width=200)
        custom_model_entry.pack(fill=tk.X, padx=20, pady=5)
        custom_model_entry.insert(0, self.config["custom_model"])
        
        # 初始化自定义模型状态
        self._toggle_custom_model(
            use_custom_model_var, 
            custom_model_entry, 
            model_buttons
        )
        
        # 创建线程数框架
        thread_frame = ctk.CTkFrame(settings_frame)
        thread_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建线程数标签
        thread_label = ctk.CTkLabel(thread_frame, text=f"线程数: {self.config['thread_count']}")
        thread_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # 创建线程数滑块
        thread_slider = ctk.CTkSlider(
            thread_frame, 
            from_=1, 
            to=10, 
            number_of_steps=9,
            command=lambda value: thread_label.configure(text=f"线程数: {int(value)}")
        )
        thread_slider.pack(fill=tk.X, padx=5, pady=5)
        thread_slider.set(self.config["thread_count"])
        
        # 创建速率限制框架
        rate_frame = ctk.CTkFrame(settings_frame)
        rate_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建RPM标签和输入框
        rpm_label = ctk.CTkLabel(rate_frame, text="RPM限制 (每分钟请求数):")
        rpm_label.pack(anchor=tk.W, padx=5, pady=5)
        
        rpm_entry = ctk.CTkEntry(rate_frame, width=100)
        rpm_entry.pack(anchor=tk.W, padx=20, pady=5)
        if self.config["rpm_limit"] is not None:
            rpm_entry.insert(0, str(self.config["rpm_limit"]))
        
        # 创建TPM标签和输入框
        tpm_label = ctk.CTkLabel(rate_frame, text="TPM限制 (每分钟令牌数):")
        tpm_label.pack(anchor=tk.W, padx=5, pady=5)
        
        tpm_entry = ctk.CTkEntry(rate_frame, width=100)
        tpm_entry.pack(anchor=tk.W, padx=20, pady=5)
        if self.config["tpm_limit"] is not None:
            tpm_entry.insert(0, str(self.config["tpm_limit"]))
        
        # 创建高级设置框架
        advanced_frame = ctk.CTkFrame(settings_frame)
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建温度标签
        temperature_label = ctk.CTkLabel(advanced_frame, text=f"温度: {self.config['temperature']:.1f}")
        temperature_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # 创建温度滑块
        temperature_slider = ctk.CTkSlider(
            advanced_frame, 
            from_=0, 
            to=1, 
            number_of_steps=10,
            command=lambda value: temperature_label.configure(text=f"温度: {value:.1f}")
        )
        temperature_slider.pack(fill=tk.X, padx=5, pady=5)
        temperature_slider.set(self.config["temperature"])
        
        # 创建块大小标签和输入框
        chunk_size_label = ctk.CTkLabel(advanced_frame, text="最大块大小 (字符数):")
        chunk_size_label.pack(anchor=tk.W, padx=5, pady=5)
        
        chunk_size_entry = ctk.CTkEntry(advanced_frame, width=100)
        chunk_size_entry.pack(anchor=tk.W, padx=20, pady=5)
        chunk_size_entry.insert(0, str(self.config["max_chunk_size"]))
        
        # 创建行数标签和输入框
        max_lines_label = ctk.CTkLabel(advanced_frame, text="最大行数:")
        max_lines_label.pack(anchor=tk.W, padx=5, pady=5)
        
        max_lines_entry = ctk.CTkEntry(advanced_frame, width=100)
        max_lines_entry.pack(anchor=tk.W, padx=20, pady=5)
        max_lines_entry.insert(0, str(self.config["max_lines"]))
        
        # 创建重叠大小标签和输入框
        overlap_size_label = ctk.CTkLabel(advanced_frame, text="重叠大小 (字符数):")
        overlap_size_label.pack(anchor=tk.W, padx=5, pady=5)
        
        overlap_size_entry = ctk.CTkEntry(advanced_frame, width=100)
        overlap_size_entry.pack(anchor=tk.W, padx=20, pady=5)
        overlap_size_entry.insert(0, str(self.config["overlap_size"]))
        
        # 创建按钮框架
        button_frame = ctk.CTkFrame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 创建测试API按钮
        test_api_button = ctk.CTkButton(
            button_frame, 
            text="测试API", 
            command=lambda: self._test_api(
                api_key_entry.get(),
                base_url_entry.get(),
                model_var.get() if not use_custom_model_var.get() else custom_model_entry.get(),
                float(temperature_slider.get())
            )
        )
        test_api_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 创建保存按钮
        save_button = ctk.CTkButton(
            button_frame, 
            text="保存设置", 
            command=lambda: self._save_api_settings(
                api_key_entry.get(),
                base_url_entry.get(),
                model_var.get(),
                custom_model_entry.get(),
                use_custom_model_var.get(),
                float(temperature_slider.get()),
                int(thread_slider.get()),
                rpm_entry.get(),
                tpm_entry.get(),
                chunk_size_entry.get(),
                max_lines_entry.get(),
                overlap_size_entry.get(),
                settings_window
            )
        )
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 创建取消按钮
        cancel_button = ctk.CTkButton(
            button_frame, 
            text="取消", 
            command=settings_window.destroy
        )
        cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def _toggle_custom_model(self, use_custom_model_var, custom_model_entry, model_buttons):
        """
        切换自定义模型状态
        
        Args:
            use_custom_model_var: 使用自定义模型变量
            custom_model_entry: 自定义模型输入框
            model_buttons: 模型选择按钮列表
        """
        if use_custom_model_var.get():
            custom_model_entry.configure(state=tk.NORMAL)
            for button in model_buttons:
                button.configure(state=tk.DISABLED)
        else:
            custom_model_entry.configure(state=tk.DISABLED)
            for button in model_buttons:
                button.configure(state=tk.NORMAL)
    
    def _test_api(self, api_key, base_url, model, temperature):
        """
        测试API连接
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            temperature: 温度参数
        """
        if not api_key:
            messagebox.showerror("错误", "请输入API密钥")
            return
        
        # 创建临时翻译器
        translator = Translator(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature
        )
        
        # 测试API
        success, message = translator.test_api()
        
        if success:
            messagebox.showinfo("成功", message)
        else:
            messagebox.showerror("错误", message)
    
    def _save_api_settings(self, 
                          api_key, 
                          base_url, 
                          model, 
                          custom_model, 
                          use_custom_model, 
                          temperature, 
                          thread_count, 
                          rpm_limit, 
                          tpm_limit, 
                          max_chunk_size, 
                          max_lines, 
                          overlap_size, 
                          settings_window):
        """
        保存API设置
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            custom_model: 自定义模型名称
            use_custom_model: 是否使用自定义模型
            temperature: 温度参数
            thread_count: 线程数
            rpm_limit: RPM限制
            tpm_limit: TPM限制
            max_chunk_size: 最大块大小
            max_lines: 最大行数
            overlap_size: 重叠大小
            settings_window: 设置窗口
        """
        if not api_key:
            messagebox.showerror("错误", "请输入API密钥")
            return
        
        # 解析RPM限制
        try:
            rpm_limit = int(rpm_limit) if rpm_limit else None
        except ValueError:
            messagebox.showerror("错误", "RPM限制必须是整数")
            return
        
        # 解析TPM限制
        try:
            tpm_limit = int(tpm_limit) if tpm_limit else None
        except ValueError:
            messagebox.showerror("错误", "TPM限制必须是整数")
            return
        
        # 解析最大块大小
        try:
            max_chunk_size = int(max_chunk_size)
            if max_chunk_size <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("错误", "最大块大小必须是正整数")
            return
        
        # 解析最大行数
        try:
            max_lines = int(max_lines)
            if max_lines <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("错误", "最大行数必须是正整数")
            return
        
        # 解析重叠大小
        try:
            overlap_size = int(overlap_size)
            if overlap_size < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("错误", "重叠大小必须是非负整数")
            return
        
        # 更新配置
        self.config["api_key"] = api_key
        self.config["base_url"] = base_url
        self.config["model"] = model
        self.config["custom_model"] = custom_model
        self.config["use_custom_model"] = use_custom_model
        self.config["temperature"] = temperature
        self.config["thread_count"] = thread_count
        self.config["rpm_limit"] = rpm_limit
        self.config["tpm_limit"] = tpm_limit
        self.config["max_chunk_size"] = max_chunk_size
        self.config["max_lines"] = max_lines
        self.config["overlap_size"] = overlap_size
        
        # 保存配置
        self._save_config()
        
        # 更新翻译器
        self._update_translator()
        
        # 关闭设置窗口
        settings_window.destroy()
        
        # 显示成功消息
        messagebox.showinfo("成功", "设置已保存")
    
    def _update_api_settings_from_config(self):
        """从配置更新API设置"""
        self._update_translator()
    
    def _update_translator(self):
        """更新翻译器"""
        if not self.config["api_key"]:
            return
        
        # 确定模型
        model = self.config["model"]
        if self.config["use_custom_model"] and self.config["custom_model"]:
            model = self.config["custom_model"]
        
        # 创建翻译器
        self.translator = Translator(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
            model=model,
            temperature=self.config["temperature"],
            thread_count=self.config["thread_count"],
            rpm_limit=self.config["rpm_limit"],
            tpm_limit=self.config["tpm_limit"]
        )
        
        # 设置回调
        self.translator.set_progress_callback(self._update_progress)
        self.translator.set_conversation_callback(self._update_conversation)
    
    def _load_file(self):
        """加载文件"""
        file_path = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[
                ("文本文件", "*.txt"),
                ("JSON文件", "*.json"),
                ("Markdown文件", "*.md"),
                ("Python文件", "*.py"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # 尝试以UTF-8编码读取
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # 清空文本框
            self.left_text.delete(1.0, tk.END)
            
            # 插入文本
            self.left_text.insert(tk.END, text)
            
            # 显示成功消息
            self.status_label.configure(text=f"已加载文件: {os.path.basename(file_path)}")
        
        except UnicodeDecodeError:
            # 尝试以其他编码读取
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    text = f.read()
                
                # 清空文本框
                self.left_text.delete(1.0, tk.END)
                
                # 插入文本
                self.left_text.insert(tk.END, text)
                
                # 显示成功消息
                self.status_label.configure(text=f"已加载文件: {os.path.basename(file_path)} (GBK编码)")
            
            except Exception as e:
                messagebox.showerror("错误", f"无法读取文件: {str(e)}")
        
        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件: {str(e)}")
    
    def _save_translation(self):
        """保存翻译"""
        # 获取翻译文本
        translated_text = self.right_text.get(1.0, tk.END)
        
        if not translated_text.strip():
            messagebox.showerror("错误", "没有翻译结果可保存")
            return
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存翻译",
            defaultextension=".txt",
            filetypes=[
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # 保存文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(translated_text)
            
            # 显示成功消息
            messagebox.showinfo("成功", f"翻译已保存到: {file_path}")
        
        except Exception as e:
            messagebox.showerror("错误", f"无法保存文件: {str(e)}")
    
    def _start_translation(self):
        """开始翻译"""
        # 检查API设置
        if not self.translator:
            messagebox.showerror("错误", "请先配置API设置")
            return
        
        # 获取原文
        source_text = self.left_text.get(1.0, tk.END)
        
        if not source_text.strip():
            messagebox.showerror("错误", "请输入要翻译的文本")
            return
        
        # 禁用开始按钮，启用停止按钮
        self.start_translation_button.configure(state=tk.DISABLED)
        self.stop_translation_button.configure(state=tk.NORMAL)
        
        # 清空右侧文本框
        self.right_text.delete(1.0, tk.END)
        
        # 清空对话文本框
        self.conversation_text.delete(1.0, tk.END)
        
        # 重置进度条
        self.progress_bar["value"] = 0
        
        # 更新状态
        self.status_label.configure(text="正在翻译...")
        
        # 获取是否使用多线程
        use_threading = self.use_threading_var.get()
        
        # 创建翻译线程
        self.translation_thread = threading.Thread(
            target=self._translation_thread_func,
            args=(source_text, use_threading)
        )
        self.translation_thread.daemon = True
        self.translation_thread.start()
    
    def _translation_thread_func(self, source_text, use_threading):
        """
        翻译线程函数
        
        Args:
            source_text: 源文本
            use_threading: 是否使用多线程
        """
        try:
            # 翻译文本
            translated_text = self.translator.translate(source_text, use_threading)
            
            # 在主线程中更新UI
            self.root.after(0, self._update_translation_result, translated_text)
        
        except Exception as e:
            # 在主线程中显示错误
            self.root.after(0, self._show_translation_error, str(e))
        
        finally:
            # 在主线程中恢复按钮状态
            self.root.after(0, self._reset_translation_buttons)
    
    def _update_translation_result(self, translated_text):
        """
        更新翻译结果
        
        Args:
            translated_text: 翻译后的文本
        """
        # 清空右侧文本框
        self.right_text.delete(1.0, tk.END)
        
        # 插入翻译结果
        self.right_text.insert(tk.END, translated_text)
        
        # 更新状态
        self.status_label.configure(text="翻译完成")
        
        # 重置进度条
        self.progress_bar["value"] = 100
    
    def _show_translation_error(self, error_message):
        """
        显示翻译错误
        
        Args:
            error_message: 错误消息
        """
        messagebox.showerror("翻译错误", error_message)
        
        # 更新状态
        self.status_label.configure(text="翻译出错")
    
    def _reset_translation_buttons(self):
        """重置翻译按钮状态"""
        self.start_translation_button.configure(state=tk.NORMAL)
        self.stop_translation_button.configure(state=tk.DISABLED)
    
    def _stop_translation(self):
        """停止翻译"""
        if self.translator:
            self.translator.stop()
            
            # 更新状态
            self.status_label.configure(text="翻译已停止")
    
    def _update_progress(self, progress, status, detail):
        """
        更新进度
        
        Args:
            progress: 进度（0-1）
            status: 状态信息
            detail: 详细信息
        """
        # 在主线程中更新UI
        self.root.after(0, self._update_progress_ui, progress, status, detail)
    
    def _update_progress_ui(self, progress, status, detail):
        """
        在UI中更新进度
        
        Args:
            progress: 进度（0-1）
            status: 状态信息
            detail: 详细信息
        """
        # 更新进度条
        self.progress_bar["value"] = progress * 100
        
        # 更新状态标签
        self.status_label.configure(text=status)
        
        # 更新详细进度标签
        self.detail_label.configure(text=detail)
    
    def _update_conversation(self, role, content):
        """
        更新对话
        
        Args:
            role: 角色（"user"或"assistant"）
            content: 内容
        """
        # 在主线程中更新UI
        self.root.after(0, self._update_conversation_ui, role, content)
    
    def _update_conversation_ui(self, role, content):
        """
        在UI中更新对话
        
        Args:
            role: 角色（"user"或"assistant"）
            content: 内容
        """
        # 插入角色
        self.conversation_text.insert(tk.END, f"{role.upper()}:\n", role)
        
        # 插入内容
        self.conversation_text.insert(tk.END, f"{content}\n\n")
        
        # 滚动到底部
        self.conversation_text.see(tk.END)
    
    def run(self):
        """运行应用程序"""
        self.root.mainloop()

if __name__ == "__main__":
    app = TranslatorApp()
    app.run()
