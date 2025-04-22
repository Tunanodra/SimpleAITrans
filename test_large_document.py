"""
测试超长文本翻译工具的性能和内存使用情况
"""

import os
import sys
import time
import random
import gc
import psutil
import string

def generate_large_document(file_path, size_chars=333000):
    """
    生成一个大型测试文档
    
    Args:
        file_path: 文件路径
        size_chars: 文档大小（字符数）
    """
    print(f"正在生成大小为 {size_chars} 字符的测试文档...")
    
    # 创建一些示例段落
    paragraphs = [
        "This is a test document for the translator application. It contains various types of content including normal text, code blocks, and special characters.",
        "The translator should be able to handle this document efficiently without running out of memory.",
        "Let's test how well the application performs with a very large document.",
        "This document will contain repeated content to reach the desired size.",
        "Here's a code block example:\n```python\ndef hello_world():\n    print('Hello, world!')\n    return True\n```",
        "And here's some JSON content:\n```json\n{\n    \"name\": \"Test Document\",\n    \"size\": \"very large\",\n    \"purpose\": \"testing\"\n}\n```",
        "Special characters test: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "Numbers: 0123456789",
        "Let's also include some longer paragraphs to make the test more realistic. This paragraph will contain multiple sentences to simulate a real document. The sentences will vary in length and structure to provide a good test case for the translator. We want to ensure that the context preservation works correctly across sentence boundaries.",
        "Technical terms should also be included: API, JSON, Python, OpenAI, GPT, tokenization, asynchronous, multithreading, memory optimization, context preservation, rate limiting.",
    ]
    
    # 创建一些带有控制字符的内容
    control_chars = [
        "\t\tIndented text with tabs",
        "Line 1\nLine 2\nLine 3",
        "Column 1\tColumn 2\tColumn 3",
        "\r\nCarriage return and newline",
        "\u001B[31mThis would be red text in a terminal\u001B[0m",
    ]
    
    # 合并所有内容
    all_content = paragraphs + control_chars
    
    # 生成随机文本直到达到所需大小
    with open(file_path, 'w', encoding='utf-8') as f:
        current_size = 0
        
        # 先写入一些结构化内容
        for _ in range(10):
            for content in all_content:
                f.write(content + "\n\n")
                current_size += len(content) + 2
        
        # 然后添加随机文本直到达到所需大小
        while current_size < size_chars:
            # 生成随机段落
            paragraph_length = random.randint(50, 200)
            words = []
            for _ in range(paragraph_length):
                word_length = random.randint(3, 12)
                word = ''.join(random.choice(string.ascii_letters) for _ in range(word_length))
                words.append(word)
            
            paragraph = ' '.join(words)
            f.write(paragraph + "\n\n")
            current_size += len(paragraph) + 2
            
            # 每隔一段时间添加一个代码块或特殊内容
            if random.random() < 0.1:
                special_content = random.choice(all_content)
                f.write(special_content + "\n\n")
                current_size += len(special_content) + 2
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"测试文档已生成: {file_path}")
    print(f"文件大小: {file_size_mb:.2f} MB")
    print(f"字符数: 约 {current_size}")

def monitor_memory_usage():
    """
    监控内存使用情况
    
    Returns:
        memory_info: 内存使用信息字典
    """
    process = psutil.Process(os.getpid())
    memory_info = {
        'rss': process.memory_info().rss / (1024 * 1024),  # RSS in MB
        'vms': process.memory_info().vms / (1024 * 1024),  # VMS in MB
    }
    return memory_info

def main():
    """主函数"""
    # 生成测试文档
    test_file = "large_test_document.txt"
    generate_large_document(test_file, size_chars=333000)
    
    # 显示初始内存使用情况
    initial_memory = monitor_memory_usage()
    print(f"初始内存使用: RSS={initial_memory['rss']:.2f} MB, VMS={initial_memory['vms']:.2f} MB")
    
    # 提示用户
    print("\n测试文档已准备就绪。请使用翻译工具打开并翻译此文档。")
    print("翻译过程中，请观察内存使用情况和翻译质量。")
    print(f"测试文档路径: {os.path.abspath(test_file)}")

if __name__ == "__main__":
    try:
        # 检查是否安装了psutil
        import psutil
    except ImportError:
        print("请先安装psutil: pip install psutil")
        sys.exit(1)
    
    main()
