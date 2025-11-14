#!/usr/bin/env python3
"""
将 .pkl 文件转换为 .jsonl 文件
支持将 DatasetInfo 对象列表转换为 JSONL 格式
"""
import pickle
import json
import os
import argparse
from datetime import datetime, date
from typing import Any


def convert_to_serializable(obj: Any) -> Any:
    """
    将对象转换为 JSON 可序列化的格式
    
    Args:
        obj: 要转换的对象
        
    Returns:
        JSON 可序列化的对象
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [convert_to_serializable(item) for item in sorted(obj)]
    elif hasattr(obj, '__dict__'):
        # 对象类型，转换为字典
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # 跳过私有属性
                result[key] = convert_to_serializable(value)
        return result
    else:
        # 其他类型，尝试转换为字符串
        return str(obj)


def dataset_info_to_dict(info) -> dict:
    """
    将 DatasetInfo 对象转换为字典
    
    Args:
        info: DatasetInfo 对象
        
    Returns:
        包含数据集信息的字典
    """
    result = {}
    
    # 提取常见属性
    common_attrs = [
        'id', 'author', 'downloads', 'downloads_all_time', 'likes',
        'created_at', 'last_modified', 'description', 'disabled',
        'private', 'gated', 'key', 'sha', 'siblings', 'tags'
    ]
    
    for attr in common_attrs:
        if hasattr(info, attr):
            value = getattr(info, attr)
            result[attr] = convert_to_serializable(value)
    
    # 处理 cardData 和 card_data
    if hasattr(info, 'cardData') and info.cardData:
        result['cardData'] = convert_to_serializable(info.cardData)
    elif hasattr(info, 'card_data') and info.card_data:
        result['cardData'] = convert_to_serializable(info.card_data)
    
    # 处理 tags - 转换为字典格式（如果存在）
    if hasattr(info, 'tags') and info.tags:
        tag_dict = {}
        for tag in info.tags:
            if ':' in tag:
                category, value = tag.split(':', 1)
                category = category.strip()
                value = value.strip()
                if category in tag_dict:
                    if isinstance(tag_dict[category], list):
                        tag_dict[category].append(value)
                    else:
                        tag_dict[category] = [tag_dict[category], value]
                else:
                    tag_dict[category] = value
            else:
                tag_dict[tag] = tag
        result['tags_dict'] = tag_dict
    
    # 提取所有其他非私有属性
    if hasattr(info, '__dict__'):
        for key, value in info.__dict__.items():
            if not key.startswith('_') and key not in result:
                result[key] = convert_to_serializable(value)
    
    return result


def pkl_to_jsonl(pkl_path: str, jsonl_path: str, convert_dataset_info: bool = True):
    """
    将 .pkl 文件转换为 .jsonl 文件
    
    Args:
        pkl_path: 输入的 .pkl 文件路径
        jsonl_path: 输出的 .jsonl 文件路径
        convert_dataset_info: 是否将 DatasetInfo 对象转换为字典格式
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"文件不存在: {pkl_path}")
    
    print(f"正在加载 .pkl 文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"加载成功，数据类型: {type(data).__name__}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理数据
    if isinstance(data, list):
        print(f"列表长度: {len(data)}")
        items = data
    elif isinstance(data, dict):
        print(f"字典键数量: {len(data)}")
        items = list(data.values())
    else:
        items = [data]
    
    print(f"正在转换为 JSONL 格式...")
    count = 0
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in items:
            if convert_dataset_info and hasattr(item, 'id'):
                # 假设是 DatasetInfo 对象
                json_obj = dataset_info_to_dict(item)
            else:
                # 通用对象转换
                json_obj = convert_to_serializable(item)
            
            f.write(json.dumps(json_obj, ensure_ascii=False, default=str) + '\n')
            count += 1
            
            if count % 100 == 0:
                print(f"已处理 {count} 条记录...")
    
    print(f"转换完成！共转换 {count} 条记录")
    print(f"输出文件: {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 .pkl 文件转换为 .jsonl 文件"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="输入的 .pkl 文件路径"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help="输出的 .jsonl 文件路径"
    )
    parser.add_argument(
        '--no-convert-dataset-info',
        action='store_true',
        help="不进行 DatasetInfo 特殊转换，使用通用转换"
    )
    
    args = parser.parse_args()
    
    try:
        pkl_to_jsonl(
            args.input,
            args.output,
            convert_dataset_info=not args.no_convert_dataset_info
        )
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

