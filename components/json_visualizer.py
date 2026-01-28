import json
import traceback
from typing import List

import pandas as pd


def json_to_excel_simple(json_data: dict) -> dict:
    """
    将JSON树形结构转换为Excel格式，sheet名为root节点的text

    返回:
        dict: {sheet_name: DataFrame} 格式，与read_excel函数保持一致
    """

    def get_sheet_name(root_node: dict) -> str:
        """获取sheet名称（root节点的text）"""
        if 'data' in root_node and 'text' in root_node['data']:
            return root_node['data']['text'].strip()
        return "Sheet1"

    def traverse(node: dict, path: List[str] = None, results: List[dict] = None, show_parse_json_dict: dict = None):
        """递归遍历所有节点，收集结果"""
        if path is None:
            path = []
        if results is None:
            results = []

        if show_parse_json_dict is None:
            show_parse_json_dict = dict()
            show_parse_json_dict["table_heads"] = []
            show_parse_json_dict["show_data"] = dict()

        # 获取当前节点的文本
        if 'data' in node and 'text' in node['data']:
            node_data = node['data']
            text = node_data['text'].strip()
            resources = node_data.get('resource', [])
            for resource_item in resources:
                if resource_item not in show_parse_json_dict["table_heads"]:
                    show_parse_json_dict["table_heads"].append(resource_item)
            if text:
                # 清理when/then前缀
                clean_text = text
                if resources is not None and len(resources) > 0:
                    ## 将标签添加到文本上，方便后面分组
                    clean_text += "".join(resources)
                path.append(clean_text)

        ## 设置默认的标签(如果使用默认标签，最后两个节点对应when和then,其他节点内容拼接到given上)
        if len(show_parse_json_dict["table_heads"]) == 0:
            show_parse_json_dict["table_heads"].append("given")
            show_parse_json_dict["table_heads"].append("when")
            show_parse_json_dict["table_heads"].append("then")

        # 如果是叶子节点（没有子节点或子节点为空）
        if not node.get('children') or len(node['children']) == 0:
            # 确保路径至少有3个节点
            if len(path) >= 3:
                # 计算需要分配的列数
                num_columns = len(show_parse_json_dict["table_heads"])
                min_path_length = num_columns

                # 如果路径长度不够，使用默认值
                if len(path) < min_path_length:
                    # 扩展路径，使用空字符串填充
                    extended_path = path + [""] * (min_path_length - len(path))
                else:
                    extended_path = path

                # 构建每列的数据
                column_parts = {}
                for idx, column_name in enumerate(show_parse_json_dict["table_heads"]):
                    if idx == 0:  # 第一列，通常为given
                        # 从第二个节点开始到倒数第(num_columns-1)个节点
                        start_idx = 1
                        end_idx = len(extended_path) - (num_columns - 1)
                        parts = []
                        for i in range(start_idx, max(start_idx, end_idx)):
                            clean_text_with_prefix = extended_path[i]
                            clean_text = clean_text_with_prefix.replace("given", "").replace("when", "").replace("then",
                                                                                                                 "").strip()
                            if column_name in clean_text_with_prefix.lower():
                                parts.append(clean_text)
                            else:
                                parts.append(clean_text_with_prefix)
                        column_value = "-".join(parts) if parts else ""
                    elif idx == num_columns - 1:  # 最后一列，通常为then
                        # 使用最后一个路径元素
                        column_value = extended_path[-1]
                    else:  # 中间列，通常为when等
                        # 使用对应位置的路径元素
                        pos = -(num_columns - idx)
                        if abs(pos) <= len(extended_path):
                            column_value = extended_path[pos]
                        else:
                            column_value = ""

                    column_parts[column_name] = column_value

                line_data_dict = column_parts

                # 添加到结果
                results.append(line_data_dict)

                for table_head in show_parse_json_dict["table_heads"]:
                    if table_head not in show_parse_json_dict["show_data"]:
                        show_parse_json_dict["show_data"][table_head] = []
                    show_parse_json_dict["show_data"][table_head].append(line_data_dict.get(table_head, "/"))

        else:
            # 继续遍历子节点
            for child in node['children']:
                traverse(child, path.copy(), results, show_parse_json_dict)

        return results, show_parse_json_dict

    try:
        # 获取根节点
        if 'root' in json_data:
            root = json_data['root']
        else:
            root = json_data

        # 获取sheet名称
        sheet_name = get_sheet_name(root)

        # 开始遍历，收集所有结果
        results, show_parse_json_dict = traverse(root)

        # 在函数结尾添加
        expected_length = len(results)
        for table_head in show_parse_json_dict["table_heads"]:
            current_length = len(show_parse_json_dict["show_data"][table_head])
            if current_length < expected_length:
                # 补齐缺失的数据
                show_parse_json_dict["show_data"][table_head].extend(["/"] * (expected_length - current_length))

        # 创建DataFrame
        df = pd.DataFrame(results)

        # 按given排序
        sort_field = show_parse_json_dict.get("table_heads", [])
        if not df.empty and len(sort_field) != 0:
            df = df.sort_values(sort_field[0]).reset_index(drop=True)

        result_dict = {
            "show_parse_json_dict": show_parse_json_dict,
            "sheets_data": {sheet_name: df}
        }
        # 返回与read_excel函数一致的格式
        return result_dict

    except Exception as e:
        raise Exception(f"处理JSON失败: {str(e)}")


# 主要处理函数
def read_json_to_excel(json_input):
    """
    读取JSON并转换为Excel格式数据

    参数:
        json_input: 可以是文件路径、文件缓冲区或JSON字符串

    返回:
        dict: {sheet_name: DataFrame} 格式
    """
    try:
        # 根据输入类型解析JSON
        if isinstance(json_input, str):
            # 检查是否为文件路径
            try:
                with open(json_input, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (FileNotFoundError, OSError):
                # 如果不是文件路径，则当作JSON字符串处理
                data = json.loads(json_input)
        elif hasattr(json_input, 'read'):
            # 如果是文件缓冲区
            content = json_input.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            data = json.loads(content)
        else:
            # 直接是字典
            data = json_input

        # 转换为Excel格式
        json2excel_dict = json_to_excel_simple(data)

        return json2excel_dict

    except Exception as e:
        error_details = traceback.format_exc()
        raise Exception(f"读取JSON文件失败: {str(e)}\n详细错误信息:\n{error_details}")


# 保存到Excel文件的函数
def save_to_excel(sheets_data: dict, output_path: str):
    """
    将转换后的数据保存到Excel文件

    参数:
        sheets_data: read_json_to_excel返回的数据
        output_path: 输出文件路径
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"文件已保存: {output_path}")
