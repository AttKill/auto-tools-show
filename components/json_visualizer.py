import json
import traceback
from typing import List

import pandas as pd


def find_parent_label(path: List[str], table_heads: List[str]) -> str:
    """
    查找路径中最近的带有标签的父节点
    """
    # 从倒数第二个节点开始回溯（排除当前节点）
    for i in range(len(path) - 2, -1, -1):  # 修改：从父节点开始回溯
        for label in table_heads:
            if label.lower() in path[i].lower():
                return label
    return None  # 如果没找到，返回 None


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

        # 如果是叶子节点（没有子节点或子节点为空）
        if not node.get('children') or len(node['children']) == 0:
            # 计算需要分配的列数
            # 如果table_heads为空，说明所有节点都没有标签
            if len(show_parse_json_dict["table_heads"]) == 0:
                # 所有数据都应该在other列
                column_parts = {"other": "/"}

                # 构建路径内容
                content_parts = []
                for path_element in path:
                    # 移除可能的标签（虽然这里没有标签）
                    clean_element = path_element
                    # 将所有内容添加到other列
                    content_parts.append(clean_element)

                if content_parts:
                    column_parts["other"] = "-".join(content_parts)

                line_data_dict = column_parts
                results.append(line_data_dict)

                # 确保other在table_heads中
                if "other" not in show_parse_json_dict["table_heads"]:
                    show_parse_json_dict["table_heads"].append("other")

                # 添加到show_data
                if "other" not in show_parse_json_dict["show_data"]:
                    show_parse_json_dict["show_data"]["other"] = []
                show_parse_json_dict["show_data"]["other"].append(line_data_dict.get("other", "/"))

            else:
                # 原始逻辑：有标签的情况
                num_columns = len(show_parse_json_dict["table_heads"])
                min_path_length = num_columns

                # 如果路径长度不够，使用默认值
                if len(path) < min_path_length:
                    # 扩展路径，使用空字符串填充
                    extended_path = path + [""] * (min_path_length - len(path))
                else:
                    extended_path = path

                # 构建每列的数据
                column_parts = {col: "/" for col in show_parse_json_dict["table_heads"]}
                # 处理每个路径元素
                for path_element in extended_path:
                    # 检查此路径元素包含哪些标签
                    content_without_labels = path_element
                    matched_labels = []

                    # 找出所有匹配的标签
                    for label in show_parse_json_dict["table_heads"]:
                        if label.lower() in path_element.lower():
                            matched_labels.append(label)
                            # 移除标签名称，保留纯内容
                            content_without_labels = content_without_labels.replace(label, "", 1).strip()

                    # 如果没有匹配到标签，需要决定如何分配
                    if not matched_labels:
                        # 分配到父节点的标签列
                        parent_label = find_parent_label(path, show_parse_json_dict["table_heads"])
                        if parent_label:
                            matched_labels.append(parent_label)
                        else:
                            # 如果到起始节点都没有标签，分配到 other 列
                            matched_labels.append("other")

                    # 将内容分配给所有匹配的标签列
                    for label in matched_labels:
                        content = content_without_labels if content_without_labels else "/"

                        # 如果该列已有内容，用"-"连接
                        if column_parts.get(label, "/") != "/":
                            if content != "/":
                                column_parts[label] = f"{column_parts[label]}-{content}"
                            # 如果内容是"/"，保持原有内容
                        else:
                            column_parts[label] = content

                # 确保所有列都有值（即使是"/"）
                for label in show_parse_json_dict["table_heads"]:
                    if label not in column_parts:
                        column_parts[label] = "/"

                # 特殊处理 other 列：只有在 other 列有数据时才保留
                if "other" in column_parts and column_parts["other"] == "/":
                    del column_parts["other"]
                elif "other" in column_parts:
                    # 如果other列有数据，确保它也在table_heads中
                    if "other" not in show_parse_json_dict["table_heads"]:
                        show_parse_json_dict["table_heads"].append("other")

                line_data_dict = column_parts

                # 添加到结果
                results.append(line_data_dict)

                for table_head in column_parts.keys():
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
            if table_head not in show_parse_json_dict["show_data"]:
                show_parse_json_dict["show_data"][table_head] = ["/"] * expected_length
            else:
                current_length = len(show_parse_json_dict["show_data"][table_head])
                if current_length < expected_length:
                    # 补齐缺失的数据
                    show_parse_json_dict["show_data"][table_head].extend(["/"] * (expected_length - current_length))

        # 创建DataFrame
        if results:
            df = pd.DataFrame(results)

            # 确保列顺序：先other列（如果有），然后是按出现顺序的其他列
            column_order = []
            if "other" in df.columns:
                column_order.append("other")
            for col in show_parse_json_dict["table_heads"]:
                if col != "other" and col in df.columns:
                    column_order.append(col)

            # 重新排列列顺序
            df = df[column_order] if column_order else df

            # 按第一个字段排序（如果有）
            if not df.empty and len(df.columns) > 0:
                first_column = df.columns[0]
                df = df.sort_values(first_column).reset_index(drop=True)
        else:
            df = pd.DataFrame()

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