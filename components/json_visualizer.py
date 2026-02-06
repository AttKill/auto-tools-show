import json
import traceback
from typing import List

import pandas as pd

from components.format_json import format_output


def json_to_excel_simple(json_data: dict) -> dict:
    """
    将JSON树形结构转换为Excel格式，sheet名为root节点的text

    Args:
        json_data: JSON数据字典

    Returns:
        dict: 包含show_parse_json_dict和sheets_data的字典格式
    """

    def get_sheet_name(root_node: dict) -> str:
        """获取sheet名称（root节点的text）"""
        if 'data' in root_node and 'text' in root_node['data']:
            return root_node['data']['text'].strip()
        return "Sheet1"

    def traverse(
            node: dict,
            path: List[str] = None,
            results: List[dict] = None,
            show_parse_json_dict: dict = None
    ) -> tuple:
        """
        递归遍历所有节点，收集结果

        Args:
            node: 当前节点
            path: 当前路径
            results: 结果列表
            show_parse_json_dict: 解析结果字典

        Returns:
            tuple: (结果列表, 解析结果字典)
        """
        if path is None:
            path = []
        if results is None:
            results = []

        if show_parse_json_dict is None:
            show_parse_json_dict = {
                "table_heads": [],
                "show_data": {}
            }

        # 获取当前节点的文本和资源标签
        if 'data' in node and 'text' in node['data']:
            node_data = node['data']
            text = node_data['text'].strip()
            resources = node_data.get('resource', [])

            # 收集所有资源标签
            for resource_item in resources:
                if resource_item not in show_parse_json_dict["table_heads"]:
                    show_parse_json_dict["table_heads"].append(resource_item)

            if text:
                # 将资源标签追加到文本后，方便后续分组处理
                clean_text = text
                if resources and len(resources) > 0:
                    clean_text += "".join(resources)
                path.append(clean_text)

        # 如果是叶子节点（没有子节点或子节点为空）
        if not node.get('children') or len(node['children']) == 0:
            # 如果没有标签，直接返回
            if len(show_parse_json_dict["table_heads"]) == 0:
                return results, show_parse_json_dict

            # 有标签的情况处理
            num_columns = len(show_parse_json_dict["table_heads"])
            min_path_length = num_columns

            # 如果路径长度不够，使用空字符串填充
            if len(path) < min_path_length:
                extended_path = path + [""] * (min_path_length - len(path))
            else:
                extended_path = path

            # 初始化每列的数据为默认值
            column_parts = {col: "/" for col in show_parse_json_dict["table_heads"]}

            # 处理每个路径元素
            for path_element in extended_path:
                content_without_labels = path_element
                matched_labels = []

                # 找出所有匹配的标签
                for label in show_parse_json_dict["table_heads"]:
                    if label.lower() in path_element.lower():
                        matched_labels.append(label)
                        # 移除标签名称，保留纯内容
                        content_without_labels = content_without_labels.replace(label, "", 1).strip()

                # 如果有匹配的标签，将内容分配给匹配的标签列
                if matched_labels:
                    for label in matched_labels:
                        content = content_without_labels if content_without_labels else "/"

                        # 如果该列已有内容，用"-"连接
                        if column_parts.get(label, "/") != "/":
                            if content != "/":
                                column_parts[label] = f"{column_parts[label]}-{content}"
                        else:
                            column_parts[label] = content
                # 如果没有匹配到标签，跳过该路径元素（移除原有的找父节点逻辑）

            # 确保所有列都有值（即使是"/"）
            for label in show_parse_json_dict["table_heads"]:
                if label not in column_parts:
                    column_parts[label] = "/"

            # 添加到结果
            results.append(column_parts)

            # 更新show_data
            for table_head, value in column_parts.items():
                if table_head not in show_parse_json_dict["show_data"]:
                    show_parse_json_dict["show_data"][table_head] = []
                show_parse_json_dict["show_data"][table_head].append(value)

        else:
            # 继续遍历子节点
            for child in node['children']:
                traverse(child, path.copy(), results, show_parse_json_dict)

        return results, show_parse_json_dict

    try:

        json_data, labels_0 = format_output(json_data, 1)
        # 获取根节点
        if 'root' in json_data:
            root = json_data['root']
        else:
            root = json_data

        # 获取sheet名称
        sheet_name = get_sheet_name(root)

        # 开始遍历，收集所有结果
        results, show_parse_json_dict = traverse(root)

        show_parse_json_dict["add_icon_items"]=labels_0

        # 确保所有标签列的数据长度一致
        expected_length = len(results)
        for table_head in show_parse_json_dict["table_heads"]:
            if table_head not in show_parse_json_dict["show_data"]:
                show_parse_json_dict["show_data"][table_head] = ["/"] * expected_length
            else:
                current_length = len(show_parse_json_dict["show_data"][table_head])
                if current_length < expected_length:
                    # 需要在开头补齐缺失的数据
                    missing_count = expected_length - current_length
                    show_parse_json_dict["show_data"][table_head] = \
                        ["/"] * missing_count + show_parse_json_dict["show_data"][table_head]

        # 创建DataFrame
        if results:
            df = pd.DataFrame(results)

            # 确保列顺序按table_heads的顺序
            column_order = []
            for col in show_parse_json_dict["table_heads"]:
                if col in df.columns:
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

        return result_dict

    except Exception as e:
        raise Exception(f"处理JSON失败: {str(e)}")


def read_json_to_excel(json_input) -> dict:
    """
    读取JSON并转换为Excel格式数据

    Args:
        json_input: 可以是文件路径、文件缓冲区或JSON字符串

    Returns:
        dict: {sheet_name: DataFrame} 格式

    Raises:
        Exception: 当JSON解析失败时
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


def save_to_excel(sheets_data: dict, output_path: str) -> None:
    """
    将转换后的数据保存到Excel文件

    Args:
        sheets_data: read_json_to_excel返回的数据
        output_path: 输出文件路径
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"文件已保存: {output_path}")