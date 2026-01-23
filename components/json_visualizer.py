import json
import pandas as pd
from typing import List


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

    def traverse(node: dict, path: List[str] = None, results: List[dict] = None):
        """递归遍历所有节点，收集结果"""
        if path is None:
            path = []
        if results is None:
            results = []

        # 获取当前节点的文本
        if 'data' in node and 'text' in node['data']:
            node_data = node['data']
            text = node_data['text'].strip()
            resources = node_data.get('resource',[])
            if text:
                # 清理when/then前缀
                clean_text = text.replace("Given:", "").replace("when:", "").replace("then:","").strip()
                if resources is not None and len(resources) > 0:
                    clean_text += "".join(resources)
                path.append(clean_text)

        # 如果是叶子节点（没有子节点或子节点为空）
        if not node.get('children') or len(node['children']) == 0:
            # 确保路径至少有3个节点
            if len(path) >= 3:

                # 构建given列：从第二个节点开始，排除最后两个
                given_parts = []
                when_parts = []
                for i in range(1, len(path) - 2):
                    clean_text_with_prefix = path[i]
                    clean_text = clean_text_with_prefix.replace("given", "").replace("when", "").replace("then", "").strip()
                    if "given" in clean_text_with_prefix:
                        given_parts.append(clean_text)
                    elif "when" in clean_text_with_prefix:
                        when_parts.append(clean_text)
                    else:
                        given_parts.append(clean_text_with_prefix)
                print(f"given_parts:{given_parts}")
                given = "-".join(given_parts) if given_parts else ""

                # 获取when,默认倒数第二个
                when_parts.append(path[-2])
                when = "-".join(when_parts)

                # 获取then,默认最后一个
                then = path[-1]

                # 添加到结果
                results.append({
                    'given': given,
                    'when': when,
                    'then': then
                })
        else:
            # 继续遍历子节点
            for child in node['children']:
                traverse(child, path.copy(), results)

        return results

    try:
        # 获取根节点
        if 'root' in json_data:
            root = json_data['root']
        else:
            root = json_data

        # 获取sheet名称
        sheet_name = get_sheet_name(root)

        # 开始遍历，收集所有结果
        results = traverse(root)

        # 创建DataFrame
        df = pd.DataFrame(results)

        # 按given排序
        if not df.empty:
            df = df.sort_values('given').reset_index(drop=True)

        # 返回与read_excel函数一致的格式
        return {sheet_name: df}

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
        sheets_data = json_to_excel_simple(data)

        return sheets_data

    except Exception as e:
        raise Exception(f"读取JSON文件失败: {str(e)}")


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
