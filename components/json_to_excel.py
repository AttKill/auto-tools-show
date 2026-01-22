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
            text = node['data']['text'].strip()
            if text:
                # 清理when/then前缀
                clean_text = text.replace("When:", "").replace("Then:", "").replace("when:", "").replace("then:",
                                                                                                         "").strip()
                path.append(clean_text)

        # 如果是叶子节点（没有子节点或子节点为空）
        if not node.get('children') or len(node['children']) == 0:
            # 确保路径至少有3个节点
            if len(path) >= 3:
                # 构建given列：从第二个节点开始，排除最后两个
                given_parts = []
                for i in range(1, len(path) - 2):
                    given_parts.append(path[i])

                given = "-".join(given_parts) if given_parts else ""

                # 获取when和then（倒数第二和最后一个）
                when = path[-2] if len(path) >= 2 else ""
                then = path[-1] if len(path) >= 1 else ""

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


# 使用示例
def example_usage(json_str: str):
    """使用示例"""

    # 样例JSON数据
    json_str = """{"root":{"data":{"text":"测试用例"},
     "children":[
     {"data":{"text":"用户登录","font-size":32},"children":[{"data":{"text":"成功登录","priority":3},"children":[{"data":{"text":"When: 输入正确用户名和密码"},"children":[{"data":{"text":"Then: 跳转到首页"},"children":[]}]}]},
     {"data":{"text":"失败登录"},"children":[{"data":{"text":"When: 输入错误密码"},"children":[{"data":{"text":"Then: 显示错误提示"},"children":[]}]}]}]},
     {"data":{"text":"购物车","progress":3},"children":[{"data":{"text":"添加商品"},"children":[{"data":{"text":"When: 点击加入购物车按钮"},"children":[{"data":{"text":"Then: 商品数量增加"},"children":[]}]}]},
     {"data":{"text":"删除商品"},"children":[{"data":{"text":"When: 点击删除按钮"},"children":[{"data":{"text":"Then: 商品从购物车移除"},"children":[]}]}]}]}]},"template":"right","theme":"fresh-blue","version":"1.4.43"}"""

    print("JSON转换示例:")
    print("=" * 60)

    # 1. 使用JSON字符串
    sheets_data = read_json_to_excel(json_str)

    for sheet_name, df in sheets_data.items():
        print(f"\nSheet名称: {sheet_name}")
        print(df.to_string(index=False))
        print(f"\n数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 显示每行数据
        print("\n详细数据:")
        for idx, row in df.iterrows():
            print(f"行 {idx}: given='{row['given']}', when='{row['when']}', then='{row['then']}'")

    return sheets_data


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


# 测试函数
if __name__ == "__main__":
    # 运行示例
    sheets_data = example_usage()

    # 保存到Excel（如果需要）
    save_to_excel(sheets_data, 'output.xlsx')

    # 验证输出是否符合要求
    print("\n" + "=" * 60)
    print("验证输出格式:")

    for sheet_name, df in sheets_data.items():
        print(f"\nSheet: {sheet_name}")
        print("-" * 60)
        print(df.to_string(index=False))