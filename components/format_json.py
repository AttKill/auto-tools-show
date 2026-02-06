import copy
from typing import Dict, List, Tuple


def format_json4icon(json_data: dict, format_type: int = 0) -> Tuple[Dict, List[str]]:
    """
    根据指定类型格式化JSON数据

    Args:
        json_data: 输入的JSON数据字典
        format_type: 格式化类型
            0 - 合并节点（继承父节点标签）
            1 - 独立节点（根据层级自定义标签）

    Returns:
        tuple: (格式化后的JSON数据, 节点标签数组)
    """

    def merge_node_icon(json_data: dict) -> Tuple[Dict, List[str]]:
        """
        合并节点格式化：继承父节点标签

        Args:
            json_data: 输入的JSON数据字典

        Returns:
            tuple: (格式化后的JSON数据, 节点标签数组)
        """
        node_labels = []

        def traverse_and_complete(node: dict, parent_resource: list = None) -> None:
            """递归遍历节点并补全resource字段"""
            if 'data' not in node:
                return

            data = node['data']

            # 检查当前节点是否有resource
            if 'resource' not in data:
                if parent_resource is not None:
                    # 非根节点：直接继承父节点的resource
                    data['resource'] = parent_resource.copy()
                else:
                    # 根节点：添加默认的root node标签
                    data['resource'] = ["root node"]

                # 添加到节点标签数组
                text = data.get('text', '')
                if text:
                    label = data['resource'][0] if data['resource'] else "unknown"
                    node_labels.append(f"{label}:{text}")
            else:
                # 如果已有resource，确保它是列表类型
                if not isinstance(data['resource'], list):
                    data['resource'] = [data['resource']]
                # 已有resource的节点不添加到node_labels中
                # 跳过，不记录

            # 获取当前节点的resource，用于传递给子节点
            current_resource = data.get('resource', [])

            # 递归处理子节点
            if 'children' in node and node['children']:
                for child in node['children']:
                    traverse_and_complete(child, current_resource)

        # 创建数据的深拷贝
        result_data = copy.deepcopy(json_data)

        # 获取根节点
        if 'root' in result_data:
            root_node = result_data['root']
            traverse_and_complete(root_node, None)
        else:
            traverse_and_complete(result_data, None)

        return result_data, node_labels

    def alone_node_icon(json_data: dict) -> Tuple[Dict, List[str]]:
        """
        独立节点格式化：根据层级自定义标签

        Args:
            json_data: 输入的JSON数据字典

        Returns:
            tuple: (格式化后的JSON数据, 节点标签数组)
        """
        node_labels = []

        def traverse_and_complete(node: dict, level: int = 0, parent_resource: list = None) -> None:
            """递归遍历节点并补全resource字段（独立节点模式）"""
            if 'data' not in node:
                return

            data = node['data']
            text = data.get('text', '')

            # 检查当前节点是否有resource
            if 'resource' not in data:
                if level == 0:
                    # 第0层：根节点
                    data['resource'] = ["root node"]
                elif level == 1:
                    # 第1层：一级子节点
                    data['resource'] = ["sub_node-01"]
                elif level == 2:
                    # 第2层：二级子节点
                    data['resource'] = ["sub_node-02"]
                elif level == 3:
                    # 第3层：三级子节点
                    data['resource'] = ["sub_node-03"]
                else:
                    # 第4层及以上：统一格式
                    data['resource'] = [f"sub_node-{level:02d}"]

                # 添加到节点标签数组
                if text:
                    label = data['resource'][0]
                    node_labels.append(f"{label}:{text}")
            else:
                # 如果已有resource，确保它是列表类型
                if not isinstance(data['resource'], list):
                    data['resource'] = [data['resource']]
                # 已有resource的节点不添加到node_labels中
                # 跳过，不记录

            # 递归处理子节点（不传递resource，只传递层级）
            if 'children' in node and node['children']:
                for child in node['children']:
                    traverse_and_complete(child, level + 1, None)

        # 创建数据的深拷贝
        result_data = copy.deepcopy(json_data)

        # 获取根节点
        if 'root' in result_data:
            root_node = result_data['root']
            traverse_and_complete(root_node, 0, None)
        else:
            traverse_and_complete(result_data, 0, None)

        return result_data, node_labels

    try:
        if format_type == 0:
            return merge_node_icon(json_data)
        elif format_type == 1:
            return alone_node_icon(json_data)
        else:
            raise ValueError(f"不支持的格式化类型: {format_type}")

    except Exception as e:
        raise Exception(f"格式化JSON失败: {str(e)}")


def format_output(json_data: dict, format_type: int = 0) -> None:
    """
    格式化JSON并打印结果

    Args:
        json_data: 输入的JSON数据字典
        format_type: 格式化类型（0-合并节点，1-独立节点）
    """
    try:
        # 格式化JSON
        formatted_json, node_labels = format_json4icon(json_data, format_type)

        for label in node_labels:
            print(f'  "{label}"')

        return formatted_json, node_labels

    except Exception as e:
        print(f"格式化失败: {e}")


# 测试数据
test_json = {
    "root": {
        "data": {
            "text": "中心主题"
        },
        "children": [
            {
                "data": {
                    "text": "AAA",
                    "resource": ["Given"]
                },
                "children": [
                    {
                        "data": {
                            "text": "BBB"
                        },
                        "children": [
                            {
                                "data": {
                                    "text": "CCC",
                                    "resource": ["When"]
                                },
                                "children": [
                                    {
                                        "data": {
                                            "text": "DDD",
                                            "resource": ["Then"]
                                        },
                                        "children": []
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    },
    "template": "right",
    "theme": "fresh-blue",
    "version": "1.4.43"
}

if __name__ == "__main__":
    print("测试合并节点格式化 (format_type=0):")
    print("-" * 50)
    formatted_0, labels_0 = format_output(test_json, 1)

    print(formatted_0)
    print(labels_0)