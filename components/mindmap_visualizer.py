from collections import defaultdict
from typing import List

import plotly.graph_objects as go
import streamlit as st


class MindMap:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.node_counter = 0
        self.collapsed_nodes = set()  # 存储被收起的节点ID

    def parse_data(self, data_list: List[str]):
        """解析数据集"""
        for item in data_list:
            parts = item.split("##")
            parent = None

            for i, part in enumerate(parts):
                node_id = f"{i}_{part}"

                if node_id not in self.nodes:
                    self.nodes[node_id] = {
                        'id': node_id,
                        'label': part,
                        'level': i,
                        'children': [],
                        'collapsed': False  # 添加收起状态
                    }

                if parent is not None:
                    self.edges.append((parent, node_id))
                    if node_id not in self.nodes[parent]['children']:
                        self.nodes[parent]['children'].append(node_id)

                parent = node_id

    def add_node(self, parent_id: str, label: str):
        """添加新节点"""
        if parent_id not in self.nodes:
            return None

        parent_level = self.nodes[parent_id]['level']
        new_level = parent_level + 1
        new_id = f"new_{self.node_counter}"

        self.nodes[new_id] = {
            'id': new_id,
            'label': label,
            'level': new_level,
            'children': [],
            'collapsed': False
        }

        self.edges.append((parent_id, new_id))
        self.nodes[parent_id]['children'].append(new_id)
        self.node_counter += 1

        return new_id

    def update_node(self, node_id: str, new_label: str):
        """更新节点标签"""
        if node_id in self.nodes:
            self.nodes[node_id]['label'] = new_label
            return True
        return False

    def toggle_node_collapse(self, node_id: str):
        """切换节点的收起/展开状态"""
        if node_id in self.nodes:
            self.nodes[node_id]['collapsed'] = not self.nodes[node_id]['collapsed']
            return True
        return False

    def is_node_visible(self, node_id: str):
        """检查节点是否可见（没有被任何父节点收起）"""
        current_id = node_id
        while current_id in self.nodes:
            node = self.nodes[current_id]
            # 找到父节点
            parent_id = None
            for edge in self.edges:
                if edge[1] == current_id:
                    parent_id = edge[0]
                    break

            if parent_id and self.nodes[parent_id]['collapsed']:
                return False

            if parent_id is None:
                break
            current_id = parent_id

        return True


def truncate_text(text, max_length=10):
    """截断文本，如果超过最大长度则添加..."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def create_plotly_mindmap(mindmap: MindMap):
    """使用Plotly创建交互式思维导图（考虑收起状态）"""

    if not mindmap.nodes:
        return go.Figure()

    # 计算节点位置
    node_positions = {}
    level_nodes = defaultdict(list)

    # 按层级分组节点（只考虑可见节点）
    for node_id, node in mindmap.nodes.items():
        if mindmap.is_node_visible(node_id):
            level_nodes[node['level']].append(node_id)

    # 计算位置
    max_level = max(level_nodes.keys()) if level_nodes else 0
    total_height = len(max(level_nodes.values(), key=len)) if level_nodes else 1

    for level, nodes in level_nodes.items():
        x_pos = level * 2  # 水平间距
        y_spacing = 2.0  # 垂直间距

        for i, node_id in enumerate(nodes):
            y_offset = (len(nodes) - 1) * y_spacing / 2
            y_pos = -i * y_spacing + y_offset
            node_positions[node_id] = (x_pos, y_pos)

    # 创建节点和边的轨迹
    node_x = []
    node_y = []
    node_text = []  # 显示的文本（截断后）
    node_hovertext = []  # 悬停时显示的完整文本
    node_ids = []
    node_font_sizes = []
    node_colors = []

    for node_id, (x, y) in node_positions.items():
        node_x.append(x)
        node_y.append(y)

        # 获取节点的完整标签
        full_label = mindmap.nodes[node_id]['label']

        # 截断文本用于显示
        display_text = truncate_text(full_label)
        node_text.append(display_text)

        # 完整文本用于悬停显示
        node_hovertext.append(full_label)

        node_ids.append(node_id)

        # 根据层级设置字体大小
        level = mindmap.nodes[node_id]['level']
        if level == 0:
            node_font_sizes.append(18)  # 根节点字体最大
            node_colors.append('#FF6B6B')  # 根节点颜色
        elif level == 1:
            node_font_sizes.append(16)
            node_colors.append('#4ECDC4')  # 一级节点颜色
        elif level == 2:
            node_font_sizes.append(14)
            node_colors.append('#45B7D1')  # 二级节点颜色
        else:
            node_font_sizes.append(12)
            node_colors.append('#96CEB4')  # 三级及以下节点颜色

    # 创建节点轨迹 - 使用文本模式，去掉矩形框
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='text+markers',
        text=node_text,
        hovertext=node_hovertext,
        hoverinfo='text',
        textposition="middle center",
        textfont=dict(
            size=node_font_sizes,
            color=node_colors,
            family="Arial, sans-serif"
        ),
        marker=dict(
            size=0,  # 将标记点大小设为0，只显示文本
            opacity=0
        ),
        customdata=node_ids
    )

    # 创建边轨迹（只绘制可见的边）
    edge_x = []
    edge_y = []

    for edge in mindmap.edges:
        if (edge[0] in node_positions and edge[1] in node_positions and
                mindmap.is_node_visible(edge[0]) and mindmap.is_node_visible(edge[1])):
            x0, y0 = node_positions[edge[0]]
            x1, y1 = node_positions[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='#888'),
        hoverinfo='none'
    )

    # 创建图形
    fig = go.Figure(data=[edge_trace, node_trace])

    # 更新布局
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600,
        dragmode='pan'  # 允许拖动画布
    )

    return fig


def visualize_mindmap():
    st.markdown("---")

    # 初始化session state
    if 'mindmap' not in st.session_state:
        st.session_state.mindmap = MindMap()

    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None

    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False

    # 侧边栏
    with st.sidebar:
        st.header("📁 数据管理")

        # 检查是否存在mindmap_data
        if 'mindmap_data' not in st.session_state:
            st.session_state.mindmap_data = {
                'nodes': [],
                'edges': []
            }

        nodes = st.session_state.mindmap_data

        example_data = convert_to_target_format(nodes) if nodes['nodes'] else []

        # 数据输入
        data_input = st.text_area(
            "待分析数据集（每行一个路径，用##分隔）",
            value="\n".join(example_data) if example_data else "中心节点##一级节点##二级节点##三级节点",
            height=200,
            disabled=True  # 设置为禁用状态，即只读
        )

        if st.button("🔧 解析数据", use_container_width=True):
            data_list = [line.strip() for line in data_input.split('\n') if line.strip()]
            st.session_state.mindmap = MindMap()
            st.session_state.mindmap.parse_data(data_list)
            st.session_state.selected_node = None
            st.rerun()

        st.markdown("---")

        # 节点操作
        st.header("🎯 节点操作")

        if st.session_state.mindmap.nodes:
            # 选择节点
            node_options = {f"{node['label']} (层级 {node['level']})": nid
                            for nid, node in st.session_state.mindmap.nodes.items()}

            selected_option = st.selectbox(
                "选择要操作的节点",
                options=list(node_options.keys()),
                index=0 if node_options else None
            )

            if selected_option:
                st.session_state.selected_node = node_options[selected_option]

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("✏️ 编辑", use_container_width=True):
                        st.session_state.edit_mode = True

                with col2:
                    # 展开/收起按钮
                    selected_node_data = st.session_state.mindmap.nodes[st.session_state.selected_node]
                    button_text = "📂 收起" if not selected_node_data['collapsed'] else "📁 展开"
                    if st.button(button_text, use_container_width=True):
                        st.session_state.mindmap.toggle_node_collapse(st.session_state.selected_node)
                        st.rerun()

                with col3:
                    if st.button("🗑️ 重置", use_container_width=True):
                        st.session_state.mindmap = MindMap()
                        st.session_state.selected_node = None
                        st.rerun()

            # 添加新节点
            st.subheader("添加新节点")
            new_node_label = st.text_input("新节点名称", value="新节点")

            if st.button("➕ 添加子节点", use_container_width=True) and st.session_state.selected_node:
                new_id = st.session_state.mindmap.add_node(
                    st.session_state.selected_node,
                    new_node_label
                )
                if new_id:
                    st.success(f"已添加节点: {new_node_label}")
                    st.rerun()

        st.markdown("---")
        st.markdown("### 📊 统计信息")
        if st.session_state.mindmap.nodes:
            visible_nodes = sum(1 for node_id in st.session_state.mindmap.nodes
                                if st.session_state.mindmap.is_node_visible(node_id))
            collapsed_nodes = sum(1 for node in st.session_state.mindmap.nodes.values()
                                  if node['collapsed'])

            st.info(f"总节点数: {len(st.session_state.mindmap.nodes)}")
            st.info(f"可见节点数: {visible_nodes}")
            st.info(f"收起节点数: {collapsed_nodes}")
            st.info(f"总连接数: {len(st.session_state.mindmap.edges)}")

    # 主区域
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🧭 思维导图")

        if st.session_state.mindmap.nodes:
            # 创建思维导图
            fig = create_plotly_mindmap(st.session_state.mindmap)

            # 显示图形
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'displaylogo': False
            })

            # 显示当前选中的节点
            if st.session_state.selected_node:
                selected_node_data = st.session_state.mindmap.nodes[st.session_state.selected_node]
                status_text = "（已收起）" if selected_node_data['collapsed'] else "（已展开）"
                st.info(f"当前选中节点: **{selected_node_data['label']}** {status_text}")

                # 编辑模式
                if st.session_state.edit_mode:
                    with st.form("edit_form"):
                        new_label = st.text_input(
                            "新名称",
                            value=selected_node_data['label']
                        )
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.form_submit_button("💾 保存"):
                                if st.session_state.mindmap.update_node(
                                        st.session_state.selected_node,
                                        new_label
                                ):
                                    st.success("节点已更新！")
                                    st.session_state.edit_mode = False
                                    st.rerun()
                        with col2:
                            if st.form_submit_button("❌ 取消"):
                                st.session_state.edit_mode = False
                                st.rerun()
        else:
            st.info("👈 请先在侧边栏输入或加载数据")

    with col2:
        st.header("📋 数据结构")

        if st.session_state.mindmap.nodes:
            # 显示原始数据格式
            def generate_paths(node_id, current_path=""):
                node = st.session_state.mindmap.nodes[node_id]
                new_path = f"{current_path}##{node['label']}" if current_path else node['label']

                # 如果节点被收起，只显示到当前节点的路径
                if node['collapsed'] or not node['children']:
                    return [new_path]

                paths = []
                for child_id in node['children']:
                    if st.session_state.mindmap.is_node_visible(child_id):
                        paths.extend(generate_paths(child_id, new_path))

                return paths

            root_nodes = [nid for nid, node in st.session_state.mindmap.nodes.items()
                          if node['level'] == 0]

            if root_nodes:
                all_paths = []
                for root in root_nodes:
                    if st.session_state.mindmap.is_node_visible(root):
                        all_paths.extend(generate_paths(root))

                export_text = ""
                added_lines = set()  # 用于存储已添加的行，避免重复
                for path in all_paths:
                    path_split_values = path.split("##")
                    for i in range(len(path_split_values)):
                        line_content = f"{'#' * (i + 1)}{path_split_values[i]}\n\n"
                        if line_content not in added_lines:  # 检查是否已存在
                            added_lines.add(line_content)
                            export_text += line_content
                st.text_area("下载后的文件可直接在<https://www.iodraw.com/mind>打开", export_text, height=300)

                # 提供下载
                st.download_button(
                    label="📥 下载数据",
                    data=export_text,
                    file_name="mindmap_data.md",
                    mime="text/plain",
                    use_container_width=True
                )


def convert_to_target_format(data):
    """
    将树形结构的数据转换为目标字符串格式

    参数:
        data: dict, 包含nodes和edges的数据

    返回:
        list: 转换后的字符串列表，每个字符串对应一条完整路径
    """
    # 创建节点映射字典
    nodes_map = {node['id']: node for node in data['nodes']}

    # 构建子节点列表
    children_map = {}
    for edge in data['edges']:
        if edge['type'] == 'hierarchy':
            parent_id = edge['from']
            child_id = edge['to']
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)

    # 找到中心节点
    center_id = None
    for node in data['nodes']:
        if node.get('type') == 'center' or node.get('level') == 0:
            center_id = node['id']
            break

    if not center_id:
        return []

    # DFS递归构建路径
    def dfs_build_paths(node_id, current_path, all_paths):
        node = nodes_map[node_id]
        new_path = current_path + [node['label']]

        # 检查是否是叶子节点或没有子节点
        if node_id not in children_map or not children_map[node_id]:
            all_paths.append(new_path)
            return

        # 递归处理所有子节点
        for child_id in children_map[node_id]:
            dfs_build_paths(child_id, new_path.copy(), all_paths)

    # 从中心节点开始
    all_paths = []
    if center_id in children_map:
        for child_id in children_map[center_id]:
            dfs_build_paths(child_id, [nodes_map[center_id]['label']], all_paths)
    else:
        # 如果中心节点没有子节点，直接返回中心节点
        all_paths.append([nodes_map[center_id]['label']])

    # 筛选包含when或then的路径（完整测试用例）
    result_paths = []
    for path in all_paths:
        # 检查路径中是否有when或then
        has_when_then = False
        for label in path:
            if isinstance(label, str) and ('When:' in label or 'Then:' in label):
                has_when_then = True
                break
        # 只有包含when/then或者路径长度>1的才保留
        if has_when_then or len(path) > 1:
            result_paths.append(path)

    # 转换为目标格式
    result = ['##'.join(path) for path in result_paths]

    return result