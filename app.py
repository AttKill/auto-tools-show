import importlib
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.json_visualizer import read_json_to_excel

# 添加utils路径到系统路径
streamlit_dir = Path(__file__).parent
sys.path.insert(0, str(streamlit_dir))
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

try:
    from mindmap_generator import MindMapGenerator
    from excel_handler import ExcelHandler
except ImportError:
    # 如果导入失败，创建必要的类
    class MindMapGenerator:
        def __init__(self):
            self.nodes = []
            self.edges = []
            self.node_counter = 0

        def _generate_id(self, prefix: str = "node") -> str:
            self.node_counter += 1
            return f"{prefix}_{self.node_counter}"

        def _add_node(self, label: str, level: int = 0, parent_id: str = None,
                      node_type: str = "default", tags: list = None) -> str:
            node_id = self._generate_id(node_type)
            node = {
                "id": node_id,
                "label": label,
                "level": level,
                "type": node_type,
                "tags": tags or []
            }
            self.nodes.append(node)

            if parent_id:
                edge = {
                    "from": parent_id,
                    "to": node_id,
                    "type": "hierarchy"
                }
                self.edges.append(edge)

            return node_id

        def generate_from_dataframe(self, df: pd.DataFrame, center_topic: str,
                                    split_columns: dict = None) -> dict:
            """
            从DataFrame生成思维导图数据

            Args:
                df: 包含given/when/then数据的DataFrame
                center_topic: 中心主题名称
                split_columns: 需要拆分的列配置，格式为 {列名: 分隔符}，默认为 {'given': '-'}
            """
            if split_columns is None:
                split_columns = {'given': '-', 'when': '-'}

            self.nodes = []
            self.edges = []
            self.node_counter = 0

            root_id = self._add_node(center_topic, 0, None, "center", ["中心主题"])
            # 存储各列的节点路径映射
            column_nodes = {}

            for _, row in df.iterrows():
                current_parent = root_id

                # 按照split_columns配置处理每一列
                for col_name, separator in split_columns.items():
                    if col_name in df.columns:
                        parts = str(row[col_name]).split(separator)

                        # 初始化当前列的节点映射
                        if col_name not in column_nodes:
                            column_nodes[col_name] = {}

                        # 逐层构建节点
                        for i, part in enumerate(parts):
                            part = part.strip()
                            if not part:
                                continue

                            # 构建路径标识
                            path_key = separator.join(parts[:i + 1])

                            # 检查节点是否已存在
                            if path_key not in column_nodes[col_name]:
                                # 为拆分列的节点添加标签前缀(原始node_label=part)
                                node_label = f"{col_name.capitalize()}: {part}"
                                node_id = self._add_node(
                                    node_label,
                                    level=self._calculate_level(current_parent, i, col_name, split_columns),
                                    parent_id=current_parent,
                                    node_type=col_name,
                                    tags=[col_name]
                                )
                                column_nodes[col_name][path_key] = node_id

                            current_parent = column_nodes[col_name][path_key]

                # 处理不在split_columns中的其他列（如then）
                for col_name in df.columns:
                    if col_name not in split_columns:  # 非拆分列
                        value = str(row[col_name]).strip()
                        if value:
                            # 为非拆分列添加节点
                            node_id = self._add_node(
                                f"{col_name.capitalize()}: {value}",
                                level=self._calculate_level(current_parent, 0, col_name, split_columns),
                                parent_id=current_parent,
                                node_type=col_name,
                                tags=[col_name]
                            )
                            current_parent = node_id

            return {
                "center_topic": center_topic,
                "nodes": self.nodes,
                "edges": self.edges,
                "statistics": {
                    "total_nodes": len(self.nodes),
                    "total_edges": len(self.edges),
                    "given_nodes": len([n for n in self.nodes if n["type"] == "given"]),
                    "when_nodes": len([n for n in self.nodes if n["type"] == "when"]),
                    "then_nodes": len([n for n in self.nodes if n["type"] == "then"])
                }
            }

        def _calculate_level(self, parent_node_id: str, position_in_split: int,
                             column_name: str, split_columns: dict) -> int:
            """计算节点层级"""
            # 找到父节点的层级
            parent_level = 0
            for node in self.nodes:
                if node["id"] == parent_node_id:
                    parent_level = node["level"]
                    break

            # 如果是拆分列中的第一个部分，层级加1；否则再加position
            if column_name in split_columns:
                return parent_level + 1 + position_in_split
            else:
                return parent_level + 1

    class ExcelHandler:
        def read_excel(self, file_buffer):
            import io
            try:
                # 读取所有sheet
                excel_file = pd.ExcelFile(file_buffer)
                sheets_data = {}

                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_buffer, sheet_name=sheet_name)

                    # 检查列名
                    df.columns = df.columns.str.lower().str.strip()

                    column_mapping = {}
                    required_columns = ['given', 'when', 'then']
                    for req_col in required_columns:
                        for actual_col in df.columns:
                            if req_col in actual_col.lower():
                                column_mapping[req_col] = actual_col
                                break

                    if len(column_mapping) == 3:
                        df = df.rename(columns={v: k for k, v in column_mapping.items()})
                        df = df[['given', 'when', 'then']]
                    elif len(df.columns) >= 3:
                        df = df.iloc[:, :3]
                        df.columns = ['given', 'when', 'then']
                    else:
                        st.warning(f"Sheet '{sheet_name}' 需要至少3列")
                        continue

                    df = df.fillna('')
                    sheets_data[sheet_name] = df

                return sheets_data

            except Exception as e:
                raise Exception(f"读取Excel文件失败: {str(e)}")

        def mindmap_to_excel(self, mindmap_data: dict) -> dict:
            return {mindmap_data.get('center_topic', '思维导图'): pd.DataFrame(columns=['given', 'when', 'then'])}

# 页面配置
st.set_page_config(
    page_title="Excel思维导图生成器",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """主应用函数"""
    # 初始化session state
    if 'mindmap_data' not in st.session_state:
        st.session_state.mindmap_data = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'excel_path' not in st.session_state:
        st.session_state.excel_path = None

    # 应用标题
    st.title("🧠 Excel思维导图生成器")
    st.markdown("从Excel文件生成思维导图，支持Given-When-Then结构")

    # 侧边栏
    with st.sidebar:
        st.header("📁 文件操作")

        uploaded_file = st.file_uploader(
            "上传Excel文件",
            type=['xlsx', 'xls'],
            help="Excel文件应包含三列：given, when, then"
        )

        st.divider()

        # 添加关于信息
        st.divider()
        st.caption("""
        **使用说明：**
        1. 上传包含given/when/then三列的Excel文件
        2. Sheet名称将作为中心主题
        3. 点击生成思维导图
        4. 支持导出为JSON/Excel/PNG
        """)

    # 主区域
    tab1, tab2, tab3 = st.tabs(["📤 导入数据", "🧭 数据分析", "📊 数据管理"])

    with tab1:
        render_data_import_tab(uploaded_file)

    with tab2:
        render_mindmap_tab()

    with tab3:
        render_data_management_tab()


def render_data_import_tab(uploaded_file):
    """渲染数据导入标签页"""
    # st.header("导入Excel数据")

    if uploaded_file is not None:

        try:
            # 读取Excel文件
            excel_handler = ExcelHandler()
            sheets_data = excel_handler.read_excel(uploaded_file)
            if not sheets_data:
                st.error("未找到有效的数据sheet")
                return

            # 显示可用的sheet页
            sheet_names = list(sheets_data.keys())
            selected_sheet = st.selectbox(
                "选择Sheet页（将作为中心主题）",
                sheet_names
            )

            if selected_sheet:
                df = sheets_data[selected_sheet]
                st.session_state.df = df
                st.session_state.excel_path = uploaded_file.name
                # 显示数据预览
                st.subheader("📋 数据预览")
                st.dataframe(df, use_container_width=True)

                # 显示统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总行数", len(df))
                with col2:
                    st.metric("Given分支数", df['given'].nunique())
                with col3:
                    st.metric("When分支数", df['when'].nunique())

                # 生成思维导图
                if st.button("🚀 生成思维导图", type="primary", use_container_width=True):
                    with st.spinner("正在生成思维导图..."):
                        generator = MindMapGenerator()
                        mindmap_data = generator.generate_from_dataframe(df, selected_sheet)
                        st.session_state.mindmap_data = mindmap_data
                        st.success(f"思维导图生成成功！共生成{mindmap_data['statistics']['total_nodes']}个节点")
                        st.rerun()

        except Exception as e:
            st.error(f"读取文件时出错: {str(e)}")
    else:
        render_example_section()


def render_example_section():
    """渲染示例数据部分"""
    st.info("请上传Excel文件开始使用")

    # 示例数据可折叠区域
    with st.expander("📋 示例文件", expanded=True):
        # 显示示例数据
        example_data = {
            'given': [
                '用户登录-成功登录',
                '用户登录-失败登录',
                '购物车-添加商品',
                '购物车-删除商品'
            ],
            'when': [
                '输入正确用户名和密码',
                '输入错误密码',
                '点击加入购物车按钮',
                '点击删除按钮'
            ],
            'then': [
                '跳转到首页',
                '显示错误提示',
                '商品数量增加',
                '商品从购物车移除'
            ]
        }

        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)

        st.markdown("""
        **Excel格式要求：**
        - 必须包含三列：`given`、`when`、`then`
        - `given`列支持用`-`分割多级结构
        - 列名不区分大小写
        - 支持多个sheet页
        """)

        # 提供示例文件下载
        @st.cache_data
        def create_example_excel():
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                example_df.to_excel(writer, sheet_name='测试用例', index=False)
            return output.getvalue()

        excel_bytes = create_example_excel()
        st.download_button(
            label="📥 下载示例文件",
            data=excel_bytes,
            file_name="mindmap_example.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # JSON转换可折叠功能
    with st.expander("🔄 JSON转换", expanded=False):
        st.markdown("在 [iodraw](https://www.iodraw.com/mind) 导出JSON文件并上传进行转换")
        uploaded_json = st.file_uploader("上传JSON文件", type=['json'], key="json_converter")

        if uploaded_json is not None:
            try:
                # 读取JSON数据
                json_data = json.load(uploaded_json)
                # 调用example_usage函数处理数据
                json2excel_dict = read_json_to_excel(json_data)
                sheets_data = json2excel_dict.get("sheets_data")
                show_parse_json_dict = json2excel_dict.get("show_parse_json_dict")
                table_heads = show_parse_json_dict.get("table_heads")
                table_heads_data =show_parse_json_dict.get("show_data")

                st.success(f"JSON文件转换完成！标签清单[{table_heads}];[{table_heads_data}]")

                example_df1 = pd.DataFrame(table_heads_data)
                st.dataframe(example_df1, use_container_width=True)

                # 生成Excel文件供下载
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df in sheets_data.items():
                        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

                st.download_button(
                    label="📥 下载转换后的Excel文件",
                    data=output.getvalue(),
                    file_name="converted_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


            except Exception as e:
                st.error(f"处理JSON文件时出错: {str(e)}")

def render_mindmap_tab():
    """渲染思维导图标签页"""
    # st.header("思维导图可视化")

    if st.session_state.mindmap_data:
        # 显示统计信息
        stats = st.session_state.mindmap_data['statistics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总节点数", stats['total_nodes'])
        with col2:
            st.metric("Given节点", stats['given_nodes'])
        with col3:
            st.metric("When节点", stats['when_nodes'])
        with col4:
            st.metric("Then节点", stats['then_nodes'])

        # 选择可视化方式
        viz_type = st.radio(
            "选择可视化类型",
            ["树状图", "思维导图"],
            horizontal=True
        )

        if viz_type == "树状图":
            render_treemap()
        else:  # 思维导图
            spec = importlib.util.spec_from_file_location(
                "skill_module",
                streamlit_dir / "components/mindmap_visualizer.py"
            )
            skill_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(skill_module)

            # 动态获取函数并调用
            skill_function = getattr(skill_module, "visualize_mindmap")
            skill_function()

        # 导出选项
        st.divider()
        st.subheader("📤 导出选项")
        render_export_options()

    else:
        st.info("请先导入数据并生成思维导图")

def render_treemap():
    """渲染树状图

    TODO 文本内容也不允许重复
    """
    try:
        mindmap_data = st.session_state.mindmap_data
        nodes = mindmap_data['nodes']
        edges = mindmap_data['edges']

        # 创建节点ID到标签的映射
        node_id_to_label = {node['id']: node['label'] for node in nodes}

        # 构建树状图数据
        labels = []
        parents = []
        colors = []
        types = []

        # 遍历所有节点
        for node in nodes:
            labels.append(node['label'])
            types.append(node['type'])

            # 设置颜色
            if node['type'] == 'center':
                colors.append('#FF6B6B')
            elif node['type'] == 'given':
                colors.append('#4ECDC4')
            elif node['type'] == 'when':
                colors.append('#45B7D1')
            elif node['type'] == 'then':
                colors.append('#96CEB4')
            else:
                colors.append('#FFEAA7')

            # 根据edges关系确定父节点，不考虑节点类型
            parent_label = ""
            for edge in edges:
                if edge['to'] == node['id']:
                    parent_id = edge['from']
                    if parent_id in node_id_to_label:
                        parent_label = node_id_to_label[parent_id]
                    break

            parents.append(parent_label)

        # 验证数据一致性
        assert len(labels) == len(parents) == len(colors) == len(types), \
            f"数据长度不一致: labels({len(labels)}), parents({len(parents)}), colors({len(colors)}), customdata({len(types)})"

        print(f"Labels: {labels}")  # 调试信息
        print(f"Parents: {parents}")  # 调试信息

        # 检查是否存在空标签或无效的父子关系
        for i, (label, parent) in enumerate(zip(labels, parents)):
            if not label:
                print(f"警告: 第{i}个节点标签为空")
            if parent not in labels and parent != "":
                print(f"警告: 第{i}个节点的父节点'{parent}'不存在于标签中")

        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            marker=dict(
                colors=colors,
                line=dict(width=2, color="white")  # 添加边框线，提高可见性
            ),
            customdata=types,
            hovertemplate="<b>%{label}</b><br>类型: %{customdata}<extra></extra>",
            textinfo="label",  # 只显示标签
        ))

        fig.update_layout(
            title="思维导图 - 树状图视图",
            height=600,
            margin=dict(t=50, b=50, l=50, r=50)  # 设置边距
        )

        # 检查是否有有效数据
        if len([l for l in labels if l]) == 0:
            st.warning("没有有效的标签数据用于显示")
            return

        st.plotly_chart(fig, use_container_width=True)

    except AssertionError as e:
        st.error(f"数据验证失败: {str(e)}")
    except Exception as e:
        st.error(f"生成树状图时出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_export_options():
    """渲染导出选项"""
    col1, col2 = st.columns(2)

    with (col1):
        if st.button("📥 导出为JSON", use_container_width=True):
            json_str = json.dumps(st.session_state.mindmap_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="下载JSON文件",
                data=json_str,
                file_name="mindmap.json",
                mime="application/json",
                use_container_width=True
            )

    with col2:
        if st.button("📊 导出为Excel", use_container_width=True):
            excel_handler = ExcelHandler()
            excel_data = excel_handler.mindmap_to_excel(st.session_state.mindmap_data)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in excel_data.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # sheet名最多31字符

            st.download_button(
                label="下载Excel文件",
                data=output.getvalue(),
                file_name="exported_mindmap.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


def render_data_management_tab():
    """渲染数据管理标签页"""
    st.header("数据管理")

    if st.session_state.df is not None:
        # 编辑数据
        st.subheader("📝 编辑数据")

        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "given": st.column_config.TextColumn(
                    "Given (使用'-'分割多级)",
                    help="输入Given条件，用'-'分割层级"
                ),
                "when": st.column_config.TextColumn(
                    "When (条件)",
                    help="输入When条件"
                ),
                "then": st.column_config.TextColumn(
                    "Then (结果)",
                    help="输入Then结果"
                )
            }
        )

        if st.button("💾 保存修改并重新生成", type="primary", use_container_width=True):
            st.session_state.df = edited_df
            st.success("数据已保存！")

            # 重新生成思维导图
            with st.spinner("正在重新生成思维导图..."):
                generator = MindMapGenerator()
                mindmap_data = generator.generate_from_dataframe(
                    edited_df,
                    st.session_state.mindmap_data["center_topic"]
                )
                st.session_state.mindmap_data = mindmap_data
                st.rerun()

        # 显示数据示例
        st.divider()
        st.subheader("🔍 数据示例")

        for i, (_, row) in enumerate(edited_df.head(3).iterrows()):
            with st.expander(f"示例 {i + 1}: {row['given']}"):
                st.write(f"**Given**: {row['given']}")
                st.write(f"**When**: {row['when']}")
                st.write(f"**Then**: {row['then']}")
                st.write(f"层级深度: {len(str(row['given']).split('-'))}")

    else:
        st.info("暂无数据，请先导入Excel文件")


# 应用入口
if __name__ == "__main__":
    main()
