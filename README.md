# diff_eq_solve

全中文科学计算智能体，用于数学物理方程的符号推导、数值求解、可视化和误差分析。项目默认离线运行，不依赖大模型 API；文献检索和 arXiv 查询是可选能力。

## 主要功能

- 自然语言或方程文本输入，自动识别求解路线。
- 支持 ODE 解析优先、数值兜底：例如 `ode: y'' + y = 0`。
- 支持自由 PDE 文本解析：例如 `pde: u_t = alpha*u_xx`、`pde: u_tt = c**2*u_xx`、`pde: u_xx + u_yy = 0`。
- 支持数学物理核心模板：经典 ODE/PDE、电磁、量子、QFT、广义相对论、流体、特殊函数、Lagrangian/Noether。
- 支持 Streamlit 本地 Web 界面、Notebook 面板和 Python API。
- 输出推导、解析解、数值解、图像、残差、误差报告、警告和复现实验代码。

## 安装

```powershell
git clone https://github.com/wangyufanshuai/diff_eq_solve.git
cd diff_eq_solve
pip install -r requirements.txt
```

## 启动 Web 界面

```powershell
streamlit run diff_eq_solver/agent_web.py
```

打开页面后可以在“问题”框输入方程，也可以在“自由 PDE 输入”区域填写 PDE、变量、初值、边界条件和区域。

## 快速测试问题

热方程：

```text
pde: u_t = alpha*u_xx
```

参数 JSON：

```json
{
  "alpha": 1.0,
  "initial_condition": "sin(pi*x)",
  "boundary_conditions": {"left": 0, "right": 0},
  "domain": {"x": [0, 1], "t": [0, 0.05]},
  "nx": 64,
  "nt": 50
}
```

波方程：

```text
pde: u_tt = c**2*u_xx
```

参数 JSON：

```json
{
  "c": 1.0,
  "initial_condition": "sin(pi*x)",
  "initial_velocity": "0",
  "boundary_conditions": {"left": 0, "right": 0},
  "domain": {"x": [0, 1], "t": [0, 0.1]},
  "nx": 64,
  "nt": 60
}
```

二维 Laplace 方程：

```text
pde: u_xx + u_yy = 0
```

参数 JSON：

```json
{
  "variables": ["x", "y"],
  "nx": 40,
  "ny": 40,
  "boundary_value": 0
}
```

简谐振子 ODE：

```text
ode: y'' + y = 0
```

参数 JSON：

```json
{
  "initial_conditions": {"y0": 1, "dy0": 0},
  "t_span": [0, 10],
  "n_points": 500
}
```

## Python API

```python
from diff_eq_solver import ScientificAgent

agent = ScientificAgent()
result = agent.run(
    "pde: u_t = alpha*u_xx",
    params={
        "alpha": 1.0,
        "initial_condition": "sin(pi*x)",
        "boundary_conditions": {"left": 0, "right": 0},
        "domain": {"x": [0, 1], "t": [0, 0.05]},
    },
    include_literature=False,
)

print(result.rendered_summary)
print(result.solver_report)
```

Notebook 面板：

```python
from diff_eq_solver import create_scientific_agent_panel

create_scientific_agent_panel()
```

## PDE 求解策略

系统会先解析 PDE 文本并分类，再尝试：

1. 符号求解：调用 SymPy `pdsolve` 和分离变量路径。
2. 数值兜底：对一维空间演化 PDE 使用有限差分 + `solve_ivp`；对简单二维椭圆 PDE 使用有限差分线性系统。
3. 误差与警告：报告 solver 状态、网格、残差或稳定性信息；缺初值/边界条件时会明确提示。

注意：项目目标是覆盖教材核心数学物理方程族，并对可解析/可数值的问题给出可靠结果；不会声称任意复杂 PDE 都存在闭式解析解。

## 运行测试

```powershell
python -m pytest -q
```

当前测试覆盖 ScientificAgent、Notebook、Web 描述、方程目录、通用 ODE、通用 PDE 解析与数值兜底。

## 目录结构

```text
diff_eq_solver/
  agent_web.py          # Streamlit Web 界面
  agent_notebook.py     # Notebook 面板
  scientific_agent.py   # 智能体路由与编排
  pde_solver.py         # 通用 PDE 解析、分类、求解
  equation_catalog.py   # 数学物理方程目录
  equations/            # 注册方程模板库
tests/                  # 单元测试
notebooks/              # 示例 Notebook
```

## 许可证

见仓库中的 `LICENSE` 文件。
