"""
Chart generation tool for LlamaIndex FunctionAgent
Provides standalone chart generation functionality based on RAG content analysis
"""

import logging
import os
import tempfile
import subprocess
import time
import re
from typing import Annotated, List, Optional, Dict, Any
from llama_index.core.tools import FunctionTool

logger = logging.getLogger("uvicorn.error")


class ChartGeneratorTool:
    """Standalone chart generation tool for data visualization"""
    
    def __init__(self):
        self.logger = logger
        
        # Check dependencies
        try:
            import matplotlib
            import numpy
            self.available = True
            self.logger.info("Chart generation tool initialized with matplotlib")
        except ImportError as e:
            self.available = False
            self.logger.warning(f"Chart generation dependencies not available: {e}")
    
    def generate_chart(self, query: str, rag_content: str, sources: List[str]) -> str:
        """Generate a chart using Python code execution based on RAG data
        
        Args:
            query: The user's original query
            rag_content: The content retrieved from RAG system
            sources: List of source documents
            
        Returns:
            String indicating success/failure and file path if successful
        """
        if not self.available:
            return "Chart generation is not available - missing dependencies"
        
        try:
            self.logger.info(f"Generating chart for query: '{query[:50]}...'")
            
            # Analyze content and generate Python code
            content_analysis = self._analyze_content_for_visualization(query, rag_content, sources)
            python_code = self._generate_chart_code(query, rag_content, content_analysis)
            
            self.logger.debug(f"Generated Python code: {python_code[:200]}...")
            
            # Execute Python code using subprocess
            chart_path = self._execute_chart_code(python_code, query)
            
            if chart_path:
                self.logger.info(f"Chart generated successfully: {chart_path}")
                return chart_path
            else:
                return "Failed to generate chart"
                
        except Exception as e:
            self.logger.error(f"Failed to generate chart: {str(e)}")
            return f"Failed to generate chart: {str(e)}"
    
    def generate_parametric_chart(self, x_label: str, y_label: str, data_points: str, chart_type: str) -> str:
        """Generate a chart with specific parameters
        
        Args:
            x_label: Label for the x-axis
            y_label: Label for the y-axis
            data_points: Comma-separated list of (x,y) coordinate pairs
            chart_type: Type of chart - either 'bar' or 'line'
            
        Returns:
            File path of generated chart if successful, error message if failed
        """
        if not self.available:
            return "Chart generation is not available - missing dependencies"
        
        try:
            self.logger.info(f"Generating {chart_type} chart: '{x_label}' vs '{y_label}'")
            
            # Parse data points
            parsed_data = self._parse_data_points(data_points)
            if not parsed_data:
                return "Invalid data points format. Use 'x1,y1;x2,y2;x3,y3'"
            
            # Generate Python code for the specific chart
            python_code = self._generate_parametric_chart_code(x_label, y_label, parsed_data, chart_type)
            
            self.logger.debug(f"Generated Python code: {python_code[:200]}...")
            
            # Execute Python code using subprocess
            chart_path = self._execute_chart_code(python_code, f"{chart_type}_chart")
            
            if chart_path:
                self.logger.info(f"Chart generated successfully: {chart_path}")
                return chart_path
            else:
                return "Failed to generate chart"
                
        except Exception as e:
            self.logger.error(f"Failed to generate parametric chart: {str(e)}")
            return f"Failed to generate chart: {str(e)}"
    
    def _parse_data_points(self, data_points: str) -> List[tuple]:
        """Parse data points string into list of (x, y) tuples"""
        try:
            points = []
            for point_pair in data_points.split(';'):
                if ',' in point_pair:
                    x_str, y_str = point_pair.strip().split(',', 1)
                    
                    # Try to convert y to float, keep x as string for labels
                    y_val = float(y_str.strip())
                    x_val = x_str.strip()
                    
                    points.append((x_val, y_val))
            
            return points
        except Exception as e:
            self.logger.error(f"Error parsing data points '{data_points}': {str(e)}")
            return []
    
    def _generate_parametric_chart_code(self, x_label: str, y_label: str, data_points: List[tuple], chart_type: str) -> str:
        """Generate Python code for parametric chart"""
        
        # Extract x labels and y values
        x_labels = [str(point[0]) for point in data_points]
        y_values = [point[1] for point in data_points]
        
        if chart_type.lower() == 'bar':
            return self._generate_bar_chart_code(x_label, y_label, x_labels, y_values)
        elif chart_type.lower() == 'line':
            return self._generate_line_chart_code(x_label, y_label, x_labels, y_values)
        else:
            # Default to bar chart
            return self._generate_bar_chart_code(x_label, y_label, x_labels, y_values)
    
    def _generate_bar_chart_code(self, x_label: str, y_label: str, x_labels: List[str], y_values: List[float]) -> str:
        """Generate bar chart Python code"""
        return f'''
import matplotlib.pyplot as plt
import numpy as np

# Data
x_labels = {x_labels}
y_values = {y_values}

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x_labels, y_values, color='#2E86AB', alpha=0.8)

# Styling
ax.set_title('Bar Chart', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('{x_label}', fontsize=12)
ax.set_ylabel('{y_label}', fontsize=12)

# Add value labels on bars
for bar, value in zip(bars, y_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(y_values)*0.01,
            f'{{value:.1f}}', ha='center', va='bottom', fontweight='bold')

# Professional styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save chart
plt.savefig('/tmp/parametric_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved to /tmp/parametric_chart.png")
'''
    
    def _generate_line_chart_code(self, x_label: str, y_label: str, x_labels: List[str], y_values: List[float]) -> str:
        """Generate line chart Python code"""
        return f'''
import matplotlib.pyplot as plt
import numpy as np

# Data
x_labels = {x_labels}
y_values = {y_values}
x_positions = range(len(x_labels))

# Create line chart
fig, ax = plt.subplots(figsize=(10, 6))
line = ax.plot(x_positions, y_values, marker='o', linewidth=2, markersize=8, color='#2E86AB')

# Styling
ax.set_title('Line Chart', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('{x_label}', fontsize=12)
ax.set_ylabel('{y_label}', fontsize=12)

# Set x-axis labels
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=45, ha='right')

# Add value labels on points
for i, (x_pos, value) in enumerate(zip(x_positions, y_values)):
    ax.text(x_pos, value + max(y_values)*0.02, f'{{value:.1f}}', 
            ha='center', va='bottom', fontweight='bold')

# Professional styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)
plt.tight_layout()

# Save chart
plt.savefig('/tmp/parametric_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved to /tmp/parametric_chart.png")
'''
    
    def _analyze_content_for_visualization(self, query: str, answer: str, sources: List[str]) -> Dict[str, Any]:
        """Extract visualizable data from the RAG response"""
        analysis = {
            "numbers": [],
            "metrics": [],
            "relationships": [],
            "processes": [],
            "entities": [],
            "comparisons": [],
            "data_points": [],
            "source_context": sources
        }
        
        # Extract numerical data and metrics from the answer
        # Find numbers and percentages
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', answer)
        analysis["numbers"] = numbers[:10]  # Limit to first 10 numbers
        
        # Find common metrics keywords
        metrics_keywords = ["reward", "stake", "score", "weight", "allocation", "distribution", 
                           "performance", "accuracy", "loss", "fee", "rate", "ratio"]
        for keyword in metrics_keywords:
            if keyword in answer.lower():
                analysis["metrics"].append(keyword)
        
        # Find relationship indicators
        relationship_indicators = ["compared to", "versus", "higher than", "lower than", 
                                 "correlates with", "depends on", "affects", "influences"]
        for indicator in relationship_indicators:
            if indicator in answer.lower():
                analysis["relationships"].append(indicator)
        
        # Find process indicators
        process_indicators = ["first", "then", "next", "finally", "step", "phase", 
                            "process", "workflow", "algorithm", "mechanism"]
        for indicator in process_indicators:
            if indicator in answer.lower():
                analysis["processes"].append(indicator)
        
        # Extract entities mentioned
        entities = ["worker", "node", "validator", "reputer", "client", "model", 
                   "network", "chain", "token", "stake", "reward"]
        for entity in entities:
            if entity in answer.lower():
                analysis["entities"].append(entity)
        
        self.logger.debug(f"Content analysis: {analysis}")
        return analysis
    
    def _determine_visualization_type(self, query: str, answer: str, analysis: Dict[str, Any]) -> str:
        """Determine the best visualization type based on content analysis"""
        
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Check for reward/distribution related content
        if (any(word in query_lower for word in ["reward", "distribution", "allocation", "payment"]) or
            any(word in answer_lower for word in ["reward", "distribution", "allocation"]) or
            "reward" in analysis["metrics"]):
            return "reward_distribution"
        
        # Check for network metrics
        if (any(word in query_lower for word in ["performance", "metrics", "score", "accuracy"]) or
            len(analysis["numbers"]) > 3):
            return "network_metrics"
        
        # Check for workflow/process
        if (any(word in query_lower for word in ["process", "workflow", "algorithm", "mechanism"]) or
            len(analysis["processes"]) > 2):
            return "workflow"
        
        # Check for comparisons
        if (any(word in query_lower for word in ["compare", "versus", "difference"]) or
            len(analysis["relationships"]) > 1):
            return "comparison"
        
        # Check for architecture
        if any(word in query_lower for word in ["architecture", "system", "network", "structure"]):
            return "architecture"
        
        return "general"
    
    def _generate_chart_code(self, query: str, answer: str, analysis: Dict[str, Any]) -> str:
        """Generate Python code for chart creation based on content analysis"""
        
        chart_type = self._determine_visualization_type(query, answer, analysis)
        
        if chart_type == "reward_distribution":
            return self._generate_reward_chart_code(analysis)
        elif chart_type == "network_metrics":
            return self._generate_metrics_chart_code(analysis)
        elif chart_type == "workflow":
            return self._generate_workflow_chart_code(analysis)
        elif chart_type == "comparison":
            return self._generate_comparison_chart_code(analysis)
        else:
            return self._generate_generic_chart_code(query, answer, analysis)
    
    def _generate_reward_chart_code(self, analysis: Dict[str, Any]) -> str:
        """Generate bar chart code for reward distribution"""
        numbers = analysis.get("numbers", [])
        entities = analysis.get("entities", [])
        
        # Extract percentages or create sample data
        if numbers:
            # Try to extract meaningful values
            values = []
            for num in numbers[:5]:
                try:
                    # Remove % sign if present and convert
                    clean_num = num.replace('%', '')
                    values.append(float(clean_num))
                except ValueError:
                    continue
        else:
            # Sample data based on typical reward distribution
            values = [60, 30, 10]
        
        # Use extracted entities or defaults
        if entities:
            labels = entities[:len(values)]
        else:
            labels = ["Workers", "Validators", "Reputers"][:len(values)]
        
        # Ensure we have matching labels and values
        if len(labels) < len(values):
            labels.extend([f"Category {i+1}" for i in range(len(labels), len(values))])
        labels = labels[:len(values)]
        
        return f'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Data from RAG analysis
values = {values}
labels = {labels}

# Create professional bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, values, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)

# Styling
ax.set_title('Reward Distribution', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Allocation (%)', fontsize=12)
ax.set_xlabel('Entity Type', fontsize=12)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
            f'{{value:.1f}}%' if value < 100 else f'{{value:.0f}}',
            ha='center', va='bottom', fontweight='bold')

# Professional styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Save chart
plt.savefig('/tmp/reward_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved to /tmp/reward_chart.png")
'''
    
    def _generate_metrics_chart_code(self, analysis: Dict[str, Any]) -> str:
        """Generate metrics dashboard code"""
        numbers = analysis.get("numbers", [])
        metrics = analysis.get("metrics", [])
        
        return f'''
import matplotlib.pyplot as plt
import numpy as np

# Sample metrics data based on analysis
metrics_data = {{
    'Performance': 85,
    'Accuracy': 92,
    'Stake': 75,
    'Rewards': 68
}}

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Performance gauge
ax1.pie([85, 15], labels=['Performance', ''], colors=['#2E86AB', '#E8E8E8'], 
        startangle=90, counterclock=False)
ax1.set_title('Performance Score', fontweight='bold')

# Accuracy bar
ax2.bar(['Accuracy'], [92], color='#A23B72', alpha=0.8)
ax2.set_ylim(0, 100)
ax2.set_title('Accuracy %', fontweight='bold')

# Stake distribution
stake_data = [75, 25]
ax3.bar(['Active', 'Available'], stake_data, color=['#F18F01', '#E8E8E8'])
ax3.set_title('Stake Distribution', fontweight='bold')

# Rewards trend (sample data)
x = np.arange(5)
rewards = [60, 65, 68, 70, 68]
ax4.plot(x, rewards, marker='o', linewidth=2, color='#2E86AB')
ax4.set_title('Rewards Trend', fontweight='bold')
ax4.set_xlabel('Time Period')

plt.tight_layout()
plt.savefig('/tmp/metrics_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved to /tmp/metrics_chart.png")
'''
    
    def _generate_generic_chart_code(self, query: str, answer: str, analysis: Dict[str, Any]) -> str:
        """Generate generic chart based on available data"""
        numbers = analysis.get("numbers", [])
        entities = analysis.get("entities", [])
        
        return f'''
import matplotlib.pyplot as plt
import numpy as np

# Create a simple visualization based on available data
fig, ax = plt.subplots(figsize=(10, 6))

# Sample data based on analysis
data = [1, 2, 3, 4, 5]
values = [20, 35, 30, 25, 40]

ax.plot(data, values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax.fill_between(data, values, alpha=0.3, color='#2E86AB')

ax.set_title('Data Visualization', fontsize=16, fontweight='bold')
ax.set_xlabel('Categories', fontsize=12)
ax.set_ylabel('Values', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/generic_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved to /tmp/generic_chart.png")
'''
    
    def _generate_workflow_chart_code(self, analysis: Dict[str, Any]) -> str:
        """Generate workflow/process chart code"""
        entities = analysis.get("entities", [])
        processes = analysis.get("processes", [])
        
        return f'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Create workflow diagram
fig, ax = plt.subplots(figsize=(12, 8))

# Define process steps
steps = ['Input', 'Process', 'Validate', 'Output']
if {entities}:
    steps = {entities[:4]} if len({entities}) >= 4 else steps

# Create workflow boxes
box_width = 2
box_height = 1
spacing = 3
start_x = 1

for i, step in enumerate(steps):
    x = start_x + i * spacing
    y = 4
    
    # Create rounded rectangle
    box = FancyBboxPatch((x, y), box_width, box_height,
                        boxstyle="round,pad=0.1",
                        facecolor='#2E86AB',
                        edgecolor='black',
                        alpha=0.8)
    ax.add_patch(box)
    
    # Add text
    ax.text(x + box_width/2, y + box_height/2, step,
           ha='center', va='center', fontweight='bold', color='white')
    
    # Add arrow to next step
    if i < len(steps) - 1:
        ax.arrow(x + box_width + 0.1, y + box_height/2, 
                spacing - box_width - 0.2, 0,
                head_width=0.2, head_length=0.2, fc='black', ec='black')

ax.set_xlim(0, start_x + len(steps) * spacing)
ax.set_ylim(2, 7)
ax.set_title('Process Workflow', fontsize=16, fontweight='bold')
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('/tmp/workflow_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved to /tmp/workflow_chart.png")
'''
    
    def _generate_comparison_chart_code(self, analysis: Dict[str, Any]) -> str:
        """Generate comparison chart code"""
        entities = analysis.get("entities", [])
        numbers = analysis.get("numbers", [])
        
        return f'''
import matplotlib.pyplot as plt
import numpy as np

# Comparison data
categories = {entities[:3] if entities else ['Option A', 'Option B', 'Option C']}
values1 = {numbers[:3] if len(numbers) >= 3 else [70, 85, 60]}
values2 = {numbers[3:6] if len(numbers) >= 6 else [65, 80, 75]}

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, values1, width, label='Metric 1', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, values2, width, label='Metric 2', color='#A23B72', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{{height:.0f}}', ha='center', va='bottom', fontweight='bold')

ax.set_title('Comparison Analysis', fontsize=16, fontweight='bold')
ax.set_xlabel('Categories', fontsize=12)
ax.set_ylabel('Values', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/comparison_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved to /tmp/comparison_chart.png")
'''
    
    def _execute_chart_code(self, python_code: str, query: str) -> Optional[str]:
        """Execute Python chart code and return the generated file path"""
        try:
            # Create a temporary Python file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(python_code)
                script_path = f.name
            
            # Execute the Python script
            result = subprocess.run(
                ['python', script_path], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            # Clean up script file
            os.unlink(script_path)
            
            if result.returncode == 0:
                # Look for the generated chart file
                chart_paths = ['/tmp/parametric_chart.png', '/tmp/reward_chart.png', '/tmp/metrics_chart.png', '/tmp/workflow_chart.png', '/tmp/comparison_chart.png', '/tmp/generic_chart.png']
                for path in chart_paths:
                    if os.path.exists(path):
                        # Move to a unique temporary location
                        unique_path = tempfile.mktemp(suffix='.png')
                        os.rename(path, unique_path)
                        return unique_path
                
                self.logger.warning("Chart code executed but no output file found")
                return None
            else:
                self.logger.error(f"Chart code execution failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing chart code: {str(e)}")
            return None


def create_chart_tool() -> FunctionTool:
    """Create a LlamaIndex FunctionTool for chart generation"""
    
    chart_generator = ChartGeneratorTool()
    
    def generate_chart_wrapper(
        x_label: Annotated[str, "Label for the x-axis"], 
        y_label: Annotated[str, "Label for the y-axis"],
        data_points: Annotated[str, "Comma-separated list of (x,y) coordinate pairs in format 'x1,y1;x2,y2;x3,y3'"],
        chart_type: Annotated[str, "Type of chart - either 'bar' or 'line'"]
    ) -> str:
        """Generate a chart with specific parameters
        
        Args:
            x_label: Label for the x-axis
            y_label: Label for the y-axis  
            data_points: Comma-separated list of (x,y) coordinate pairs
            chart_type: Type of chart - either 'bar' or 'line'
            
        Returns:
            File path of generated chart if successful, error message if failed
        """
        return chart_generator.generate_parametric_chart(x_label, y_label, data_points, chart_type)
    
    return FunctionTool.from_defaults(
        fn=generate_chart_wrapper,
        name="generate_chart", 
        description="""Generate bar or line charts with specific data points and axis labels.
        Use this tool when users request specific charts with data.
        
        Parameters:
        - x_label: The label for the x-axis (horizontal axis)
        - y_label: The label for the y-axis (vertical axis)  
        - data_points: The data as (x,y) pairs in format 'x1,y1;x2,y2;x3,y3' (e.g., 'Jan,10;Feb,20;Mar,15')
        - chart_type: Either 'bar' for bar charts or 'line' for line charts
        
        Returns the file path of the generated chart image."""
    )
