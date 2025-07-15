"""
Chart generation tool for LlamaIndex FunctionAgent
Provides standalone chart generation functionality based on RAG content analysis
"""

import logging
import os
import uuid
from pathlib import Path
from llama_cloud import ImageBlock
from llama_index.core.llms import ChatMessage
import matplotlib
import matplotlib.pyplot as plt
from typing import Annotated, Callable, List, Literal, Optional
from llama_index.core.tools import FunctionTool
from config import get_config

matplotlib.use('Agg')  # Use non-interactive backend
logger = logging.getLogger("uvicorn.error")


class ChartGeneratorTool:
    def __init__(self):
        self.config = get_config()

    async def generate_chart(
        self,
        x_label: str,
        y_label: str,
        data_points: str,
        chart_type: Literal["bar", "line"],
    ) -> str:
        try:
            logger.info(f"Generating {chart_type} chart: '{x_label}' vs '{y_label}'")
            
            # Parse data points
            parsed_data = self._parse_data_points(data_points, chart_type)
            if not parsed_data:
                return "Invalid data points format. Use 'x1,y1;x2,y2;x3,y3'"
            
            # Generate chart directly in-process
            chart_path, chart_url = await self._generate_chart(x_label, y_label, parsed_data, chart_type)
            
            if chart_path and chart_url:
                logger.info(f"Chart generated successfully: {chart_path} -> {chart_url}")
                return chart_url
            else:
                return "Failed to generate chart"
                
        except Exception as e:
            logger.error(f"Failed to generate parametric chart: {str(e)}")
            return f"Failed to generate chart: {str(e)}"
    
    def _parse_data_points(self, data_points: str, chart_type: Literal["bar", "line"]) -> List[tuple]:
        """Parse data points string into list of (x, y) tuples"""
        try:
            points = []
            for point_pair in data_points.split(';'):
                if ',' in point_pair:
                    x_str, y_str = point_pair.strip().split(',', 1)
                    
                    y_val = float(y_str.strip())
                    if chart_type.lower() == 'line':
                        x_val = float(x_str.strip())
                    elif chart_type.lower() == 'bar':
                        x_val = x_str.strip()
                    else:
                        raise ValueError(f"Unsupported chart type: {chart_type}")
                    
                    points.append((x_val, y_val))
            
            return points
        except Exception as e:
            logger.error(f"Error parsing data points '{data_points}': {str(e)}")
            return []
    
    async def _generate_chart(self, x_label: str, y_label: str, data_points: List[tuple], chart_type: Literal["bar", "line"]) -> tuple[Optional[str], Optional[str]]:
        # Create figure with configured size
        fig, ax = plt.subplots(figsize=self.config.chart.figure_size, dpi=self.config.chart.dpi)

        # Separate x and y values
        x_values = [str(point[0]) for point in data_points]
        y_values = [float(point[1]) for point in data_points]

        # Create the chart based on type
        if chart_type.lower() == 'line':
            ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=8, color='steelblue')
            ax.grid(True, alpha=0.3)
        else:  # Default to bar chart
            bars = ax.bar(x_values, y_values, color='steelblue', alpha=0.7)
            # Add value labels on bars
            for bar, value in zip(bars, y_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(y_values) * 0.01,
                       f'{value}', ha='center', va='bottom')

        # Set labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{y_label} vs {x_label}', fontsize=14, fontweight='bold')

        # Improve layout
        plt.tight_layout()

        # Generate unique filename and save to static directory
        chart_filename = f"chart_{uuid.uuid4().hex[:8]}.png"
        static_dir = Path("static/images")
        static_dir.mkdir(parents=True, exist_ok=True)
        chart_path = static_dir / chart_filename
        
        plt.savefig(str(chart_path), dpi=self.config.chart.dpi, bbox_inches='tight')
        plt.close()

        # Construct the public URL for the chart
        chart_url = f"{self.config.base_url}/images/{chart_filename}"
        
        logger.info(f"Chart generated successfully: {chart_path} -> {chart_url}")
        return str(chart_path), chart_url


async def generate_chart(
    x_label: Annotated[str, "Label for the x-axis"],
    y_label: Annotated[str, "Label for the y-axis"],
    data_points: Annotated[str, "Comma-separated list of (x,y) coordinate pairs in format 'x1,y1;x2,y2;x3,y3'"],
    chart_type: Annotated[Literal["bar", "line"], "Type of chart - either 'bar' or 'line'"],
):
    chart_generator = ChartGeneratorTool()
    return await chart_generator.generate_chart(x_label, y_label, data_points, chart_type)

chart_tool = FunctionTool.from_defaults(
    fn=generate_chart,
    name="generate_chart",
    description="""Generate bar or line charts with specific data points and axis labels.  Use this tool when users request specific charts with data."""
)
