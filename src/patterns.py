import numpy as np
from shapely import LineString, Polygon

class PatternsForUnitSquare:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin;self.xmax = xmax;self.ymin = ymin;self.ymax = ymax
    def extractPoints(self,cntr, pointSpacing,reverseOrder=False): 
        line = LineString(cntr);num_points = max(int(line.length / pointSpacing), 2)
        distances = np.linspace(0, line.length, num_points)
        points = np.array([line.interpolate(distance).coords[0] for distance in distances])
        return points[::-1] if reverseOrder else points
    def squareWave(self, axis='y', step_size=0.1):
        margin=0.05
        x_grid = np.arange(self.xmin+margin, self.xmax-margin+step_size, step_size)
        y_grid = np.arange(self.ymin+margin, self.ymax-margin+step_size, step_size)
        x_scan, y_scan= [],[]
        if axis == 'x':
            for i, yi in enumerate(y_grid):
                x_scan.append(x_grid[::(-1)**i])  # Reverse when i is odd
                y_scan.append(np.ones_like(x_grid) * yi)
        elif axis == 'y':
            for i, xi in enumerate(x_grid):
                y_scan.append(y_grid[::(-1)**i])  # Reverse when i is odd
                x_scan.append(np.ones_like(y_grid) * xi)
        x_scan = np.concatenate(x_scan)
        y_scan = np.concatenate(y_scan)
        points = list(zip(x_scan, y_scan))
        # x, y = zip(*points);plt.plot(x, y, 'o-');plt.xlabel('X-axis')
        # plt.ylabel('Y-axis');plt.title('Square Wave Pattern');plt.grid(True);plt.show()
        return points


    def compute_step_size(self,period, amplitude_u, amplitude_v, num_points):
        """
        Compute the step size required to get exactly `num_points` inside the square wave.
        """
        total_path_length = (2 * period * amplitude_v) + ((2 * period - 1) * amplitude_u)
        step_size = total_path_length / (num_points - 1)  # (N-1) gaps for N points
        return step_size

    def squareWaveGaiotto(self,period=2, num_points=69, amplitude_u=0.7, amplitude_v=0.9, 
                offset_u=0.15, offset_v=0.05, axis='y'):
        """
        Generate a square wave pattern inside a defined area with exactly `num_points`.
        """
        step_size = self.compute_step_size(period, amplitude_u, amplitude_v, num_points)
        
        if axis == 'y':
            x_grid = np.linspace(offset_u, offset_u + amplitude_u, 2 * period + 1)
            y_grid = np.linspace(offset_v, offset_v + amplitude_v, int(amplitude_v / step_size))

            x_scan, y_scan = [], []
            for i, xi in enumerate(x_grid):
                y_scan.append(y_grid[::(-1)**i])  # Reverse direction for each column
                x_scan.append(np.ones_like(y_grid) * xi)

        elif axis == 'x':
            y_grid = np.linspace(offset_v, offset_v + amplitude_v, 2 * period + 1)
            x_grid = np.linspace(offset_u, offset_u + amplitude_u, int(amplitude_u / step_size))

            x_scan, y_scan = [], []
            for i, yi in enumerate(y_grid):
                x_scan.append(x_grid[::(-1)**i])  # Reverse direction for each row
                y_scan.append(np.ones_like(x_grid) * yi)

        x_scan = np.concatenate(x_scan)
        y_scan = np.concatenate(y_scan)
        
        points = list(zip(x_scan, y_scan))
        boundary_x = [0, 1, 1, 0, 0]
        boundary_y = [0, 0, 1, 1, 0]

        # Extract trajectory points
        x, y = zip(*points)

        # Define corner colors & positions
        corner_colors = ['blue', 'magenta', 'yellow', 'gray']
        corner_positions = [(0, 0), (0, 1), (1, 1), (1, 0)]

        # # Plot the square boundary
        # plt.figure(figsize=(5, 5))
        # plt.plot(boundary_x, boundary_y, 'k--', linewidth=1.5, label="Boundary")  # Square boundary
        # plt.plot(x, y, 'o-', markersize=3, label="Square Wave")  # Square wave path

        # # Start and end points (Green = Start, Red = End)
        # plt.scatter(x[0], y[0], color='green', s=80, label="Start Point")  
        # plt.scatter(x[-1], y[-1], color='red', s=80, label="End Point")  

        # # Highlight the four corners as squares
        # for (cx, cy), color in zip(corner_positions, corner_colors):
        #     plt.scatter(cx, cy, color=color, s=150, marker='s', edgecolors='None', linewidth=2)

        # # Axis labels and styling
        # plt.xlabel('U')
        # plt.ylabel('V')
        # plt.grid(True)
        # plt.show()
        return points



    def multiCirclePattern(self, center=(0.5, 0.5),spacing=0.01, numOfCircles=1):
        numOfPoints=100
        scan = []
        if numOfCircles == 1:
            radii = [min(center[0], center[1], self.xmax - center[0], self.ymax - center[1])]
        else:
            radii = np.linspace(0, min(center[0], center[1], self.xmax - center[0], self.ymax - center[1]), numOfCircles + 1)[1:]
        theta = np.linspace(0, 2 * np.pi, numOfPoints)
        for r in radii:
            x_scan = center[0] + r * np.cos(theta)
            y_scan = center[1] + r * np.sin(theta)
            points = list(zip(x_scan, y_scan))
            points=self.extractPoints(LineString(points),spacing)
            scan.extend(points)
        # x,y=zip(*scan);plt.plot(x,y,'o-');plt.xlabel('X-axis');plt.ylabel('Y-axis');plt.title('Circular Pattern');plt.grid(True)
        # plt.axis('equal');plt.show()
        return scan
    def smallSquare(self, margin=0.1, pointSpacing=0.05):
        new_xmin = self.xmin + margin
        new_xmax = self.xmax - margin
        new_ymin = self.ymin + margin
        new_ymax = self.ymax - margin
        corners = [
            (new_xmin, new_ymin),  # Bottom-left
            (new_xmin, new_ymax),  # Top-left
            (new_xmax, new_ymax),  # Top-right
            (new_xmax, new_ymin),  # Bottom-right
            (new_xmin, new_ymin)   # Back to Bottom-left to close the square
        ]
        boundary_points = self.extractPoints(LineString(corners), pointSpacing)
        return boundary_points

    def multiSquarePattern(self, spacing=0.01, margin=0.0, square_spacing=0.02):
        """
        Generates multiple nested square boundaries inside the unit square,
        each separated by 'square_spacing' distance from the previous one and the outer boundary.
        
        :param spacing: distance between points along the square boundary
        :param margin: margin from the outer boundary to the first square
        :param square_spacing: distance between successive nested squares
        :return: list of (x, y) points forming the nested square pattern
        """
        center = (0.5, 0.5)
        scan = []

        # Initialize square boundaries
        current_xmin = self.xmin + margin
        current_xmax = self.xmax - margin
        current_ymin = self.ymin + margin
        current_ymax = self.ymax - margin

        while current_xmin < current_xmax and current_ymin < current_ymax:
            # Define the current square's corners
            corners = [
                (current_xmin, current_ymin),  # Bottom-left
                (current_xmin, current_ymax),  # Top-left
                (current_xmax, current_ymax),  # Top-right
                (current_xmax, current_ymin),  # Bottom-right
                (current_xmin, current_ymin)   # Close the square
            ]

            # Sample points along the square boundary
            boundary_points = self.extractPoints(LineString(corners), spacing)
            scan.extend(boundary_points)

            # Shrink for the next nested square
            current_xmin += square_spacing
            current_xmax -= square_spacing
            current_ymin += square_spacing
            current_ymax -= square_spacing

            # Stop if square collapses
            if current_xmin >= current_xmax or current_ymin >= current_ymax:
                break

        return scan


    def generate_rose_curve(self,  k=5, radius=0.4, num_points=100):
        center=(0.5, 0.5)

        theta = np.linspace(0, 2 * np.pi, num_points)
        r = radius * np.cos(k * theta)  # Rose curve equation

        # Calculate x and y coordinates of the points
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)

        return list(zip(x, y))



    def spiralPattern(self, center=(0.5, 0.5), maxRadius=0.45, numOfPoints=100, numOfTurns=5):
        theta = np.linspace(0, 2 * np.pi * numOfTurns, numOfPoints)
        r = np.linspace(0, maxRadius, numOfPoints)
        x_scan = center[0] + r * np.cos(theta);y_scan = center[1] + r * np.sin(theta)
        points = list(zip(x_scan, y_scan));scan = self.extractPoints(LineString(points),((LineString(points)).length)/numOfPoints)
        # x,y=zip(*scan);plt.plot(x,y,'o-');plt.xlabel('X-axis');plt.ylabel('Y-axis')
        # plt.title('Spiral Pattern');plt.grid(True);plt.axis('equal')
        # plt.show()
        return scan
    def outwardRectangularSpiral(self,center=(0.5, 0.5), bbox=[0, 1, 0, 1], increment=0.1):
    # Initialize the starting point and direction
        x, y = center
        points = [(x, y)]
        dx, dy = 0, increment  
        steps = 1;total_steps_taken = 0
        while True:
            for _ in range(2):  
                for _ in range(steps):
                    x += dx
                    y += dy
                    if x < bbox[0] or x > bbox[1] or y < bbox[2] or y > bbox[3]:
                        return points  
                    points.append((x, y))
                if dx == 0 and dy == increment:
                    dx, dy = increment, 0  # Move right
                elif dx == increment and dy == 0:
                    dx, dy = 0, -increment  # Move down
                elif dx == 0 and dy == -increment:
                    dx, dy = -increment, 0  # Move left
                elif dx == -increment and dy == 0:
                    dx, dy = 0, increment  # Move up