import * as d3 from 'd3'

export type Label = {
    x: number,
    y: number,
    width: number,
    height: number,
    vx?: number, // Velocity in x direction
    vy?: number,  // Velocity in y direction
}

export type Labels = Label[]

// Function to remove overlap between labels
export function removeLabelOverlap(labels: Labels, alpha: number = 1000, gravity: number = 0.1, padding_ratio: number = 1, maxIterations: number = 5): Labels {
    // Initialize each label's velocity and record the original position
    let resultLabels = labels.map(label => ({ ...label, vx: 0, vy: 0, originalX: label.x, originalY: label.y }));

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        // Build a quadtree for efficient spatial searching
        const quad = d3.quadtree(resultLabels, d => d.x, d => d.y);

        for (const label of resultLabels) {
            // Traverse each node in the quadtree to find nearby labels
            quad.visit((q, x1, y1, x2, y2) => {
                if (q.data && q.data !== label) {
                    let x = label.x - q.data.x
                    let y = label.y - q.data.y
                    let absX = Math.abs(x)
                    let absY = Math.abs(y)
                    let xSpacing = (1 + padding_ratio) * (q.data.width + label.width) / 2 // Adjust xSpacing
                    let ySpacing = (1 + padding_ratio) * (q.data.height + label.height) / 2 // Adjust ySpacing

                    // If labels are within a certain range, calculate repulsive force
                    if (absX < xSpacing && absY < ySpacing) {
                        const l = Math.sqrt(x * x + y * y);
                        const area = (xSpacing - absX) * (ySpacing - absY)
                        if (area > 0) {
                            const repulse = (1 / l - 1 / (Math.sqrt(xSpacing * xSpacing + ySpacing * ySpacing))) * alpha;

                            label.vx += x * repulse * 1;
                            label.vy += y * repulse * 1.1;
                            q.data.vx -= x * repulse * 1;
                            q.data.vy -= y * repulse * 1.1;
                        }
                    }
                }
                return false;
            });

            // Calculate the gravitational pull of the label towards its original position
            let x = label.originalX - label.x;
            let y = label.originalY - label.y;
            label.vx += x * gravity;
            label.vy += y * gravity;
        }

        // Update each label's position
        for (const label of resultLabels) {
            label.x += label.vx;
            label.y += label.vy;
            label.vx = 0;
            label.vy = 0;
        }
    }

    return resultLabels;
}
