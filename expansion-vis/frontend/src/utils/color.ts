export const rgbToRgba = (rgb: string, alpha: number): string => {
    const result = rgb.match(/\d+/g)

    if (!result) {
        return rgb
    }

    const [r, g, b] = result.map(Number)
    return `rgba(${r},${g},${b},${alpha})`
}

export const getLighterColor = (rgbColor: string, lightenFactor: number = 0.8): string => {
    const rgbValues = rgbColor.match(/\d+/g)

    if (!rgbValues || rgbValues.length !== 3) {
        throw new Error('Invalid RGB color format')
    }

    let [r, g, b] = rgbValues.map(Number)

    r = Math.min(Math.floor(r + (255 - r) * (1 - lightenFactor)), 255)
    g = Math.min(Math.floor(g + (255 - g) * (1 - lightenFactor)), 255)
    b = Math.min(Math.floor(b + (255 - b) * (1 - lightenFactor)), 255)

    return `rgb(${r}, ${g}, ${b})`
}
