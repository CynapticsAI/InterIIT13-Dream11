export const COLORS = {
    primary: "#e10000", // Dream11 Red
    secondary: "#3f8efc", // Dream11 Blue
    tertiary: "#ffd152", // Dream11 Khaki
    success: "#17c749", // Dream11 Bluish Green
    warning: "#ff5407", // Dream11 Vermilion
    background: "#ffffff", // White
    text: "#1d1d1d", // Dark Gray
    lightText: "#6a6b69", // Medium Gray
    border: "#e1e1e1", // Light Gray
  }
  
  export const getColorArray = (length: number) => {
    const baseColors = [
      COLORS.primary,
      COLORS.secondary,
      COLORS.tertiary,
      COLORS.success,
      COLORS.warning,
    ]
    return Array(length).fill(0).map((_, i) => baseColors[i % baseColors.length])
  }
  
  