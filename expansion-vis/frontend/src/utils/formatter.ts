export const formatToFixed = (num: number, decimalPlaces: number = 2) =>
    Number.isInteger(num) ? num.toFixed(0) : num.toFixed(decimalPlaces)

export const capitalizeFirstLetter = (str: string) =>
    str.charAt(0).toUpperCase() + str.slice(1).toLowerCase()
