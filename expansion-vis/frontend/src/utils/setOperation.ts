export const intersectionOf = <T>(setA: Set<T>, setB: Set<T>): Set<T> => {
    return new Set(Array.from(setA).filter(item => setB.has(item)));
}

export const unionOf = <T>(setA: Set<T>, setB: Set<T>): Set<T> => {
    let result = new Set<T>();
    setA.forEach(item => result.add(item));
    setB.forEach(item => result.add(item));
    return result;
}
