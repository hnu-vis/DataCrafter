import { intersectionOf } from "../utils/setOperation";

export class DataItem {
    key: string;

    constructor(key: string) {
        this.key = key;
    }
}

export class DataBase<T extends DataItem> {
    readonly data: T[];

    displayedDataIndices: Set<number>;
    selectedDataIndices: Set<number>;

    private itemIndexMap = new Map<string | T, number>();
    private indicesItemsMap = new Map<string, T[]>();

    private constructor(data: T[]) {
        this.data = data;
        this.displayedDataIndices = new Set(data.keys());
        this.selectedDataIndices = new Set(data.keys());
    }

    static of<T extends DataItem>(data: T[]) {
        return new DataBase(data);
    }

    getDataByIndices(indices: Set<number>): T[] {
        const indicesArray = Array.from(indices);
        const sortedIndices = indicesArray.sort((a, b) => a - b);
        const result = sortedIndices.map(i => this.data[i]);
        return result;
    }

    findItem(item: string | T): T {
        if (this.findIndex(item) == -1) {
            return null
        }
        return this.data[this.findIndex(item)]
    }

    findIndex(item: string | T): number {
        if (this.itemIndexMap.has(item)) {
            return this.itemIndexMap.get(item)!;
        }

        let index: number;
        if (typeof item === 'string') {
            index = this.data.findIndex(dataItem => dataItem.key === item);
        } else {
            index = this.data.indexOf(item);
        }

        if (index !== -1) {
            this.itemIndexMap.set(item, index);
        }

        return index;
    }

    findItems(items: (string | T)[]): T[] {
        return this.getDataByIndices(this.findIndices(items));
    }

    findIndices(items: (string | T)[]): Set<number> {
        const indices = new Set<number>();

        items.forEach((item: string | T) => {
            const index = this.findIndex(item);
            if (index !== -1) {
                indices.add(index);
            }
        });

        return indices;
    }

    filter(predicate: (item: T) => boolean): DataBase<T> {
        const filteredItems = this.data.filter(predicate);
        const filteredDatabase = DataBase.of(filteredItems);
        filteredDatabase.displayedDataIndices = intersectionOf(filteredDatabase.displayedDataIndices, this.displayedDataIndices);
        filteredDatabase.selectedDataIndices = intersectionOf(filteredDatabase.displayedDataIndices, this.selectedDataIndices);
        return filteredDatabase;
    }

    isAllDisplayed() {
        return this.displayedDataIndices.size === this.data.length;
    }

    isNoneDisplayed() {
        return this.displayedDataIndices.size === 0;
    }

    isAllSelected() {
        if (this.displayedDataIndices.size !== this.selectedDataIndices.size) {
            return false;
        }

        for (const index of Array.from(this.displayedDataIndices)) {
            if (!this.selectedDataIndices.has(index)) {
                return false;
            }
        }

        return true;
    }

    isNoneSelected() {
        return this.selectedDataIndices.size === 0;
    }

    reset() {
        this.displayedDataIndices = new Set(this.data.keys());
        this.selectedDataIndices = new Set(this.data.keys());
    }

    displayedData() {
        return this.getDataByIndices(this.displayedDataIndices);
    }

    displayedDataNum() {
        return this.displayedDataIndices.size;
    }

    selectedData() {
        return this.getDataByIndices(this.selectedDataIndices);
    }

    selectedDataNum() {
        return this.selectedDataIndices.size;
    }

    displayAll() {
        this.displayedDataIndices = new Set(this.data.keys());
    }

    selectAll() {
        this.selectedDataIndices = new Set(Array.from(this.displayedDataIndices));
    }

    displayNone() {
        this.displayedDataIndices = new Set();
        this.selectedDataIndices = new Set();
    }

    selectNone() {
        this.selectedDataIndices = new Set();
    }

    displayedDataContains(item: string | T): boolean {
        return this.displayedDataIndices.has(this.findIndex(item))
    }

    selectedDataContains(item: string | T): boolean {
        return this.selectedDataIndices.has(this.findIndex(item))
    }

    display(item: string | T | string[] | T[]) {
        this.displayNone();
        if (Array.isArray(item)) {
            this.addItemsToDisplayedData(item);
        } else {
            this.addItemToDisplayedData(item);
        }
    }

    addDisplay(item: string | T | string[] | T[]) {
        if (Array.isArray(item)) {
            this.addItemsToDisplayedData(item);
        } else {
            this.addItemToDisplayedData(item);
        }
    }

    hide(item: string | T | string[] | T[]) {
        if (Array.isArray(item)) {
            this.removeItemsFromDisplayedData(item);
        } else {
            this.removeItemFromDisplayedData(item);
        }
    }

    select(item: string | T | string[] | T[]) {
        this.selectNone();
        if (Array.isArray(item)) {
            this.addItemsToSelectedData(item);
        } else {
            this.addItemToSelectedData(item);
        }
    }

    addSelect(item: string | T | string[] | T[]) {
        if (Array.isArray(item)) {
            this.addItemsToSelectedData(item);
        } else {
            this.addItemToSelectedData(item);
        }
    }

    deselect(item: string | T | string[] | T[]) {
        if (Array.isArray(item)) {
            this.removeItemsFromSelectedData(item);
        } else {
            this.removeItemFromSelectedData(item);
        }
    }

    addItemToDisplayedData(item: string | T) {
        const index = this.findIndex(item);
        if (index !== -1) {
            this.displayedDataIndices.add(index);
        }
    }

    addItemToSelectedData(item: string | T) {
        const index = this.findIndex(item);
        if (index !== -1 && this.displayedDataIndices.has(index)) {
            this.selectedDataIndices.add(index);
        }
    }

    removeItemFromDisplayedData(item: string | T) {
        const index = this.findIndex(item);
        if (index !== -1) {
            this.displayedDataIndices.delete(index);
            this.selectedDataIndices.delete(index);
        }
    }

    removeItemFromSelectedData(item: string | T) {
        const index = this.findIndex(item);
        if (index !== -1) {
            this.selectedDataIndices.delete(index);
        }
    }

    toggleItemInDisplayedData(item: string | T) {
        const index = this.findIndex(item);
        if (index !== -1) {
            if (this.displayedDataIndices.has(index)) {
                this.displayedDataIndices.delete(index);
                this.selectedDataIndices.delete(index);
            } else {
                this.displayedDataIndices.add(index);
            }
        }
    }

    toggleItemInSelectedData(item: string | T) {
        const index = this.findIndex(item);
        if (index !== -1 && this.displayedDataIndices.has(index)) {
            if (this.selectedDataIndices.has(index)) {
                this.selectedDataIndices.delete(index);
            } else {
                this.selectedDataIndices.add(index);
            }
        }
    }

    addItemsToDisplayedData(items: string[] | T[]) {
        const indices = this.findIndices(items);
        indices.forEach(index => this.displayedDataIndices.add(index));
    }

    addItemsToSelectedData(items: string[] | T[]) {
        const indices = this.findIndices(items);
        indices.forEach(index => {
            if (this.displayedDataIndices.has(index)) {
                this.selectedDataIndices.add(index);
            }
        });
    }

    removeItemsFromDisplayedData(items: string[] | T[]) {
        const indices = this.findIndices(items);
        indices.forEach(index => {
            this.displayedDataIndices.delete(index);
            this.selectedDataIndices.delete(index);
        });
    }

    removeItemsFromSelectedData(items: string[] | T[]) {
        const indices = this.findIndices(items);
        indices.forEach(index => this.selectedDataIndices.delete(index));
    }
}
