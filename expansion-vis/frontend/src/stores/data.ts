import { defineStore } from 'pinia';
import { DataItem, DataBase } from './dataBase';
import axios from 'axios';
import { config } from '../config/config';
import { removeLabelOverlap } from '../utils/labelLayout';
import type { Label } from '../utils/labelLayout';


/**
 * - Word: `[x, y, word, frequency, originalNum, generatedNum, isParent(0|1)]`
 * 
 * - Image: `[x, y, path, isGenerated(0|1)]`
 */
export type ProjectionData = {
    [category: string]: {
        [index: number]: [number, number, string, number, number, number, number]
    }
}

export type PromptsData = {
    [category: string]: {
        text: string,
        accepted: boolean,
        original_text: string
    }[]
}

/**
 * - Image: `[x, y, category, isGenerated(0|1), url, originalUrl, prompt, template, captions]`
 */
export type ImageData = [number, number, string, number, string, string, string, string, string[]]
export type ImagesData = ImageData[]

export class Image extends DataItem {
    x: number;
    y: number;
    category: string;
    isGenerated: boolean;
    originalUrl: string;
    prompt: string;
    template: string;
    captions: string[];

    constructor(x: number, y: number, category: string, isGenerated: boolean | number, url: string, originalUrl: string, prompt: string, template: string, captions: string[]) {
        super(url);
        this.x = x;
        this.y = y;
        this.category = category;
        if (typeof isGenerated == 'boolean') {
            this.isGenerated = isGenerated;
        } else {
            this.isGenerated = isGenerated == 1;
        }
        this.originalUrl = originalUrl;
        this.prompt = prompt;
        this.template = template;
        this.captions = captions
    }

    static new(image: ImageData) {
        return new Image(...image);
    }
}

const getImagesFromImagesData = (images: ImagesData): DataBase<Image> => {
    return DataBase.of(images.map(image => {
        image[2] = image[2].replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
        return Image.new(image);
    }))
}

/**
 * - Word: `[x, y, word, isParent(0|1), originalNum, generatedNum, relatedWords]`
 */
export type WordData = [number, number, string, number, number, number, string[]]
export type WordsData = WordData[]

export class Word extends DataItem {
    x: number;
    y: number;
    isParent: boolean;
    originalNum: number;
    generatedNum: number;
    relatedWords: string[];

    constructor(x: number, y: number, word: string, isParent: boolean | number, originalNum: number, generatedNum: number, relatedWords: string[]) {
        super(word);
        this.x = x;
        this.y = y;
        if (typeof isParent == 'boolean') {
            this.isParent = isParent;
        } else {
            this.isParent = isParent == 1;
        }
        this.originalNum = originalNum;
        this.generatedNum = generatedNum;
        this.relatedWords = relatedWords;
    }

    static new(word: WordData) {
        return new Word(...word);
    }

    frequency = () => {
        return this.originalNum + this.generatedNum;
    }
}

const getWordsFromWordsData = (words: WordsData): DataBase<Word> => {
    return DataBase.of(words.map(word => Word.new(word)))
}

export function getFontSize(frequency: number) {
    let size: number;
    if (frequency < 10) size = 8
    else if (frequency < 20) size = 10
    else if (frequency < 50) size = 12
    else if (frequency < 100) size = 14
    else size = 16
    return size * 1.5
}

export const getHeight = (word: string, fontSize: number) => {
    return fontSize * (4 / 3) * 1.02 / 591
}

const getRelativeWidth = (word: string): number => {
    const widths: { [key: string]: number } = {
        'l': 0.5,
        'i': 0.5,
        'j': 0.5,
        'f': 0.75, 'r': 0.5,
        't': 0.75,
    };

    let width = 0;
    for (const char of word) {
        if (widths[char] !== undefined) {
            width += widths[char];
        } else {
            width += 1;
        }
    }

    return width;
};

export const getWidth = (word: string, fontSize: number) => {
    const pieR = fontSize * 4 / 5
    const relativeWidth = getRelativeWidth(word)
    return (pieR + relativeWidth * fontSize / 1.6) * 1.02 / 1045
}

export const getPieWidth = (word: string, fontSize: number) => {
    return (fontSize * 4 / 5) * 1.02 / 1045
}

function removeOverlap(words: Word[]) {
    const labels: Label[] = words.map((word) => {
        const fontSize = getFontSize(word.generatedNum + word.originalNum)
        const height = getHeight(word.key, fontSize)
        const width = getWidth(word.key, fontSize)
        return {
            x: word.x,
            y: word.y,
            height: height,
            width: width
        }
    })
    const newLabels = removeLabelOverlap(labels, 0.01, 0.01, 0.01, 10)
    words.forEach((word, index) => {
        let newX: number = newLabels[index].x;
        let newY: number = newLabels[index].y;
        const width = newLabels[index].width;
        const height = newLabels[index].height;
        if (newX + width > 1.01) {
            newX = 1.01 - width + getPieWidth(word.key, getFontSize(word.generatedNum + word.originalNum) * 2);
        }
        if (newY + height > 1.01) {
            newY = 1.01 - height;
        }
        word.x = newX;
        word.y = newY;
    })
    return words
}

export class Prompt extends DataItem {
    category: string;
    template: string;

    constructor(key: string, category: string) {
        super(key);
        this.category = category;
    }

    static new(key: string, category: string) {
        return new Prompt(key, category);
    }
}

export class Template extends DataItem {
    category: string;
    prompts: string[];
    oldTemplate: string;
    isAccepted: boolean = true;
    isChecked: boolean = false;
    isSelecting: boolean = false;
    isReady: boolean = false;

    constructor(key: string, category: string, prompts: string[]) {
        super(key);
        this.category = category;
        this.prompts = prompts;
        this.oldTemplate = key;
    }

    static new(key: string, category: string, prompts: string[]) {
        return new Template(key, category, prompts)
    }
}

const getTemplatesFromImages = (images: DataBase<Image>): Template[] => {
    const promptMap = new Map<string, { category: string; prompts: string[] }>()

    images.data
        .filter((item) => item.isGenerated)
        .forEach((item) => {
            if (!promptMap.has(item.template)) {
                promptMap.set(item.template, {
                    category: item.category,
                    prompts: [item.prompt]
                })
            } else {
                promptMap.get(item.template).prompts.push(item.prompt)
            }
        })

    return Array.from(promptMap.entries()).map(([key, value]) =>
        Template.new(key, value.category, value.prompts)
    )
}

export type MetricData = {
    Distance: number
    Diversity: number
    Informativeness: number
}

export type DataStoreInterface = {
    data: {
        images: DataBase<Image>,
        words: DataBase<Word>,
        templates: Template[],
        metric: MetricData
    }[]
    currentDataIndex: number
    compareWords: (word: Word) => number
    isReversed: boolean
}

export const getOriginalNum = (word: Word, images: Image[]): number => {
    const exceptWords = ['nature', 'structure', 'environment']
    let result = 0;
    if (word.isParent || exceptWords.includes(word.key)) {
        return word.originalNum
    }
    images.forEach(image => {
        if (image.isGenerated) {
            return
        }
        if (image.captions.includes(word.key)) {
            result++;
        }
    })
    return result;
}

export const getGeneratedNum = (word: Word, images: Image[]): number => {
    const exceptWords = ['nature', 'structure', 'environment']
    let result = 0;
    if (word.isParent || exceptWords.includes(word.key)) {
        return word.generatedNum
    }
    images.forEach(image => {
        if (!image.isGenerated) {
            return
        }
        if (image.captions.includes(word.key)) {
            result++;
        }
    })
    return result;
}

export const GORatioMethod = (images: Image[]) => (word: Word) => {
    const generatedNum = getGeneratedNum(word, images)
    const originalNum = getOriginalNum(word, images) || 1
    return generatedNum / originalNum
}

export const GFRatioMethod = (images: Image[]) => (word: Word) => {
    const generatedNum = getGeneratedNum(word, images)
    const originalNum = getOriginalNum(word, images)
    const frequency = (generatedNum + originalNum) || 1
    return generatedNum / frequency
}

export const OFRatioMethod = (images: Image[]) => (word: Word) => {
    const generatedNum = getGeneratedNum(word, images)
    const originalNum = getOriginalNum(word, images)
    const frequency = (generatedNum + originalNum) || 1
    return originalNum / frequency
}

export const generatedNumMethod = (images: Image[]) => (word: Word) => {
    return getGeneratedNum(word, images)
}

export const originalNumMethod = (images: Image[]) => (word: Word) => {
    return getOriginalNum(word, images)
}

export const frequencyMethod = (images: Image[]) => (word: Word) => {
    return getGeneratedNum(word, images) + getOriginalNum(word, images)
}

export const DataStore = defineStore('dataStore', {
    state: (): DataStoreInterface => ({
        data: [],
        currentDataIndex: -1,
        compareWords: (word) => (word.generatedNum / (word.originalNum || 1)),
        isReversed: false
    }),

    getters: {
        words(state: DataStoreInterface): DataBase<Word> {
            if (state.currentDataIndex < 0) {
                return DataBase.of([] as Word[]);
            }
            return state.data[state.currentDataIndex]?.words;
        },

        images(state: DataStoreInterface): DataBase<Image> {
            if (state.currentDataIndex < 0) {
                return DataBase.of([] as Image[]);
            }
            return state.data[state.currentDataIndex]?.images;
        },

        templates(state: DataStoreInterface): Template[] {
            if (state.currentDataIndex < 0) {
                return [] as Template[];
            }
            return state.data[state.currentDataIndex]?.templates;
        },

        originalImages(state: DataStoreInterface): DataBase<Image> {
            return (this.images as DataBase<Image>).filter(item => !item.isGenerated)
        },

        generatedImages(state: DataStoreInterface): DataBase<Image> {
            return (this.images as DataBase<Image>).filter(item => item.isGenerated)
        },

        metric(state: DataStoreInterface): MetricData {
            if (state.currentDataIndex < 0) {
                return null;
            }
            return state.data[state.currentDataIndex]?.metric;
        },

        prompts(state: DataStoreInterface): Set<string> {
            const images: DataBase<Image> = this.images
            return images.data.reduce((prompts: Set<string>, image: Image) => prompts.add(image.prompt), new Set<string>())
        },

        categories(state: DataStoreInterface): Set<string> {
            const images: DataBase<Image> = this.images
            return images.data.reduce((categories: Set<string>, image: Image) => categories.add(image.category), new Set<string>())
        },

        sortedWords(state: DataStoreInterface): Word[] {
            const words: DataBase<Word> = this.words;
            if (!words) {
                return [] as Word[];
            }
            return [...words.data].sort((a, b) => (state.compareWords(b) - state.compareWords(a)) * (state.isReversed ? -1 : 1));
        }
    },

    actions: {
        setData(imagesData: ImagesData, wordsData: WordsData, metricData: MetricData) {
            const images = getImagesFromImagesData(imagesData)
            const words = getWordsFromWordsData(wordsData)
            const newWords = DataBase.of(removeOverlap(words.data))
            const templates = getTemplatesFromImages(images)

            this.data.push({
                images: images,
                words: newWords,
                templates: templates,
                metric: metricData
            })

            this.currentDataIndex += 1;

            this.compareWords = GORatioMethod(images.selectedData())
        },

        reset() {
            this.data = [];
            this.currentDataIndex = -1;
            this.compareWords = (word: Word) => (word.generatedNum / (word.originalNum || 1));
            this.isReversed = false
        },

        changeIndex(index: number) {
            if (index < 0 || index > this.data.length) { return; }
            this.currentDataIndex = index;
        },

        getRelatedImages(words: string | Word | (string | Word)[]): Image[] {
            const wordArray = Array.isArray(words) ? words : [words];

            const allCaptions = new Set<string>();
            wordArray.forEach(word => {
                const caption = typeof word === 'string' ? word : word.key;
                allCaptions.add(caption);
            });

            const relatedImages = (this.images.data as Image[]).filter(image =>
                image.captions.some(caption => allCaptions.has(caption))
            );

            return Array.from(new Set(relatedImages));
        },

        getRelatedWords(words: string | Word | (string | Word)[]): Word[] {
            const wordArray = Array.isArray(words) ? words : [words];

            const allCaptions = new Set<string>();
            wordArray.forEach(word => {
                const caption = typeof word === 'string' ? word : word.key;
                allCaptions.add(caption);
            });

            const allRelatedWords = new Set<string>();
            (this.words.data as Word[]).filter(word =>
                wordArray.includes(word.key)
            ).forEach(word => {
                word.relatedWords.forEach(item => allRelatedWords.add(item))
            });

            const relatedWords = (this.words.data as Word[]).filter(word => allRelatedWords.has(word.key));

            return relatedWords;
        },

        getCaptions(images: string | Image | (string | Image)[]): Word[] {
            const imageArray = Array.isArray(images) ? images : [images];

            const allCaptions = new Set<string>();

            imageArray.forEach((image: string | Image) => {
                let captions: string[];
                if (typeof image == 'string') {
                    captions = (this.images.data as Image[]).find(item => item.key == image).captions;
                } else {
                    captions = image.captions;
                }
                captions.forEach(caption => allCaptions.add(caption));
            });

            const relatedWords = (this.words.data as Word[]).filter(word => allCaptions.has(word.key));

            return relatedWords;
        },

        getImageGroup(images: string | Image | (string | Image)[]): [Image, Image[], Image[]][] {
            const inputArray = Array.isArray(images) ? images : [images];

            const imageArray = (this.images as DataBase<Image>).findItems(inputArray)

            const originalImagesSet = new Set<Image>();

            imageArray.forEach(image => {
                const imageObject = typeof image === 'string' ? (this.images as DataBase<Image>).findItem(image) : image;

                if (imageObject.isGenerated) {
                    const originalImage = (this.images as DataBase<Image>).findItem(imageObject.originalUrl);
                    originalImagesSet.add(originalImage);
                } else {
                    originalImagesSet.add(imageObject);
                }
            });

            // [originalImage, selectedGeneratedImages, unselectedGeneratedImages]
            const result: [Image, Image[], Image[]][] = [];
            originalImagesSet.forEach(originalImage => {
                const generatedImages = (this.images as DataBase<Image>).data.filter(image => image.isGenerated && (image.originalUrl === originalImage.key));
                const inImages: Image[] = []
                const notInImages: Image[] = []
                generatedImages.forEach(item => {
                    if (imageArray.includes(item)) {
                        inImages.push(item)
                    } else {
                        notInImages.push(item)
                    }
                })
                result.push([originalImage, inImages, notInImages]);
            });

            return result;
        },

        getImagesFromPrompt(prompt: string | Prompt): Image[] {
            let key: string | undefined;

            if (typeof prompt == 'string') {
                key = prompt
            } else {
                key = prompt.key
            }

            if (key) {
                return this.images.data.filter((image: Image) => key === image.prompt);
            }

            return [] as Image[];
        },

        deleteDataAtIndex(index: number) {
            if (index < 0 || index >= this.data.length) {
                return;
            }

            this.data.splice(index, 1);

            if (this.currentDataIndex >= this.data.length) {
                this.currentDataIndex = this.data.length - 1;
            }

            if (this.data.length === 0) {
                this.currentDataIndex = -1;
            }
        }
    }
})