<template>
    <div class="categories">
        <button
            v-for="category in totalCategories"
            :key="category"
            class="category-button"
            :style="
                displayedCategories.includes(category) && getPromptCount(category) !== 0
                    ? `background-color: ${colorMap[category]}; color: black`
                    : 'background-color: #ddd; color: white'
            "
            @click="clickCategory(category)"
            :disabled="getPromptCount(category) === 0 && !categories.includes(category)"
        >
            {{ category }}
        </button>
    </div>
    <TransitionGroup name="list" class="templates" tag="div">
        <div
            v-for="promptItem in displayedTemplates.filter(
                (prompt) => !removedTemplates.includes(prompt)
            )"
            :key="promptItem.oldTemplate"
            class="template-item"
        >
            <div v-if="promptItem.isAccepted" class="template-item-tools">
                <button class="logo" @click="editTemplate(promptItem)">
                    <svg width="14" height="14" viewBox="0 0 1024 1024">
                        <path
                            d="M835.368421 458.105263c0-21.557895 18.863158-40.421053 40.421053-40.421052s40.421053 18.863158 40.421052 40.421052v323.368421c0 88.926316-72.757895 161.684211-161.68421 161.684211H215.578947c-88.926316 0-161.684211-72.757895-161.68421-161.684211V242.526316c0-88.926316 72.757895-161.684211 161.68421-161.684211h538.947369c21.557895 0 40.421053 18.863158 40.421052 40.421053S776.084211 161.684211 754.526316 161.684211H215.578947C169.768421 161.684211 134.736842 196.715789 134.736842 242.526316v538.947368c0 45.810526 35.031579 80.842105 80.842105 80.842105h538.947369c45.810526 0 80.842105-35.031579 80.842105-80.842105V458.105263z"
                            fill="rgb(119, 119, 119)"
                            stroke="rgb(119, 119, 119)"
                            stroke-width="20"
                        />
                        <path
                            d="M417.684211 727.578947c-10.778947 0-21.557895-2.694737-29.642106-10.778947-16.168421-16.168421-16.168421-40.421053 0-56.589474l512-512c8.084211-10.778947 18.863158-13.473684 29.642106-13.473684 10.778947 0 21.557895 2.694737 29.642105 10.778947 16.168421 16.168421 16.168421 40.421053 0 56.589474l-512 512c-10.778947 10.778947-18.863158 13.473684-29.642105 13.473684z"
                            fill="rgb(119, 119, 119)"
                            stroke="rgb(119, 119, 119)"
                            stroke-width="20"
                        />
                    </svg>
                </button>
                <button class="logo" @click="removeTemplate(promptItem)">
                    <!-- ❌ -->
                    <svg width="18" height="18" viewBox="0 0 1024 1024">
                        <path
                            d="M853.333333 298.666667l-44.8 0c2.133333 6.4 2.133333 14.933333 2.133333 21.333333l0 490.666667c0 46.933333-38.4 85.333333-85.333333 85.333333L298.666667 896c-46.933333 0-85.333333-38.4-85.333333-85.333333L213.333333 320c0-6.4 2.133333-14.933333 2.133333-21.333333L170.666667 298.666667c-12.8 0-21.333333-8.533333-21.333333-21.333333s8.533333-21.333333 21.333333-21.333333l149.333333 0L320 170.666667c0-23.466667 19.2-42.666667 42.666667-42.666667l320 0c23.466667 0 42.666667 19.2 42.666667 42.666667l0 85.333333 128 0c12.8 0 21.333333 8.533333 21.333333 21.333333S866.133333 298.666667 853.333333 298.666667zM682.666667 170.666667 362.666667 170.666667l0 85.333333 320 0L682.666667 170.666667zM256 298.666667l0 512c0 23.466667 19.2 42.666667 42.666667 42.666667l426.666667 0c23.466667 0 42.666667-19.2 42.666667-42.666667L768 298.666667 256 298.666667zM661.333333 725.333333c-12.8 0-21.333333-8.533333-21.333333-21.333333L640 426.666667c0-12.8 8.533333-21.333333 21.333333-21.333333s21.333333 8.533333 21.333333 21.333333l0 277.333333C682.666667 716.8 674.133333 725.333333 661.333333 725.333333zM512 746.666667c-12.8 0-21.333333-8.533333-21.333333-21.333333L490.666667 405.333333c0-12.8 8.533333-21.333333 21.333333-21.333333s21.333333 8.533333 21.333333 21.333333l0 320C533.333333 738.133333 524.8 746.666667 512 746.666667zM362.666667 725.333333c-12.8 0-21.333333-8.533333-21.333333-21.333333L341.333333 426.666667c0-12.8 8.533333-21.333333 21.333333-21.333333 12.8 0 21.333333 8.533333 21.333333 21.333333l0 277.333333C384 716.8 375.466667 725.333333 362.666667 725.333333z"
                            fill="rgb(119, 119, 119)"
                            stroke="rgb(119, 119, 119)"
                            stroke-width="30"
                        />
                    </svg>
                </button>
            </div>
            <div v-else class="template-item-tools">
                <button class="logo" @click="acceptTemplate(promptItem)">
                    <!-- ✔️ -->
                    <svg width="16" height="16" viewBox="0 0 1024 1024">
                        <path
                            xmlns="http://www.w3.org/2000/svg"
                            d="M511.9 959.4C303.2 959.3 122.4 814.9 76 611.5 21.4 374.1 169.8 133.9 406.4 77.6 520.5 50.4 629 66.2 731.7 122.8c13.4 7.4 19.1 20.1 15.1 33.4-3.7 12.5-15.6 21.1-28.6 19.7-5.2-0.6-10.5-2.7-15.1-5.3-40.7-22.6-83.9-38.3-130-45.1-105-15.5-201.8 6.3-288.3 68.2-85.8 61.5-139.1 145-157.7 248.9-37.4 208.5 95 401.7 292.6 449.7 208.1 50.5 417.9-75 471.2-282.5 25.9-100.7 12.7-197.2-37.4-288.5-3.4-6.2-6.2-12.5-5.4-19.7 1.4-12.6 10-22.3 21.8-24.7 12.5-2.6 23.9 2.4 30.5 13.7 13.2 22.9 24.3 47 33.1 72 87.9 248.3-55.7 517.7-310.9 583.1-36.3 9.3-73.3 13.8-110.7 13.7z m-55.7-373c2.4-3.4 3.4-5.4 4.8-6.9 65.5-65.6 131.1-131.1 196.6-196.8 7.9-7.9 16.8-12.2 28.1-9.6 15 3.3 24.5 18.2 21.2 33.2-1 4.4-3 8.6-6 12-1.4 1.6-2.9 3.1-4.4 4.6L479 640.5c-15 15-30.4 15.1-45.3 0.1-36.1-36-72.2-72-108.2-108.1-15.8-15.9-9.6-41.4 11.4-47.2 11-3.1 20.5 0.1 28.5 8.1 28.6 28.7 57.4 57.3 86.1 86.1 1.4 1.5 2.4 3.5 4.7 6.9z"
                            fill="#25B14D"
                            stroke="#25B14D"
                            stroke-width="80"
                        />
                    </svg>
                </button>
                <button class="logo" @click="rejectTemplate(promptItem)">
                    <!-- ❌ -->
                    <svg width="18" height="18" viewBox="0 0 1024 1024">
                        <path
                            xmlns="http://www.w3.org/2000/svg"
                            d="M512 451.66l90.51-90.51c16.662-16.662 43.677-16.662 60.34 0 16.662 16.663 16.662 43.678 0 60.34L572.34 512l90.51 90.51c16.662 16.662 16.662 43.677 0 60.34-16.663 16.662-43.678 16.662-60.34 0L512 572.34l-90.51 90.51c-16.662 16.662-43.677 16.662-60.34 0-16.662-16.663-16.662-43.678 0-60.34L451.66 512l-90.51-90.51c-16.662-16.662-16.662-43.677 0-60.34 16.663-16.662 43.678-16.662 60.34 0L512 451.66z m301.699-241.359c16.662 16.662 16.662 43.678 0 60.34-16.662 16.662-43.678 16.662-60.34 0-133.299-133.3-349.42-133.3-482.718 0-133.3 133.299-133.3 349.42 0 482.718 133.299 133.3 349.42 133.3 482.718 0 94.738-94.738 124.417-234.552 79.422-358.281-8.054-22.145 3.37-46.626 25.515-54.68 22.146-8.053 46.626 3.37 54.68 25.516C969.2 520.52 932.098 695.3 813.699 813.7c-166.624 166.624-436.774 166.624-603.398 0-166.624-166.624-166.624-436.774 0-603.398 166.624-166.624 436.774-166.624 603.398 0z"
                            fill="#FA5A5A"
                            stroke="#FA5A5A"
                            stroke-width="40"
                        />
                    </svg>
                </button>
            </div>
            <div class="template-container">
                <textarea
                    v-model="promptItem.key"
                    @input="onTextInput(promptItem)"
                    :class="{ 'selected-prompt': selectedTemplates.includes(promptItem) }"
                    :style="`background-color: ${rgbToRgba(
                        colorMap[promptItem.category],
                        selectedTemplates.includes(promptItem) ? 0.5 : 0.3
                    )};
                            cursor: ${promptItem.isAccepted ? 'pointer' : 'text'};`"
                    :readonly="promptItem.isAccepted"
                    @click="promptItem.isAccepted ? selectTemplate(promptItem) : () => {}"
                ></textarea>
                <div
                    class="template-changed-content"
                    v-if="
                        promptItem.isReady &&
                        promptItem.key !== promptItem.oldTemplate &&
                        (() => {
                            console.log(promptItem.isReady)
                            return true
                        })()
                    "
                    v-html="formatText(promptItem.key, promptItem.oldTemplate)"
                ></div>
            </div>
        </div>
    </TransitionGroup>
</template>

<script setup lang="ts">
import { ref, watch, computed, onMounted } from 'vue'
import { DataStore, Word, Image, Prompt, Template } from '../stores/data'
import { DistributionStore } from '../stores/distribution'
import { InfoStore } from '../stores/info'
import { config } from '../config/config'
import * as Diff from 'diff'
import eventBus from '../utils/eventBus'
import gsap from 'gsap'

const dataStore = DataStore()
const distributionStore = DistributionStore()
const infoStore = InfoStore()

const colorMap = ref(config.petsGeneratedColorMap)

const templates = ref<Template[]>([])
const displayedTemplates = ref<Template[]>([])
const removedTemplates = ref<Template[]>([])
const selectedTemplates = ref<Template[]>([])
const isSelecting = computed(() => selectedTemplates.value.length !== 0)
const isTemplateChanging = ref(false)

const totalCategories = ref<string[]>([])
const clickedCategory = ref('')
const deletedCategories = ref<string[]>([])

const categories = computed(() =>
    Array.from(new Set(templates.value.map((promptItem) => promptItem.category)))
)

const displayedCategories = computed(() =>
    Array.from(new Set(displayedTemplates.value.map((promptItem) => promptItem.category)))
)

let oldIsSelecting: boolean = false

let oldDisplayedImagesIndices: Set<number> = new Set()
let oldSelectedImagesIndices: Set<number> = new Set()
let oldDisplayedWordsIndices: Set<number> = new Set()
let oldSelectedWordsIndices: Set<number> = new Set()
let oldDisplayedTemplates: Template[] = []

watch(
    () => dataStore.images.displayedData(),
    async () => {
        if (isTemplateChanging.value) {
            return
        }
        oldRenderPrompts(dataStore.images.displayedData(), dataStore.images.selectedData())
    }
)

watch(
    () => dataStore.images.selectedData(),
    async () => {
        if (isTemplateChanging.value) {
            return
        }
        oldRenderPrompts(dataStore.images.displayedData(), dataStore.images.selectedData())
    }
)

function getTemplatesFromImages(images: Image[]): Template[] {
    const promptMap = new Map<string, { category: string; prompts: string[] }>()

    images.forEach((item) => {
        if (item.isGenerated) {
            if (!promptMap.has(item.template)) {
                promptMap.set(item.template, {
                    category: item.category,
                    prompts: [item.prompt]
                })
            } else {
                promptMap.get(item.template).prompts.push(item.prompt)
            }
        } else {
            dataStore.images
                .displayedData()
                .filter((image) => image.isGenerated && image.originalUrl === item.key)
                .forEach((image) => {
                    if (!promptMap.has(image.template)) {
                        promptMap.set(image.template, {
                            category: image.category,
                            prompts: [image.prompt]
                        })
                    } else {
                        promptMap.get(image.template).prompts.push(image.prompt)
                    }
                })
        }
    })

    return Array.from(promptMap.entries()).map(([key, value]) =>
        Template.new(key, value.category, value.prompts)
    )
}

async function oldRenderPrompts(
    images: Image[],
    selectedImages: Image[],
    emptyAsFull: boolean = true
) {
    if (isSelecting.value) {
        return
    }

    const highlightedImages = selectedImages.length === 0 && emptyAsFull ? images : selectedImages
    const newTemplates = getTemplatesFromImages(highlightedImages)

    console.log('new templates: ', newTemplates)

    if (totalCategories.value.length === 0) {
        totalCategories.value = Array.from(new Set(newTemplates.map((item) => item.category)))
    }

    templates.value = newTemplates

    displayedTemplates.value = newTemplates.filter(
        (item) =>
            removedTemplates.value.findIndex((prompt) => prompt.key === item.key) === -1 &&
            (clickedCategory.value === '' ? true : item.category === clickedCategory.value)
    )
}

function clickCategory(category: string) {
    if (displayedCategories.value.length === 1 && displayedCategories.value[0] === category) {
        displayedTemplates.value = templates.value
        clickedCategory.value = ''
    } else {
        displayedTemplates.value = templates.value.filter((item) => item.category === category)
        clickedCategory.value = category
    }
}

function rgbToRgba(rgb: string, alpha: number): string {
    const result = rgb.match(/\d+/g)
    if (!result) return rgb
    const [r, g, b] = result.map(Number)
    return `rgba(${r},${g},${b},${alpha})`
}

function acceptTemplate(template: Template) {
    if (template.isAccepted === true) {
        return
    }
    if (displayedTemplates.value.includes(template)) {
        template.isAccepted = true
        template.isReady = false
        template.oldTemplate = template.key
        template.prompts = []
    }
}

function rejectTemplate(template: Template) {
    if (template.isAccepted === true) {
        return
    }
    if (displayedTemplates.value.includes(template)) {
        template.isAccepted = true
        template.isReady = false
        template.key = template.oldTemplate
    }
}

function removeTemplate(template: Template) {
    if (removedTemplates.value.findIndex((item) => item.key === template.key) !== -1) {
        return
    }
    removedTemplates.value.push(template)
    if (getPromptCount(template.category) === 0) {
        deletedCategories.value.push(template.category)
    }
}

function onTextInput(template: Template) {
    if (displayedTemplates.value.includes(template)) {
        template.isAccepted = false
        template.isReady = true
    }
}

function editTemplate(template: Template) {
    if (displayedTemplates.value.includes(template)) {
        template.isAccepted = false
        template.isReady = true
    }
}

async function selectTemplate(template: Template) {
    if (distributionStore.isBrushing) {
        return
    }

    if (selectedTemplates.value.includes(template)) {
        selectedTemplates.value = selectedTemplates.value.filter((item) => item !== template)
    } else {
        selectedTemplates.value.push(template)
    }

    if (oldIsSelecting === false && isSelecting.value === true) {
        oldDisplayedImagesIndices = dataStore.images.displayedDataIndices
        oldSelectedImagesIndices = dataStore.images.selectedDataIndices

        oldDisplayedWordsIndices = dataStore.words.displayedDataIndices
        oldSelectedWordsIndices = dataStore.words.selectedDataIndices

        oldDisplayedTemplates = displayedTemplates.value
    } else if (oldIsSelecting === true && isSelecting.value === false) {
        dataStore.images.displayedDataIndices = oldDisplayedImagesIndices
        dataStore.images.selectedDataIndices = oldSelectedImagesIndices

        dataStore.words.displayedDataIndices = oldDisplayedWordsIndices
        dataStore.words.selectedDataIndices = oldSelectedWordsIndices

        displayedTemplates.value = oldDisplayedTemplates
    }
    oldIsSelecting = isSelecting.value

    const images = selectedTemplates.value
        .flatMap((item): string[] => item.prompts)
        .flatMap((prompt: string) => {
            return dataStore.getImagesFromPrompt(prompt)
        })

    if (images.length !== 0) {
        const words = dataStore.getCaptions(images)
        dataStore.images.select(images)
        dataStore.words.select(words)
    }

    eventBus.emit('selectImages')
    eventBus.emit('selectWords')
}

function escapeHtml(text: string): string {
    const map: { [key: string]: string } = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    }
    return text.replace(/[&<>"']/g, (m) => map[m])
}

function formatText(newText: string, originalText: string): string {
    const diff = Diff.diffWords(originalText, newText)
    return diff
        .map((part) => {
            const color = part.added ? 'green' : part.removed ? 'red' : 'black'
            const textDecoration = part.removed ? 'line-through' : 'none'
            const escapedValue = escapeHtml(part.value)
            return `<span style="color: ${color}; text-decoration: ${textDecoration}">${escapedValue}</span>`
        })
        .join('')
}

const onGeneratingData = () => {
    isTemplateChanging.value = false
}

const onAddPrompts = async (params: {
    selectedImages: Image[]
    selectedWords: Word[]
    clickedWord: string
}) => {
    isTemplateChanging.value = true
    const selectedImages = params.selectedImages
    const selectedWords = params.selectedWords
    const clickedWord = params.clickedWord

    if (infoStore.currentStep === 5) {
        const template = {
            category: 'Pug',
            prompts: [],
            oldTemplate:
                '[ a watercolor painting of | a charming cartoon illustration of | a highly detailed photorealistic painting of ] [ glowing | radiant | bold | tranquil | striking | delicate | muted | ethereal ] Pug dog [ , ] [ lounging under a blooming cherry blossom tree | romping through a vibrant tulip garden | curled up by a crackling bonfire on a beach | wandering through a lush vegetable garden | chasing bubbles in a sunlit courtyard | snoozing on a hammock under the stars | perched atop a boulder in a misty forest | relaxing on a gingham blanket under an oak tree | peeking out from a cozy dog bed by a window | frolicking through a field of wildflowers ]',
            key: '[ a watercolor painting of | a charming cartoon illustration of | a highly detailed photorealistic painting of ] [ glowing | radiant | bold | tranquil | striking | delicate | muted | ethereal ] Pug dog [ , ] [ lounging under a blooming cherry blossom tree | romping through a vibrant tulip garden | curled up by a crackling bonfire on a beach | wandering through a lush vegetable garden | chasing bubbles in a sunlit courtyard | snoozing on a hammock under the stars | perched atop a boulder in a misty forest | relaxing on a gingham blanket under an oak tree | peeking out from a cozy dog bed by a window | frolicking through a field of wildflowers ]',
            isAccepted: false,
            isChecked: false,
            isSelecting: false,
            isReady: false
        }

        templates.value.unshift(template)
        displayedTemplates.value.unshift(template)

        return
    }

    eventBus.emit('changeStep', 5)

    const template = {
        category: 'Bengal',
        prompts: [],
        oldTemplate:
            '[ a whimsical digital painting of | a playful cartoon image of | a detailed realistic painting of ] [ bright | vivid | colorful | serene | high-contrast | soft | pastel | dreamy ] Bengal cat[ , ] [ resting on a sunlit porch | playing in a vibrant flower bed | lounging on a fluffy rug by a fireplace | exploring a colorful backyard | chasing butterflies in a sunny meadow | napping on a garden swing | climbing a tree in a lush forest | sitting on a picnic blanket in a sunny park | peeking out from a cozy cat house | playing hide and seek among autumn leaves | resting on a sunlit porch | playing in a vibrant flower bed | lounging on a fluffy rug by a fireplace | exploring a colorful backyard | chasing butterflies in a sunny meadow | napping on a garden swing | climbing a tree in a lush forest | sitting on a picnic blanket in a sunny park | peeking out from a cozy cat house | playing hide and seek among autumn leaves ]',
        key: '[ a whimsical digital painting of | a playful cartoon image of | a detailed realistic painting of ] [ bright | vivid | colorful | serene | high-contrast | soft | pastel | dreamy ] Bengal cat [ , ] [ resting on a sunlit porch | playing in a vibrant flower bed | lounging on a fluffy rug by a fireplace | exploring a colorful backyard | chasing butterflies in a sunny meadow | napping on a garden swing | climbing a tree in a lush forest | sitting on a picnic blanket in a sunny park | peeking out from a cozy cat house | playing hide and seek among autumn leaves | resting on a sunlit porch | playing in a vibrant flower bed | lounging on a fluffy rug by a fireplace | exploring a colorful backyard | chasing butterflies in a sunny meadow | napping on a garden swing | climbing a tree in a lush forest | sitting on a picnic blanket in a sunny park | peeking out from a cozy cat house | playing hide and seek among autumn leaves ]',
        isAccepted: false,
        isChecked: false,
        isSelecting: false,
        isReady: false
    }

    templates.value.unshift(template)
    displayedTemplates.value.unshift(template)
}

const onRemovePrompts = async (params: {
    selectedImages: Image[]
    selectedWords: Word[]
    clickedWord: string
}) => {
    isTemplateChanging.value = true
    const selectedImages = params.selectedImages
    const selectedWords = params.selectedWords
    const clickedWord = params.clickedWord

    if (clickedWord == 'tiger') {
        eventBus.emit('changeStep', 4)
        const template = templates.value.find((item) => item.category == 'Bengal')
        template.isReady = false
        template.isAccepted = false

        let words =
            '[ a whimsical digital painting of | a whimsical cartoon image of | a charming illustration of | a vibrant digital illustration of | a playful cartoon image of | a realistic photo of ] [ colorful | bright | high-contrast | soft-toned | serene | lively ] Bengal cat, playfully interacting with its surroundings in a sunny garden'

        template.key = ''

        const wordArray = words.split(' ')

        const tl = gsap.timeline()
        const duration = 1
        const eachDuration = duration / wordArray.length
        wordArray.forEach((word, index) => {
            tl.to(
                {},
                {
                    duration: eachDuration,
                    onComplete: () => {
                        template.key += word + ' '
                        console.log(`Added word: ${word}`)
                    }
                }
            )
        })
        setTimeout(() => {
            template.isReady = true
        }, duration * 1000)
    }
}

function getPromptCount(category: string): number {
    return displayedTemplates.value.filter(
        (item) => item.category === category && !removedTemplates.value.includes(item)
    ).length
}

async function init() {
    selectedTemplates.value = []
    isTemplateChanging.value = false
    await oldRenderPrompts(dataStore.images.displayedData(), dataStore.images.selectedData())
}

onMounted(async () => {
    await init()
    eventBus.on('addPrompts', onAddPrompts)
    eventBus.on('removePrompts', onRemovePrompts)
    eventBus.on('generateDataComplete', onGeneratingData)
    eventBus.on('clearSelection', onGeneratingData)
})
</script>

<style scoped>
.categories {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    column-gap: 8px;
    row-gap: 8px;
}

.category-button {
    min-width: 50px;
    padding: 1px 10px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 500;
}

.templates {
    display: flex;
    flex: 1;
    flex-direction: column;
    row-gap: 16px;
    margin-top: 8px;
    width: 100%;
    overflow: scroll;
}

.template-item {
    display: grid;
    grid-template-columns: 1fr 24fr;
    gap: 8px;
}

.template-item-tools {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0;
}

.template-container {
    position: relative;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    justify-content: center;
}

.template-container textarea {
    box-sizing: content-box;
    color: #000;
    font-size: 14px;
    padding: 4px 10px;
    border-radius: 10px;
    line-height: 21px;
    height: calc(21px * 4 + 8px);
    overflow: auto;
}

.add-template-button {
    opacity: 0;
    position: absolute;
    top: calc(100% - 10px);
    left: 50%;
    transform: translateX(-50%);
    font-size: 12px;
    padding: 1px 6px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: opacity 0.3s ease-in-out;
}

.add-template-button:hover {
    opacity: 1;
}

.selected-prompt {
    border: 2px solid black;
}

.template-changed-content {
    flex-grow: 1;
    padding: 0 10px;
    font-size: 14px;
}

.list-move,
.list-enter-active,
.list-leave-active {
    transition: all 0.8s ease-in-out;
}

.list-enter-from,
.list-leave-to {
    opacity: 0;
    transform: translateX(30px);
}

.list-leave-active {
    position: absolute;
    width: 93.2%;
}
</style>
