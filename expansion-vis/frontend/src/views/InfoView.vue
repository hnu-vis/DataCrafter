<template>
    <div class="info-container">
        <div class="controls">
            <label for="dataset-select" class="title-label">DATASET</label>
            <select
                id="dataset-select"
                v-model="dataset"
                class="dataset-select"
                @change="changeDataset"
            >
                <option value="pets">PETS</option>
                <option value="COCO">COCO</option>
                <option value="OCT">OCT</option>
                <option value="Quality">Quality</option>
            </select>
        </div>
        <div class="generation-controls">
            <label for="total-generation" class="title-label">Generated multiples</label>
            <div class="generation-input">
                <input
                    type="number"
                    id="total-generation"
                    v-model="iterationNum"
                    min="1"
                    class="total-generation-input"
                />
                <button
                    :style="buttonStyle"
                    @click="generateData(dataset, guidanceScale, iterationNum, infoStore.currentStep)"
                    class="generate-button"
                >
                    Generate
                </button>
            </div>
        </div>
    </div>
    <div class="chart-container">
        <header>
            <div>
                <p>Original Images</p>
                <svg width="20" height="20">
                    <circle cx="10" cy="10" r="6" fill="rgb(180, 180, 180)" />
                </svg>
            </div>
            <p>Total Number: {{ originalImagesNum.toFixed(0) }}</p>
        </header>
        <div ref="originalChartRef" class="chart"></div>
    </div>
    <div class="chart-container">
        <header>
            <div>
                <p>Generated Images</p>
                <svg width="20" height="20">
                    <circle
                        cx="10"
                        cy="10"
                        r="6"
                        stroke="rgb(180, 180, 180)"
                        stroke-width="2"
                        fill="none"
                    />
                </svg>
            </div>
            <p>Total Number: {{ generatedImagesNum.toFixed(0) }}</p>
        </header>
        <div ref="generatedChartRef" class="chart"></div>
    </div>

    <div class="word-cloud">
        <header class="word-cloud-header">
            <h2>Content Labels</h2>
            <div class="word-cloud-tools">
                <div class="word-cloud-sort-tool">
                    <button @click="toggleReverse">
                        <svg
                            t="1723796523002"
                            class="icon"
                            viewBox="0 0 1024 1024"
                            version="1.1"
                            xmlns="http://www.w3.org/2000/svg"
                            p-id="9794"
                            xmlns:xlink="http://www.w3.org/1999/xlink"
                            width="16"
                            height="16"
                        >
                            <path
                                d="M608 981.333333c-4.266667 0-8.533333 0-10.666667-2.133333-12.8-4.266667-21.333333-17.066667-21.333333-29.866667v-874.666666c0-17.066667 14.933333-32 32-32s32 14.933333 32 32v789.333333l157.866667-179.2c12.8-12.8 32-14.933333 44.8-2.133333 12.8 12.8 14.933333 32 2.133333 44.8l-213.333333 243.2c-6.4 6.4-14.933333 10.666667-23.466667 10.666666zM416 981.333333c-17.066667 0-32-14.933333-32-32v-789.333333l-157.866667 179.2c-12.8 12.8-32 14.933333-44.8 2.133333-12.8-10.666667-14.933333-32-2.133333-44.8l213.333333-243.2c10.666667-12.8 32-14.933333 44.8-2.133333 6.4 4.266667 10.666667 14.933333 10.666667 23.466667v874.666666c0 17.066667-14.933333 32-32 32z"
                                fill="#58a6cd"
                                p-id="9795"
                            ></path>
                        </svg>
                    </button>
                    <select v-model="sortKey">
                        <option value="total">Frequency</option>
                        <option value="G/O ratio">G/O Ratio</option>
                    </select>
                </div>
                <button v-if="wordCloudState === 'All'" @click="clearAll">
                    <svg
                        t="1723790290245"
                        class="icon"
                        viewBox="0 0 1024 1024"
                        version="1.1"
                        xmlns="http://www.w3.org/2000/svg"
                        p-id="9640"
                        xmlns:xlink="http://www.w3.org/1999/xlink"
                        width="20"
                        height="20"
                    >
                        <path
                            d="M800 938.666667h-576C147.2 938.666667 85.333333 876.8 85.333333 800v-576C85.333333 147.2 147.2 85.333333 224 85.333333h573.866667c76.8 0 138.666667 61.866667 138.666666 138.666667v573.866667c2.133333 78.933333-59.733333 140.8-136.533333 140.8z m-576-789.333334C183.466667 149.333333 149.333333 183.466667 149.333333 224v573.866667c0 40.533333 34.133333 74.666667 74.666667 74.666666h573.866667c40.533333 0 74.666667-34.133333 74.666666-74.666666V224C874.666667 183.466667 840.533333 149.333333 800 149.333333h-576z"
                            fill="#58a6cd"
                            p-id="9641"
                        ></path>
                        <path
                            d="M469.333333 646.4c-14.933333 0-27.733333-6.4-38.4-14.933333L320 518.4c-14.933333-10.666667-19.2-29.866667-8.533333-44.8 10.666667-14.933333 29.866667-19.2 44.8-8.533333 2.133333 2.133333 6.4 4.266667 8.533333 8.533333l106.666667 104.533333 189.866666-189.866666c14.933333-10.666667 34.133333-6.4 44.8 8.533333 8.533333 10.666667 6.4 25.6 0 36.266667l-198.4 198.4c-10.666667 8.533333-23.466667 14.933333-38.4 14.933333z"
                            fill="#58a6cd"
                            p-id="9642"
                        ></path>
                    </svg>
                </button>
                <button v-else-if="wordCloudState === 'Some'" @click="selectAll">
                    <svg
                        t="1723790286996"
                        class="icon"
                        viewBox="0 0 1024 1024"
                        version="1.1"
                        xmlns="http://www.w3.org/2000/svg"
                        p-id="9486"
                        xmlns:xlink="http://www.w3.org/1999/xlink"
                        width="20"
                        height="20"
                    >
                        <path
                            d="M800 938.666667h-576C147.2 938.666667 85.333333 876.8 85.333333 800v-576C85.333333 147.2 147.2 85.333333 224 85.333333h573.866667c76.8 0 138.666667 61.866667 138.666666 138.666667v573.866667c2.133333 78.933333-59.733333 140.8-136.533333 140.8z m-576-789.333334C183.466667 149.333333 149.333333 183.466667 149.333333 224v573.866667c0 40.533333 34.133333 74.666667 74.666667 74.666666h573.866667c40.533333 0 74.666667-34.133333 74.666666-74.666666V224C874.666667 183.466667 840.533333 149.333333 800 149.333333h-576z"
                            fill="#58a6cd"
                            p-id="9487"
                        ></path>
                        <path
                            d="M693.333333 544h-362.666666c-17.066667 0-32-14.933333-32-32s14.933333-32 32-32h362.666666c17.066667 0 32 14.933333 32 32s-14.933333 32-32 32z"
                            fill="#58a6cd"
                            p-id="9488"
                        ></path>
                    </svg>
                </button>
                <button v-else @click="selectAll">
                    <svg
                        t="1723790281600"
                        class="icon"
                        viewBox="0 0 1024 1024"
                        version="1.1"
                        xmlns="http://www.w3.org/2000/svg"
                        p-id="9333"
                        xmlns:xlink="http://www.w3.org/1999/xlink"
                        width="20"
                        height="20"
                    >
                        <path
                            d="M800 938.666667h-576C147.2 938.666667 85.333333 876.8 85.333333 800v-576C85.333333 147.2 147.2 85.333333 224 85.333333h573.866667c76.8 0 138.666667 61.866667 138.666666 138.666667v573.866667c2.133333 78.933333-59.733333 140.8-136.533333 140.8z m-576-789.333334C183.466667 149.333333 149.333333 183.466667 149.333333 224v573.866667c0 40.533333 34.133333 74.666667 74.666667 74.666666h573.866667c40.533333 0 74.666667-34.133333 74.666666-74.666666V224C874.666667 183.466667 840.533333 149.333333 800 149.333333h-576z"
                            fill="#d1d1d1"
                            p-id="9334"
                        ></path>
                    </svg>
                </button>
            </div>
        </header>
        <TransitionGroup name="list" class="word-cloud-content" tag="div">
            <button
                v-for="(word, index) in dataStore.sortedWords"
                :key="word.key"
                :class="
                    dataStore.words.displayedDataContains(word)
                        ? `displayed-word`
                        : `not-displayed-word`
                "
                @click="toggleDisplayedWord(word)"
            >
                {{ word.key }} ({{ formatToFixed(dataStore.compareWords(word)) }})
            </button>
        </TransitionGroup>
    </div>
</template>

<script setup lang="ts">
import {
    ref,
    onMounted,
    computed,
    watch,
    StyleValue,
    onUnmounted,
    onBeforeMount
} from 'vue'

import axios from 'axios'
import * as echarts from 'echarts'
import eventBus from '../utils/eventBus'
import gsap from 'gsap'

import { config } from '../config/config'

import { getImagesData, getWordsData, getMetricData } from '../apis/data'

import {
    DataStore,
    Word,
    Image,
    MetricData,
    WordsData,
    ImagesData,
    GORatioMethod,
    GFRatioMethod,
    OFRatioMethod,
    originalNumMethod,
    generatedNumMethod,
    frequencyMethod
} from '../stores/data'
import { DistributionStore } from '../stores/distribution'
import { formatToFixed } from '../utils/formatter'
import { InfoStore } from '../stores/info'

const colorMap = config.petsColorMap
const generatedColorMap = config.petsGeneratedColorMap

const dataStore = DataStore()
const distributionStore = DistributionStore()
const infoStore = InfoStore()

const dataset = ref('pets')

const changeDataset = (value: Event) => {
    eventBus.emit('changeDataset')
}

const guidanceScale = ref(20)
const iterationNum = ref(2)

const changeStep = async (dataset: string, step: number) => {
    try {
        const stepData = { step }
        const datasetData = { case: dataset }

        const stepResponse = await axios.post(`${config.baseURL}/changeStep`, stepData)
        const datasetResponse = await axios.post(`${config.baseURL}/changeCase`, datasetData)

        return [stepResponse.data, datasetResponse.data]
    } catch (error) {
        console.error('Error:', error.response ? error.response.data : error.message)
        throw error
    }
}

watch(dataset, (value) => {
    if (value === 'Quality') {
        console.log('change to Quality')
        distributionStore.showPie = false
        distributionStore.wordMode = '☆'
    } else {
        distributionStore.showPie = true
        distributionStore.wordMode = 'default'
    }
})

let originalChart: echarts.ECharts
let generatedChart: echarts.ECharts

const originalChartRef = ref<HTMLDivElement | null>(null)
const generatedChartRef = ref<HTMLDivElement | null>(null)

const categories = ref<string[]>([])

const originalImagesNum = ref(0)

watch(
    () => dataStore.originalImages.selectedDataNum(),
    () => {
        const originalNum = dataStore.images
            .selectedData()
            .filter((image) => !image.isGenerated).length
        gsap.to(originalImagesNum, {
            duration: 0.5,
            value: originalNum
        })
    }
)

const generatedImagesNum = ref(0)

watch(
    () => dataStore.generatedImages.selectedDataNum(),
    () => {
        const generatedNum = dataStore.images.selectedData().filter((image) => image.isGenerated).length
        gsap.to(generatedImagesNum, {
            duration: 0.5,
            value: generatedNum
        })
    }
)

const option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            rotate: -45,
            interval: 0,
            textStyle: {
                fontWeight: 'bold',
                color: 'black',
                fontSize: 10
            },
            axisTick: {
                show: false
            },
            axisLine: {
                show: false
            },
            verticalAlign: 'start'
        }
    },
    yAxis: {
        show: false,
        min: 0,
        max: 200
    },
    grid: {
        show: false,
        left: '-3%',
        right: '7%',
        top: '5%',
        bottom: '0%',
        containLabel: true
    },
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        },
        formatter: function (params: any[]) {
            let result = params[0].name + '<br/>'
            params.forEach((item) => {
                result += 'number: ' + item.value + '<br/>'
            })
            return result
        }
    },
    animation: true,
    animationDuration: 2000
}

const getOriginalDataMap = (images: Image[]) => {
    const dataMap = categories.value.reduce((map, category) => {
        map[category] = 0
        return map
    }, {})

    images.forEach((image) => {
        if (image.isGenerated) {
            return
        }
        dataMap[image.category] += 1
    })

    return dataMap
}

const getGeneratedDataMap = (images: Image[]) => {
    const dataMap = categories.value.reduce((map, category) => {
        map[category] = 0
        return map
    }, {})

    images.forEach((image) => {
        if (!image.isGenerated) {
            return
        }
        dataMap[image.category] += 1
    })

    return dataMap
}

const renderOriginalChart = async (images: Image[]) => {
    const dataMap = getOriginalDataMap(images)

    const data = Object.entries(dataMap).map(([category, value]) => ({
        value: value,
        name: category,
        itemStyle: {
            color: colorMap[category]
        }
    }))

    originalChart &&
        originalChart.setOption({
            series: [{ type: 'bar', data: data }]
        })
}

const renderGeneratedChart = async (images: Image[]) => {
    const dataMap = getGeneratedDataMap(images)

    const data = Object.entries(dataMap).map(([category, value]) => ({
        value: value,
        name: category,
        itemStyle: {
            color: generatedColorMap[category]
        }
    }))

    generatedChart &&
        generatedChart.setOption({
            series: [{ type: 'bar', data: data }]
        })
}

watch(
    () => dataStore.images.selectedData(),
    async () => {
        const images = dataStore.images.selectedData()

        const originalDataMap = getOriginalDataMap(images)
        const generatedDataMap = getGeneratedDataMap(images)

        renderOriginalChart(images)
        renderGeneratedChart(images)

        let maxValue = 0

        Object.values(originalDataMap).forEach((value: number) => {
            if (value >= maxValue) {
                maxValue = value
            }
        })

        Object.values(generatedDataMap).forEach((value: number) => {
            if (value >= maxValue) {
                maxValue = value
            }
        })

        setTimeout(() => {
            if (maxValue > 200) {
                maxValue = maxValue
            } else if (maxValue > 100) {
                maxValue = 200
            } else if (maxValue > 50) {
                maxValue = 100
            } else if (maxValue > 30) {
                maxValue = 50
            } else if (maxValue > 10) {
                maxValue = 30
            } else {
                maxValue = 10
            }

            originalChart.setOption({
                yAxis: {
                    show: false,
                    min: 0,
                    max: maxValue / 0.8
                }
            })
            generatedChart.setOption({
                yAxis: {
                    show: false,
                    min: 0,
                    max: maxValue / 0.8
                }
            })
        }, 500)
    }
)

const displayedWords = ref<Word[]>([])

const generateData = async (
    dataset: string,
    guidanceScale: number,
    iterationStepNum: number,
    step: number
) => {
    if (iterationStepNum == 0) {
        infoStore.currentStep = 0
        dataStore.reset()
        categories.value = []
        displayedWords.value = []
        return
    }
    if (step != 4 && step != 5 && step != 6 && step != 7) {
        if (guidanceScale == 5) {
            step = 1
        } else if (guidanceScale == 100) {
            step = 2
        } else if (guidanceScale == 20) {
            step = 3
        } else {
            step = iterationStepNum
        }
    }

    const metricKey = {
        option: 'epoch',
        name: 'OCT'
    }

    try {
        eventBus.emit('generateDataStart')
        await changeStep(dataset, step)
        const imagesData: ImagesData = await getImagesData()
        const wordsData: WordsData = await getWordsData()
        const rawMetricData = await getMetricData(metricKey)

        const CmmdData = Object.values(rawMetricData).at(-1)['CMMD']
        const DiversityData = Object.values(rawMetricData).at(-1)['Diversity']
        const InformativenessData = Object.values(rawMetricData).at(-1)['Informativeness']
        const metricData: MetricData = {
            Distance: CmmdData,
            Diversity: DiversityData,
            Informativeness: InformativenessData
        }

        dataStore.setData(imagesData, wordsData, metricData)
        eventBus.emit('generateDataComplete')

        categories.value = Array.from(dataStore.categories)
        displayedWords.value = dataStore.sortedWords

        originalChart.setOption({ xAxis: { data: categories.value } })
        generatedChart.setOption({ xAxis: { data: categories.value } })

        const images = dataStore.images.selectedData()
        await renderOriginalChart(images)
        await renderGeneratedChart(images)
    } catch (error) {
        console.error('Error sending data to backend:', error)
    }
}

const progress = ref(100)
let timerId: any = null

const handleGenerateDataStart = () => {
    progress.value = 0

    if (timerId) {
        clearInterval(timerId)
    }

    timerId = setInterval(() => {
        if (progress.value < 100) {
            progress.value += 10
            console.log(progress.value)
        } else {
            clearInterval(timerId)
        }
    }, 500)
}

const handleGenerateDataComplete = () => {
    if (timerId) {
        clearInterval(timerId)
    }
    progress.value = 100
    console.log(progress.value)
}

const buttonStyle = computed<StyleValue>(() => ({
    transition: `background 0.5s ease`,
    background: `linear-gradient(to right, #ffffff 0%, #ffffff ${progress.value}%, #eeeeee ${progress.value}%, #eeeeee 100%)`
}))

const wordCloudState = computed<'All' | 'Some' | 'None'>(() => {
    const displayedLength = dataStore.words.displayedDataIndices?.size
    const totalLength = dataStore.words.data?.length
    if (displayedLength === totalLength) {
        return 'All'
    } else if (displayedLength === 0) {
        return 'None'
    } else {
        return 'Some'
    }
})

function selectAll() {
    dataStore.words.displayAll()
    dataStore.words.selectAll()
    eventBus.emit('displayWords')
}

function clearAll() {
    dataStore.words.displayNone()
    dataStore.words.selectNone()
    eventBus.emit('displayWords')
}

const sortKey = ref('total')

const toggleReverse = () => {
    dataStore.isReversed = !dataStore.isReversed
    eventBus.emit('selectWords')
}

watch(sortKey, (newSortKey) => {
    if (newSortKey === 'generatedNum') {
        dataStore.compareWords = generatedNumMethod(dataStore.images.selectedData())
    } else if (newSortKey === 'originalNum') {
        dataStore.compareWords = originalNumMethod(dataStore.images.selectedData())
    } else if (newSortKey === 'G/O ratio') {
        dataStore.compareWords = GORatioMethod(dataStore.images.selectedData())
    } else if (newSortKey === 'total') {
        dataStore.compareWords = frequencyMethod(dataStore.images.selectedData())
    } else if (newSortKey === 'G/F ratio') {
        dataStore.compareWords = GFRatioMethod(dataStore.images.selectedData())
    } else if (newSortKey === 'O/F ratio') {
        dataStore.compareWords = OFRatioMethod(dataStore.images.selectedData())
    }
    eventBus.emit('selectWords')
})

const handleSelectImages = () => {
    if (sortKey.value === 'generatedNum') {
        dataStore.compareWords = generatedNumMethod(dataStore.images.selectedData())
    } else if (sortKey.value === 'originalNum') {
        dataStore.compareWords = originalNumMethod(dataStore.images.selectedData())
    } else if (sortKey.value === 'G/O ratio') {
        dataStore.compareWords = GORatioMethod(dataStore.images.selectedData())
    } else if (sortKey.value === 'total') {
        dataStore.compareWords = frequencyMethod(dataStore.images.selectedData())
    } else if (sortKey.value === 'G/F ratio') {
        dataStore.compareWords = GFRatioMethod(dataStore.images.selectedData())
    } else if (sortKey.value === 'O/F ratio') {
        dataStore.compareWords = OFRatioMethod(dataStore.images.selectedData())
    }
}

const toggleDisplayedWord = (word: string | Word) => {
    dataStore.words.toggleItemInDisplayedData(word)
    dataStore.words.addItemToSelectedData(word)
    eventBus.emit('displayWords')
}

onBeforeMount(() => {
    eventBus.on('changeStep', (step: number) => {
        infoStore.currentStep = step
    })
    eventBus.on('generateDataStart', handleGenerateDataStart)
    eventBus.on('generateDataComplete', handleGenerateDataComplete)
    eventBus.on('selectImages', handleSelectImages)
    eventBus.on('displayImages', handleSelectImages)
})

onMounted(async () => {
    if (timerId) {
        clearInterval(timerId)
    }
    dataStore.reset()

    originalChart = echarts.init(originalChartRef.value, null, {
        renderer: 'svg',
        useDirtyRect: true
    })
    generatedChart = echarts.init(generatedChartRef.value, null, {
        renderer: 'svg',
        useDirtyRect: true
    })

    originalChart.setOption(option)
    generatedChart.setOption(option)

    await generateData(dataset.value, guidanceScale.value, iterationNum.value, infoStore.currentStep)
})

onUnmounted(async () => {
    if (timerId) {
        clearInterval(timerId)
    }
    eventBus.off('changeStep')
    eventBus.off('generateDataStart')
    eventBus.off('generateDataComplete')
})
</script>

<style scoped>
button {
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
}

button svg {
    display: block;
    margin: auto;
}

.generation-controls {
    margin-top: 5px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.title-label {
    font-size: 14px;
    font-weight: bold;
    line-height: 20px;
    color: #888;
}

.slider-container {
    position: relative;
    font-size: 15px;
    width: 100%;
    display: flex;
    justify-content: space-between;
    gap: 10px;
}

.slider-wrapper {
    position: relative;
    flex: 1;
}

.guidance-slider {
    appearance: none;
    width: 100%;
    height: 8px;
    background: rgb(220, 220, 220);
    border-radius: 15px;
}

.guidance-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 13px;
    height: 13px;
    background: rgb(119, 119, 119);
    cursor: pointer;
    border-radius: 50%;
    box-shadow: 0 0 2px rgba(0, 0, 0, 0.2);
}

.slider-value {
    position: absolute;
    top: -25px;
    font-size: 12px;
    font-weight: bold;
    color: #b4b4b4;
    background-color: white;
    padding: 2px 5px;
    border-radius: 4px;
    border: 1px solid #ccc;
    white-space: nowrap;
    visibility: hidden;
}

.slider-wrapper:hover .slider-value {
    visibility: visible;
}

.generation-input {
    display: flex;
    gap: 15px;
}

.total-generation-input {
    width: 80%;
    height: 30px;
    font-size: 15px;
    padding: 5px;
    border: 1px solid #ccc;
    border-radius: 4px;
}

.generate-button {
    padding: 0 16px;
    height: 30px;
    line-height: 30px;
    font-size: 14px;
    font-weight: bold;
    border-radius: 4px;
    transition: background 0.5s ease;
}

.word-cloud {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: hidden;
    border: 1px solid rgba(0, 0, 0, 0.1);
    background-color: white;
    border-radius: 10px;
    padding: 5px 12px;
    row-gap: 8px;
}

.word-cloud-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.word-cloud-header > h2 {
    font-size: 13px;
    font-weight: bold;
    color: black;
    vertical-align: center;
}

.word-cloud-tools {
    display: flex;
    align-items: center;
    column-gap: 16px;
}

.word-cloud-sort-tool {
    display: flex;
    align-items: center;
    column-gap: 4px;
}

.word-cloud-sort-tool select {
    font-size: 12px;
    padding: 0 32px 0 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background-color: white;
    background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='12' fill='currentColor' class='bi bi-chevron-down' viewBox='0 0 16 16'%3E%3Cpath fill-rule='evenodd' d='M1.646 6.646a.5.5 0 0 1 .708 0L8 12.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 8px center;
    background-size: 8px 8px;
}

.word-cloud-tools button {
    border: none;
}

.word-cloud-content {
    flex: 1;
    display: flex;
    flex-wrap: wrap;
    overflow-y: auto;
    row-gap: 8px;
    column-gap: 8px;
}

.word-cloud-content button {
    font-size: 13px;
    font-weight: bold;
    color: #242424;
    background-color: #ffffff;
    border: 1px solid #d1d1d1;
    border-radius: 20px;
    padding: 0 5px;
}

.word-cloud-content button:hover {
    background-color: #f5f5f5;
    border-color: #c7c7c7;
}

.chart-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    border: 1px solid rgba(0, 0, 0, 0.1);
    background-color: white;
    border-radius: 10px;
    padding: 5px 12px;
}

.chart-container header {
    display: grid;
    grid-template-columns: 1fr 1fr;
    align-items: center;
}

.chart-container header * {
    font-size: 13px;
    font-weight: bold;
    line-height: 20px;
    color: black;
}

.chart-container header > :nth-child(1) {
    display: flex;
    column-gap: 8px;
}

.chart-container header > :nth-child(2) {
    justify-self: end;
}

.chart-container .chart {
    flex: 1;
    width: 100%;
    height: 100%;
}

.info-container {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 0;
}

.controls {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    margin-bottom: 8px;
}

.dataset-select {
    width: 100%;
    font-size: 14px;
    padding: 5px;
    line-height: 30px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background-color: white;
    appearance: none;
    background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='12' fill='currentColor' class='bi bi-chevron-down' viewBox='0 0 16 16'%3E%3Cpath fill-rule='evenodd' d='M1.646 6.646a.5.5 0 0 1 .708 0L8 12.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 8px center;
    background-size: 16px 16px;
}

.dataset-label {
    font-size: 13px;
    font-weight: bold;
    color: #888;
}

button.generate-button {
    background-color: #ffffff;
    border: 1px solid #d1d1d1;
    color: #242424;
}

button.generate-button:hover {
    background-color: #f5f5f5;
    border-color: #c7c7c7;
}

button.displayed-word {
    opacity: 1;
}

button.not-displayed-word {
    opacity: 0.4;
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
}
</style>
